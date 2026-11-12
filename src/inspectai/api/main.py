
import logging
import math
import os
import sqlite3
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from functools import partial

import anyio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from inspectai.inference.predictor import InspectAIPredictor
from inspectai.inspection.pipe import inspect_pipe
from inspectai.inspection.video import inspect_video
from inspectai.storage.history import (
    get_inspection,
    initialize_database,
    list_inspections,
    save_inspection,
)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT = (
    PROJECT_ROOT
    / "reports/patchcore_384/Patchcore/MVTecAD/"
    / "bottle/v0/weights/lightning/model.ckpt"
)

REFERENCE_CLOUD = (
    PROJECT_ROOT
    / "data/processed/pointclouds/reference_pipe.ply"
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_POINTCLOUD_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024

MAX_IMAGE_PIXELS = 25_000_000

predictor = None

# Limit concurrent expensive inference operations.
inference_limiter = anyio.CapacityLimiter(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    initialize_database()

    predictor = await anyio.to_thread.run_sync(
        InspectAIPredictor,
        str(CHECKPOINT),
    )

    try:
        yield
    finally:
        predictor = None


app = FastAPI(
    title="InspectAI API",
    description="Industrial anomaly detection using PatchCore",
    version="0.2.0",
    lifespan=lifespan,
)


def safe_filename(filename: str | None) -> str:
    """Remove directory components from uploaded filenames."""
    if not filename:
        return "unnamed"

    cleaned = filename.replace("\\", "/").split("/")[-1]
    cleaned = "".join(
        character
        for character in cleaned
        if character.isprintable()
        and character not in "\r\n\t"
    )

    return cleaned[:255] or "unnamed"


async def read_upload(
    file: UploadFile,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read an upload with a strict size limit."""
    data = await file.read(max_bytes + 1)

    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{label} exceeds the upload size limit",
        )

    if not data:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded {label.lower()} is empty",
        )

    return data


def save_report(
    inspection_type: str,
    filename: str,
    decision: str,
    report: dict,
    anomaly_score: float | None = None,
):
    """Persist a completed inspection."""
    save_inspection(
        inspection_type=inspection_type,
        filename=filename,
        decision=decision,
        anomaly_score=anomaly_score,
        report=report,
    )


async def execute_inspection(function, *args, **kwargs):
    task = partial(function, *args, **kwargs)

    return await anyio.to_thread.run_sync(
        task,
        limiter=inference_limiter,
    )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
    }


@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready",
        )

    if file.content_type not in {
        "image/png",
        "image/jpeg",
    }:
        raise HTTPException(
            status_code=415,
            detail="Only PNG and JPEG images are supported",
        )

    filename = safe_filename(file.filename)

    image_bytes = await read_upload(
        file,
        MAX_IMAGE_BYTES,
        "Image",
    )

    temporary_path = None

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("Unsupported image format")

            width, height = image.size

            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions are too large")

            image.verify()

        with NamedTemporaryFile(
            suffix=".png",
            delete=False,
        ) as temp:
            temporary_path = Path(temp.name)

        with Image.open(BytesIO(image_bytes)) as image:
            image.convert("RGB").save(temporary_path)

        result = await execute_inspection(
            predictor.predict,
            str(temporary_path),
        )

        mask = result["predicted_mask"]

        score = float(result["anomaly_score"])

        if not math.isfinite(score):
            raise RuntimeError("Non-finite anomaly score")

        report = {
            "filename": filename,
            "classification": result["classification"],
            "is_anomalous": bool(result["is_anomalous"]),
            "anomaly_score": score,
            "defective_pixels": int(mask.sum()),
            "mask_height": int(mask.shape[0]),
            "mask_width": int(mask.shape[1]),
            "defect_area_percentage": round(
                100 * mask.sum() / mask.size,
                2,
            ),
        }

        await anyio.to_thread.run_sync(
            save_report,
            "2D Image",
            filename,
            report["classification"],
            report,
            score,
        )

        return report

    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        ValueError,
        OSError,
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image",
        )

    except sqlite3.Error:
        logger.exception("Image history persistence failed")
        raise HTTPException(
            status_code=503,
            detail="Inspection history storage unavailable",
        )

    except Exception:
        logger.exception("Image inspection failed")
        raise HTTPException(
            status_code=500,
            detail="Image inspection failed",
        )

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.post("/inspect/3d")
async def inspect_pointcloud(
    file: UploadFile = File(...),
    tolerance_mm: float = Form(1.0),
):
    if not math.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise HTTPException(
            status_code=422,
            detail="Tolerance must be positive and finite",
        )

    filename = safe_filename(file.filename)

    if Path(filename).suffix.lower() != ".ply":
        raise HTTPException(
            status_code=415,
            detail="Only PLY point clouds are supported",
        )

    if not REFERENCE_CLOUD.is_file():
        raise HTTPException(
            status_code=503,
            detail="Reference point cloud is unavailable",
        )

    cloud_bytes = await read_upload(
        file,
        MAX_POINTCLOUD_BYTES,
        "Point cloud",
    )

    if not (
        cloud_bytes.startswith(b"ply\n")
        or cloud_bytes.startswith(b"ply\r\n")
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid PLY file header",
        )

    temporary_path = None

    try:
        with NamedTemporaryFile(
            suffix=".ply",
            delete=False,
        ) as temp:
            temporary_path = Path(temp.name)
            temp.write(cloud_bytes)

        report = await execute_inspection(
            inspect_pipe,
            reference_path=REFERENCE_CLOUD,
            scan_path=temporary_path,
            tolerance_mm=tolerance_mm,
        )

        report["scan_file"] = filename
        report["reference_file"] = REFERENCE_CLOUD.name

        await anyio.to_thread.run_sync(
            save_report,
            "3D Point Cloud",
            filename,
            report["quality_control"]["decision"],
            report,
        )

        return report

    except (ValueError, RuntimeError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported point cloud",
        )

    except sqlite3.Error:
        logger.exception("3D history persistence failed")
        raise HTTPException(
            status_code=503,
            detail="Inspection history storage unavailable",
        )

    except Exception:
        logger.exception("3D inspection failed")
        raise HTTPException(
            status_code=500,
            detail="3D inspection failed",
        )

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.post("/inspect/video")
async def inspect_uploaded_video(
    file: UploadFile = File(...),
    sample_interval_seconds: float = Form(1.0),
):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready",
        )

    if (
        not math.isfinite(sample_interval_seconds)
        or sample_interval_seconds <= 0
    ):
        raise HTTPException(
            status_code=422,
            detail="Sampling interval must be positive and finite",
        )

    filename = safe_filename(file.filename)

    if Path(filename).suffix.lower() != ".mp4":
        raise HTTPException(
            status_code=415,
            detail="Only MP4 videos are supported",
        )

    video_bytes = await read_upload(
        file,
        MAX_VIDEO_BYTES,
        "Video",
    )

    # MP4 files typically contain an ftyp box near the start.
    if b"ftyp" not in video_bytes[:32]:
        raise HTTPException(
            status_code=400,
            detail="Invalid MP4 file header",
        )

    temporary_path = None

    try:
        with NamedTemporaryFile(
            suffix=".mp4",
            delete=False,
        ) as temp:
            temporary_path = Path(temp.name)
            temp.write(video_bytes)

        report = await execute_inspection(
            inspect_video,
            video_path=temporary_path,
            predictor=predictor,
            sample_interval_seconds=sample_interval_seconds,
            max_frames=100,
        )

        report["video_file"] = filename

        await anyio.to_thread.run_sync(
            save_report,
            "Video",
            filename,
            report["decision"],
            report,
        )

        return report

    except (ValueError, RuntimeError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported video",
        )

    except sqlite3.Error:
        logger.exception("Video history persistence failed")
        raise HTTPException(
            status_code=503,
            detail="Inspection history storage unavailable",
        )

    except Exception:
        logger.exception("Video inspection failed")
        raise HTTPException(
            status_code=500,
            detail="Video inspection failed",
        )

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.get("/inspections")
def inspection_history(limit: int = 100):
    if not 1 <= limit <= 1000:
        raise HTTPException(
            status_code=422,
            detail="Limit must be between 1 and 1000",
        )

    try:
        return {
            "inspections": list_inspections(limit=limit)
        }

    except sqlite3.Error:
        logger.exception("History retrieval failed")
        raise HTTPException(
            status_code=503,
            detail="Inspection history unavailable",
        )


@app.get("/inspections/{inspection_id}")
def inspection_detail(inspection_id: int):
    if inspection_id < 1:
        raise HTTPException(
            status_code=422,
            detail="Inspection ID must be positive",
        )

    try:
        inspection = get_inspection(inspection_id)

    except sqlite3.Error:
        logger.exception("History detail retrieval failed")
        raise HTTPException(
            status_code=503,
            detail="Inspection history unavailable",
        )

    if inspection is None:
        raise HTTPException(
            status_code=404,
            detail="Inspection not found",
        )

    return inspection
