
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from inspectai.api.main import app


VIDEO_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/processed/videos/bottle_inspection.mp4"
)

client = TestClient(app)


def fake_inspect_video(
    video_path,
    predictor,
    sample_interval_seconds=1.0,
    max_frames=100,
):
    return {
        "inspection_type": "Video anomaly inspection",
        "video_file": Path(video_path).name,
        "fps": 10,
        "processed_video_frames": 80,
        "inspected_frames": 8,
        "defective_frames": 2,
        "decision": "FAIL",
        "sampling_interval_seconds": sample_interval_seconds,
        "inspection_truncated": False,
        "predictions": [],
    }


def test_video_inspection_success():
    assert VIDEO_PATH.is_file()

    with (
        patch("inspectai.api.main.predictor", object()),
        patch(
            "inspectai.api.main.inspect_video",
            side_effect=fake_inspect_video,
        ),
    ):
        with VIDEO_PATH.open("rb") as video:
            response = client.post(
                "/inspect/video",
                files={
                    "file": (
                        "bottle_inspection.mp4",
                        video,
                        "video/mp4",
                    )
                },
                data={"sample_interval_seconds": "1.0"},
            )

    assert response.status_code == 200

    report = response.json()

    assert report["decision"] == "FAIL"
    assert report["inspected_frames"] == 8
    assert report["defective_frames"] == 2
    assert report["inspection_truncated"] is False


def test_invalid_video_extension():
    with patch("inspectai.api.main.predictor", object()):
        response = client.post(
            "/inspect/video",
            files={
                "file": (
                    "invalid.txt",
                    b"invalid content",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 415


def test_invalid_sampling_interval():
    with patch("inspectai.api.main.predictor", object()):
        response = client.post(
            "/inspect/video",
            files={
                "file": (
                    "sample.mp4",
                    b"dummy content",
                    "video/mp4",
                )
            },
            data={"sample_interval_seconds": "-1"},
        )

    assert response.status_code == 422


def test_video_model_not_ready():
    with patch("inspectai.api.main.predictor", None):
        response = client.post(
            "/inspect/video",
            files={
                "file": (
                    "sample.mp4",
                    b"dummy content",
                    "video/mp4",
                )
            },
        )

    assert response.status_code == 503
