
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from inspectai.api import main as api


@pytest.fixture
def client(monkeypatch, tmp_path):
    """
    Test API requests without loading the real PatchCore model
    or modifying the production inspection database.
    """
    import inspectai.storage.history as history

    test_db = tmp_path / "inspection_history.db"

    monkeypatch.setattr(
        api,
        "initialize_database",
        lambda: history.initialize_database(test_db),
    )

    monkeypatch.setattr(
        api,
        "save_inspection",
        lambda **kwargs: history.save_inspection(
            **kwargs, db_path=test_db
        ),
    )

    monkeypatch.setattr(
        api,
        "list_inspections",
        lambda limit=100: history.list_inspections(
            limit=limit, db_path=test_db
        ),
    )

    monkeypatch.setattr(
        api,
        "get_inspection",
        lambda inspection_id: history.get_inspection(
            inspection_id, db_path=test_db
        ),
    )

    class FakePredictor:
        def predict(self, path):
            import numpy as np

            return {
                "classification": "NORMAL",
                "is_anomalous": False,
                "anomaly_score": 0.12,
                "predicted_mask": np.zeros(
                    (32, 32), dtype=np.uint8
                ),
            }

    monkeypatch.setattr(api, "predictor", FakePredictor())

    with TestClient(api.app, raise_server_exceptions=False) as test_client:
        # The real lifespan loads the model. Restore the mock.
        monkeypatch.setattr(api, "predictor", FakePredictor())
        yield test_client


def make_image_bytes():
    buffer = io.BytesIO()

    Image.new(
        "RGB",
        (32, 32),
        color=(120, 140, 160),
    ).save(buffer, format="PNG")

    return buffer.getvalue()


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_image_inspection_and_history(client):
    response = client.post(
        "/inspect",
        files={
            "file": (
                "test.png",
                make_image_bytes(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    report = response.json()

    assert report["classification"] == "NORMAL"
    assert report["anomaly_score"] == 0.12
    assert report["defective_pixels"] == 0

    history_response = client.get("/inspections")

    assert history_response.status_code == 200

    records = history_response.json()["inspections"]

    assert len(records) == 1
    assert records[0]["inspection_type"] == "2D Image"
    assert records[0]["filename"] == "test.png"

    detail_response = client.get(
        f"/inspections/{records[0]['id']}"
    )

    assert detail_response.status_code == 200
    assert detail_response.json()["report"] == report


def test_reject_empty_image(client):
    response = client.post(
        "/inspect",
        files={
            "file": (
                "empty.png",
                b"",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


def test_reject_invalid_image_content(client):
    response = client.post(
        "/inspect",
        files={
            "file": (
                "fake.png",
                b"this is not an image",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


def test_reject_unsupported_image_type(client):
    response = client.post(
        "/inspect",
        files={
            "file": (
                "test.gif",
                b"GIF89a",
                "image/gif",
            )
        },
    )

    assert response.status_code == 415


def test_reject_oversized_image(client):
    response = client.post(
        "/inspect",
        files={
            "file": (
                "large.png",
                b"x" * (api.MAX_IMAGE_BYTES + 1),
                "image/png",
            )
        },
    )

    assert response.status_code == 413


def test_reject_invalid_ply(client):
    response = client.post(
        "/inspect/3d",
        files={
            "file": (
                "fake.ply",
                b"not a point cloud",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400


def test_reject_invalid_3d_tolerance(client):
    response = client.post(
        "/inspect/3d",
        data={"tolerance_mm": -1},
        files={
            "file": (
                "test.ply",
                b"ply\nformat ascii 1.0\nend_header\n",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 422


def test_reject_invalid_mp4(client):
    response = client.post(
        "/inspect/video",
        files={
            "file": (
                "fake.mp4",
                b"not a valid mp4",
                "video/mp4",
            )
        },
    )

    assert response.status_code == 400


def test_reject_invalid_video_interval(client):
    response = client.post(
        "/inspect/video",
        data={"sample_interval_seconds": 0},
        files={
            "file": (
                "test.mp4",
                b"\x00\x00\x00\x18ftypisom",
                "video/mp4",
            )
        },
    )

    assert response.status_code == 422


def test_missing_history_record(client):
    response = client.get("/inspections/99999")

    assert response.status_code == 404


def test_invalid_history_limit(client):
    response = client.get("/inspections?limit=0")

    assert response.status_code == 422


def test_history_is_empty_initially(client):
    response = client.get("/inspections")

    assert response.status_code == 200
    assert response.json()["inspections"] == []
