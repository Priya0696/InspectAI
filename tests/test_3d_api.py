
from pathlib import Path

from fastapi.testclient import TestClient

from inspectai.api.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/processed/pointclouds"

client = TestClient(app)


def upload_pointcloud(filename):
    path = DATA_DIR / filename

    with path.open("rb") as file:
        return client.post(
            "/inspect/3d",
            files={
                "file": (
                    filename,
                    file,
                    "application/octet-stream",
                )
            },
            data={"tolerance_mm": "1.0"},
        )


def test_defective_pipe_fails():
    response = upload_pointcloud(
        "registered_defective_pipe.ply"
    )

    assert response.status_code == 200

    report = response.json()

    assert report["quality_control"]["decision"] == "FAIL"
    assert report["measurement"]["defective_points"] > 0
    assert (
        report["measurement"]["maximum_inward_deviation_mm"]
        > 1.0
    )


def test_reference_pipe_passes():
    response = upload_pointcloud("reference_pipe.ply")

    assert response.status_code == 200

    report = response.json()

    assert report["quality_control"]["decision"] == "PASS"
    assert report["measurement"]["defective_points"] == 0


def test_invalid_pointcloud_rejected():
    response = client.post(
        "/inspect/3d",
        files={
            "file": (
                "invalid.txt",
                b"not a point cloud",
                "text/plain",
            )
        },
        data={"tolerance_mm": "1.0"},
    )

    assert response.status_code == 415


def test_invalid_tolerance_rejected():
    path = DATA_DIR / "reference_pipe.ply"

    with path.open("rb") as file:
        response = client.post(
            "/inspect/3d",
            files={
                "file": (
                    "reference_pipe.ply",
                    file,
                    "application/octet-stream",
                )
            },
            data={"tolerance_mm": "-1"},
        )

    assert response.status_code == 422
