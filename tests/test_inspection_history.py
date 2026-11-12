
import pytest

from inspectai.storage.history import (
    initialize_database,
    save_inspection,
    list_inspections,
    get_inspection,
)


@pytest.fixture
def db_path(tmp_path):
    """Use a separate temporary database for each test."""
    path = tmp_path / "inspection_history.db"
    initialize_database(path)
    return path


def test_database_initialization(db_path):
    assert db_path.exists()


def test_save_image_inspection(db_path):
    report = {
        "classification": "DEFECTIVE",
        "anomaly_score": 0.98,
        "defective_pixels": 250,
    }

    inspection_id = save_inspection(
        inspection_type="2D Image",
        filename="defective.png",
        decision="DEFECTIVE",
        anomaly_score=0.98,
        report=report,
        db_path=db_path,
    )

    assert inspection_id == 1

    saved = get_inspection(
        inspection_id,
        db_path=db_path,
    )

    assert saved["inspection_type"] == "2D Image"
    assert saved["filename"] == "defective.png"
    assert saved["decision"] == "DEFECTIVE"
    assert saved["anomaly_score"] == pytest.approx(0.98)
    assert saved["report"] == report
    assert saved["timestamp_utc"]


def test_save_3d_inspection(db_path):
    report = {
        "quality_control": {
            "decision": "FAIL",
            "defective_points": 321,
        }
    }

    inspection_id = save_inspection(
        inspection_type="3D Point Cloud",
        filename="defective_pipe.ply",
        decision="FAIL",
        report=report,
        db_path=db_path,
    )

    saved = get_inspection(
        inspection_id,
        db_path=db_path,
    )

    assert saved["decision"] == "FAIL"
    assert saved["anomaly_score"] is None
    assert saved["report"]["quality_control"]["defective_points"] == 321


def test_save_video_inspection(db_path):
    report = {
        "decision": "FAIL",
        "inspected_frames": 8,
        "defective_frames": 2,
    }

    inspection_id = save_inspection(
        inspection_type="Video",
        filename="inspection.mp4",
        decision="FAIL",
        report=report,
        db_path=db_path,
    )

    saved = get_inspection(
        inspection_id,
        db_path=db_path,
    )

    assert saved["report"]["inspected_frames"] == 8
    assert saved["report"]["defective_frames"] == 2


def test_list_inspections_newest_first(db_path):
    for number in range(3):
        save_inspection(
            inspection_type="2D Image",
            filename=f"image_{number}.png",
            decision="NORMAL",
            report={"number": number},
            db_path=db_path,
        )

    inspections = list_inspections(
        db_path=db_path,
    )

    assert len(inspections) == 3
    assert [item["id"] for item in inspections] == [3, 2, 1]


def test_list_inspections_limit(db_path):
    for number in range(5):
        save_inspection(
            inspection_type="Video",
            filename=f"video_{number}.mp4",
            decision="PASS",
            report={"number": number},
            db_path=db_path,
        )

    inspections = list_inspections(
        limit=2,
        db_path=db_path,
    )

    assert len(inspections) == 2
    assert [item["id"] for item in inspections] == [5, 4]


@pytest.mark.parametrize(
    "invalid_limit",
    [0, -1, 1001, 1.5, "ten"],
)
def test_invalid_history_limit(db_path, invalid_limit):
    with pytest.raises(ValueError):
        list_inspections(
            limit=invalid_limit,
            db_path=db_path,
        )


def test_missing_inspection(db_path):
    result = get_inspection(
        999,
        db_path=db_path,
    )

    assert result is None


def test_database_persistence(db_path):
    inspection_id = save_inspection(
        inspection_type="2D Image",
        filename="persistent.png",
        decision="NORMAL",
        report={"classification": "NORMAL"},
        db_path=db_path,
    )

    # A separate database connection should retrieve
    # the previously committed record.
    saved = get_inspection(
        inspection_id,
        db_path=db_path,
    )

    assert saved is not None
    assert saved["filename"] == "persistent.png"


def test_empty_history(db_path):
    assert list_inspections(db_path=db_path) == []
