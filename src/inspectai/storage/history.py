
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATABASE_PATH = (
    PROJECT_ROOT / "data/processed/inspection_history.db"
)


def get_connection(db_path=DATABASE_PATH):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(db_path),
        timeout=10,
    )
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database(db_path=DATABASE_PATH):
    with get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                inspection_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                decision TEXT NOT NULL,
                anomaly_score REAL,
                report_json TEXT NOT NULL
            )
            """
        )


def save_inspection(
    inspection_type,
    filename,
    decision,
    report,
    anomaly_score=None,
    db_path=DATABASE_PATH,
):
    initialize_database(db_path)

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO inspections (
                timestamp_utc,
                inspection_type,
                filename,
                decision,
                anomaly_score,
                report_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                inspection_type,
                filename,
                decision,
                anomaly_score,
                json.dumps(report),
            ),
        )

        return cursor.lastrowid


def list_inspections(
    limit=100,
    db_path=DATABASE_PATH,
):
    if not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("Limit must be between 1 and 1000")

    initialize_database(db_path)

    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                timestamp_utc,
                inspection_type,
                filename,
                decision,
                anomaly_score
            FROM inspections
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_inspection(
    inspection_id,
    db_path=DATABASE_PATH,
):
    initialize_database(db_path)

    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM inspections
            WHERE id = ?
            """,
            (inspection_id,),
        ).fetchone()

    if row is None:
        return None

    result = dict(row)
    result["report"] = json.loads(
        result.pop("report_json")
    )

    return result
