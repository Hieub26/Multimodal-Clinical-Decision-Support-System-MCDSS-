"""
PostgreSQL database setup with asyncpg.
Handles table creation and CRUD operations for case history.
"""

import json
import asyncpg
from datetime import datetime
from app.config import settings
from app.utils.logger import db_logger

DB_URL = settings.database_url


async def init_db():
    """Create database tables if they don't exist."""
    try:
        conn = await asyncpg.connect(DB_URL)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id SERIAL PRIMARY KEY,
                case_id TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                input_type TEXT NOT NULL,
                symptoms_text TEXT,
                clinical_question TEXT,
                image_path TEXT,
                diagnosis_json TEXT,
                primary_diagnosis TEXT,
                confidence REAL,
                safety_status TEXT,
                report_path TEXT
            )
        """)
        await conn.close()
        db_logger.info("Database initialized")
    except Exception as e:
        db_logger.error(f"Database initialization failed: {e}")
        # Note: If database doesn't exist, this will fail. You might need to create it manually first.


async def save_case(case_data: dict) -> str:
    """Save a diagnosis case to the database."""
    conn = await asyncpg.connect(DB_URL)
    await conn.execute(
        """INSERT INTO cases 
        (case_id, created_at, input_type, symptoms_text, clinical_question,
         image_path, diagnosis_json, primary_diagnosis, confidence,
         safety_status, report_path)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)""",
        case_data["case_id"],
        case_data.get("created_at", datetime.utcnow().isoformat()),
        case_data.get("input_type", "unknown"),
        case_data.get("symptoms_text"),
        case_data.get("clinical_question"),
        case_data.get("image_path"),
        json.dumps(case_data.get("diagnosis", {})),
        case_data.get("primary_diagnosis", ""),
        case_data.get("confidence", 0.0),
        case_data.get("safety_status", "unknown"),
        case_data.get("report_path"),
    )
    await conn.close()
    db_logger.info(f"Case saved: {case_data['case_id']}")
    return case_data["case_id"]


async def get_case(case_id: str) -> dict | None:
    """Retrieve a single case by ID."""
    conn = await asyncpg.connect(DB_URL)
    row = await conn.fetchrow(
        "SELECT * FROM cases WHERE case_id = $1", case_id
    )
    await conn.close()
    if row:
        return dict(row)
    return None


async def get_all_cases(limit: int = 50, offset: int = 0) -> list[dict]:
    """Retrieve all cases with pagination."""
    conn = await asyncpg.connect(DB_URL)
    rows = await conn.fetch(
        "SELECT * FROM cases ORDER BY created_at DESC LIMIT $1 OFFSET $2",
        limit, offset
    )
    await conn.close()
    return [dict(row) for row in rows]


async def get_case_count() -> int:
    """Get total number of cases."""
    conn = await asyncpg.connect(DB_URL)
    count = await conn.fetchval("SELECT COUNT(*) FROM cases")
    await conn.close()
    return count or 0
