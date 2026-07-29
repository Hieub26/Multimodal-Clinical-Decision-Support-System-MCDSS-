"""
PostgreSQL database setup with asyncpg connection pool, Repository pattern, and JSONB support.
Provides enterprise DatabaseManager & CaseRepository architecture with asyncpg codecs.
"""

import json
from datetime import datetime
from typing import Any
import asyncpg

from app.config import settings
from app.db.models import CaseRecord
from app.db.mapper import CaseMapper
from app.db.migrator import SchemaMigrator
from app.db.exceptions import DatabaseError, CaseSaveError, CaseNotFoundError
from app.utils.logger import db_logger


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Callback to set up custom asyncpg JSONB codecs per connection."""
    try:
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )
    except Exception as e:
        db_logger.debug(f"JSONB codec registration note: {e}")


class DatabaseManager:
    """Manages PostgreSQL connection pool lifecycle cleanly without containing DDL logic."""

    def __init__(self, db_url: str = settings.database_url):
        self.db_url = db_url
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Initialize connection pool and apply schema migrations."""
        if self._pool is not None:
            return

        try:
            self._pool = await asyncpg.create_pool(
                self.db_url,
                min_size=settings.db_pool_min_size,
                max_size=settings.db_pool_max_size,
                init=_init_connection,
            )

            # Apply DDL schema migrations via dedicated migrator
            async with self.acquire() as conn:
                await SchemaMigrator.apply_migrations(conn)

            db_logger.info(
                f"DatabaseManager pool ready (min={settings.db_pool_min_size}, max={settings.db_pool_max_size})"
            )
        except Exception as e:
            db_logger.error(f"DatabaseManager initialization error: {e}")
            raise DatabaseError(f"Database initialization failed: {e}") from e

    async def close(self) -> None:
        """Close connection pool gracefully."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            db_logger.info("DatabaseManager connection pool closed")

    def acquire(self):
        """Acquire a connection from the pool."""
        if self._pool is None:
            raise DatabaseError("Database pool not initialized. Call initialize() first.")
        return self._pool.acquire()


class CaseRepository:
    """Repository handling CRUD operations for Case History records with CaseMapper & Exception handling."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def save_case(self, case_input: CaseRecord | dict[str, Any]) -> str:
        """Save a diagnosis case using ACID transaction safety and CaseMapper."""
        record = CaseMapper.to_record(case_input)

        if isinstance(record.diagnosis_json, (dict, list)):
            diag_val = json.dumps(record.diagnosis_json)
        elif isinstance(record.diagnosis_json, str):
            diag_val = record.diagnosis_json
        else:
            diag_val = "{}"

        try:
            async with self.db.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        """
                        INSERT INTO cases 
                        (case_id, created_at, input_type, symptoms_text, clinical_question,
                         image_path, diagnosis_json, primary_diagnosis, confidence,
                         safety_status, report_path)
                        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11)
                        ON CONFLICT (case_id) DO UPDATE SET
                            symptoms_text = EXCLUDED.symptoms_text,
                            clinical_question = EXCLUDED.clinical_question,
                            image_path = EXCLUDED.image_path,
                            diagnosis_json = EXCLUDED.diagnosis_json,
                            primary_diagnosis = EXCLUDED.primary_diagnosis,
                            confidence = EXCLUDED.confidence,
                            safety_status = EXCLUDED.safety_status,
                            report_path = EXCLUDED.report_path
                        """,
                        record.case_id,
                        record.created_at,
                        record.input_type,
                        record.symptoms_text,
                        record.clinical_question,
                        record.image_path,
                        diag_val,
                        record.primary_diagnosis,
                        record.confidence,
                        record.safety_status,
                        record.report_path,
                    )
            db_logger.info(f"Case saved to database: {record.case_id}")
            return record.case_id
        except asyncpg.PostgresError as e:
            db_logger.error(f"PostgreSQL error while saving case '{record.case_id}': {e}")
            raise CaseSaveError(record.case_id, str(e)) from e
        except Exception as e:
            db_logger.error(f"Unexpected error while saving case '{record.case_id}': {e}")
            raise CaseSaveError(record.case_id, str(e)) from e

    async def get_case(self, case_id: str) -> CaseRecord | None:
        """Retrieve a single case by case_id returning a CaseRecord model."""
        try:
            async with self.db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM cases WHERE case_id = $1", case_id
                )
            if not row:
                return None
            return CaseMapper.from_row(row)
        except asyncpg.PostgresError as e:
            db_logger.error(f"PostgreSQL error while fetching case '{case_id}': {e}")
            raise DatabaseError(f"Database error fetching case {case_id}: {e}") from e

    async def get_all_cases(self, limit: int = 50, offset: int = 0) -> list[CaseRecord]:
        """Retrieve historical cases returning CaseRecord models."""
        try:
            async with self.db.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM cases ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                    limit, offset
                )
            return [CaseMapper.from_row(r) for r in rows]
        except asyncpg.PostgresError as e:
            db_logger.error(f"PostgreSQL error while fetching case history: {e}")
            raise DatabaseError(f"Database error fetching cases: {e}") from e

    async def get_case_count(self) -> int:
        """Get total number of recorded cases."""
        try:
            async with self.db.acquire() as conn:
                count = await conn.fetchval("SELECT COUNT(*) FROM cases")
            return count or 0
        except asyncpg.PostgresError as e:
            db_logger.error(f"PostgreSQL error while counting cases: {e}")
            raise DatabaseError(f"Database error counting cases: {e}") from e


# --- Global Instances for Backward Compatibility ---
db_manager = DatabaseManager()
case_repository = CaseRepository(db_manager)


# --- Backward-Compatible Function Proxies ---
async def init_db() -> None:
    """Initialize database connection pool and schema."""
    await db_manager.initialize()


async def close_db() -> None:
    """Close database connection pool gracefully."""
    await db_manager.close()


async def save_case(case_data: CaseRecord | dict[str, Any]) -> str:
    """Save a diagnosis case to the database."""
    return await case_repository.save_case(case_data)


async def get_case(case_id: str) -> dict[str, Any] | None:
    """Retrieve a case by ID returning dictionary representation."""
    record = await case_repository.get_case(case_id)
    if record:
        return CaseMapper.to_dict(record)
    return None


async def get_all_cases(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """Retrieve cases with pagination returning list of dictionary representations."""
    records = await case_repository.get_all_cases(limit, offset)
    return [CaseMapper.to_dict(r) for r in records]


async def get_case_count() -> int:
    """Get total case count."""
    return await case_repository.get_case_count()
