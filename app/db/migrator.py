"""
Database Schema Migrator: Manages DDL migrations and index creation independently from DatabaseManager.
"""

import asyncpg
from app.utils.logger import db_logger


class SchemaMigrator:
    """Handles DDL execution and schema migration routines."""

    @staticmethod
    async def apply_migrations(conn: asyncpg.Connection) -> None:
        """Apply PostgreSQL table DDL and index migrations."""
        db_logger.info("Applying database schema migrations...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id SERIAL PRIMARY KEY,
                case_id TEXT UNIQUE NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                input_type TEXT NOT NULL,
                symptoms_text TEXT,
                clinical_question TEXT,
                image_path TEXT,
                diagnosis_json JSONB,
                primary_diagnosis TEXT,
                confidence DOUBLE PRECISION DEFAULT 0.0,
                safety_status TEXT,
                report_path TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_cases_created_at ON cases (created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_cases_case_id ON cases (case_id);
        """)

        # Safely alter column types for pre-existing tables created with TEXT columns
        try:
            await conn.execute("""
                ALTER TABLE cases 
                    ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at::timestamptz,
                    ALTER COLUMN diagnosis_json TYPE JSONB USING diagnosis_json::jsonb,
                    ALTER COLUMN confidence TYPE DOUBLE PRECISION USING confidence::double precision;
            """)
            db_logger.info("Database columns successfully migrated to TIMESTAMPTZ and JSONB.")
        except Exception as e:
            db_logger.debug(f"Schema column alteration migration note: {e}")

        db_logger.info("Database schema migrations applied successfully.")
