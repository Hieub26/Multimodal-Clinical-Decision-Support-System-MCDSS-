"""
Data mapper for converting between DTOs, Dicts, Database Rows, and CaseRecord domain models.
"""

import json
from datetime import datetime, timezone
from typing import Any
import asyncpg

from app.db.models import CaseRecord


def _as_utc_datetime(value: Any) -> datetime:
    """Coerce a timestamp to a timezone-aware UTC datetime.

    Naive values are taken to be UTC. They must not reach asyncpg as-is: for
    TIMESTAMPTZ it interprets a naive datetime in the machine's local zone,
    which shifts stored times by the UTC offset.
    """
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            value = None
    if not isinstance(value, datetime):
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class CaseMapper:
    """Handles mapping between dictionaries, database rows, and CaseRecord objects."""

    @staticmethod
    def to_record(data: CaseRecord | dict[str, Any]) -> CaseRecord:
        """Convert input dict or model to a validated CaseRecord instance with datetime created_at."""
        if isinstance(data, CaseRecord):
            record = data
        else:
            diagnosis_data = data.get("diagnosis") or data.get("diagnosis_json") or {}

            record = CaseRecord(
                case_id=data["case_id"],
                created_at=_as_utc_datetime(data.get("created_at")),
                input_type=data.get("input_type", "unknown"),
                symptoms_text=data.get("symptoms_text"),
                clinical_question=data.get("clinical_question"),
                image_path=data.get("image_path"),
                diagnosis_json=diagnosis_data,
                primary_diagnosis=data.get("primary_diagnosis", ""),
                confidence=float(data.get("confidence", 0.0)),
                safety_status=data.get("safety_status", "unknown"),
                report_path=data.get("report_path"),
            )

        # Guarantee created_at is a timezone-aware datetime object
        record.created_at = _as_utc_datetime(record.created_at)

        return record

    @staticmethod
    def from_row(row: asyncpg.Record | dict[str, Any]) -> CaseRecord:
        """Convert an asyncpg database row to a CaseRecord object."""
        r = dict(row)
        diag = r.get("diagnosis_json")
        if isinstance(diag, str):
            # Rows written before the JSONB double-encoding fix
            try:
                diag = json.loads(diag)
            except json.JSONDecodeError:
                diag = {}

        return CaseRecord(
            id=r.get("id"),
            case_id=r["case_id"],
            created_at=_as_utc_datetime(r.get("created_at")),
            input_type=r.get("input_type", "unknown"),
            symptoms_text=r.get("symptoms_text"),
            clinical_question=r.get("clinical_question"),
            image_path=r.get("image_path"),
            diagnosis_json=diag,
            primary_diagnosis=r.get("primary_diagnosis", ""),
            confidence=float(r.get("confidence", 0.0)),
            safety_status=r.get("safety_status", "unknown"),
            report_path=r.get("report_path"),
        )

    @staticmethod
    def to_dict(record: CaseRecord) -> dict[str, Any]:
        """Convert a CaseRecord to a clean dictionary representation."""
        data = record.model_dump()
        if isinstance(data.get("created_at"), datetime):
            data["created_at"] = data["created_at"].isoformat()
        return data
