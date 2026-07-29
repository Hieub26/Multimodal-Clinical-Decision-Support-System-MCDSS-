"""
Data mapper for converting between DTOs, Dicts, Database Rows, and CaseRecord domain models.
"""

import json
from datetime import datetime
from typing import Any
import asyncpg

from app.db.models import CaseRecord


class CaseMapper:
    """Handles mapping between dictionaries, database rows, and CaseRecord objects."""

    @staticmethod
    def to_record(data: CaseRecord | dict[str, Any]) -> CaseRecord:
        """Convert input dict or model to a validated CaseRecord instance with datetime created_at."""
        if isinstance(data, CaseRecord):
            record = data
        else:
            created_at_val = data.get("created_at")
            if isinstance(created_at_val, str):
                try:
                    created_at_val = datetime.fromisoformat(created_at_val)
                except ValueError:
                    created_at_val = datetime.utcnow()
            elif not isinstance(created_at_val, datetime):
                created_at_val = datetime.utcnow()

            diagnosis_data = data.get("diagnosis") or data.get("diagnosis_json") or {}

            record = CaseRecord(
                case_id=data["case_id"],
                created_at=created_at_val,
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

        # Guarantee created_at is a python datetime object
        if isinstance(record.created_at, str):
            try:
                record.created_at = datetime.fromisoformat(record.created_at)
            except ValueError:
                record.created_at = datetime.utcnow()
        elif not isinstance(record.created_at, datetime):
            record.created_at = datetime.utcnow()

        return record

    @staticmethod
    def from_row(row: asyncpg.Record | dict[str, Any]) -> CaseRecord:
        """Convert an asyncpg database row to a CaseRecord object."""
        r = dict(row)
        diag = r.get("diagnosis_json")
        if isinstance(diag, str):
            try:
                diag = json.loads(diag)
            except json.JSONDecodeError:
                diag = {}

        created_at = r.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.utcnow()
        elif not isinstance(created_at, datetime):
            created_at = datetime.utcnow()

        return CaseRecord(
            id=r.get("id"),
            case_id=r["case_id"],
            created_at=created_at,
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
