"""
Database models / type definitions for the case history system.
"""

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field


class CaseRecord(BaseModel):
    """Represents a stored diagnosis case in PostgreSQL."""
    id: int | None = None
    case_id: str = Field(..., description="Unique case identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Case creation timestamp (timezone-aware UTC)",
    )
    input_type: str = Field("unknown", description="Modality type (text, image, multimodal)")
    symptoms_text: str | None = None
    clinical_question: str | None = None
    image_path: str | None = None
    diagnosis_json: dict[str, Any] | str | None = None
    primary_diagnosis: str = ""
    confidence: float = 0.0
    safety_status: str = "unknown"
    report_path: str | None = None
