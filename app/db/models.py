"""
Database models / type definitions for the case history system.
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class CaseRecord(BaseModel):
    """Represents a stored diagnosis case."""
    id: Optional[int] = None
    case_id: str
    created_at: str
    input_type: str
    symptoms_text: Optional[str] = None
    clinical_question: Optional[str] = None
    image_path: Optional[str] = None
    diagnosis_json: Optional[str] = None
    primary_diagnosis: Optional[str] = None
    confidence: Optional[float] = None
    safety_status: Optional[str] = None
    report_path: Optional[str] = None
