"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional


class TextDiagnosisRequest(BaseModel):
    """Request for text-based diagnosis."""
    symptoms_text: Optional[str] = Field(None, description="Patient symptoms description")
    clinical_question: Optional[str] = Field(None, description="Clinical question to analyze")


class ImageDiagnosisRequest(BaseModel):
    """Request for image-based diagnosis (image sent as form data)."""
    pass


class MultimodalDiagnosisRequest(BaseModel):
    """Request for multimodal diagnosis."""
    symptoms_text: Optional[str] = None
    clinical_question: Optional[str] = None
    # image is sent as UploadFile in form data


class DiagnosisResult(BaseModel):
    """Individual diagnosis output."""
    primary_diagnosis: str = ""
    confidence: float = 0.0
    explanation: str = ""
    question_answer: str = ""
    severity: str = "unknown"
    urgency: str = "routine"
    recommended_actions: list[str] = []


class SafetyInfo(BaseModel):
    """Safety assessment information."""
    status: str = "unknown"
    is_approved: bool = False
    risk_factors: list[str] = []
    message: str = ""


class ValidationInfo(BaseModel):
    """Guideline validation information."""
    guideline_consistent: bool = False
    support_score: float = 0.0
    notes: list[str] = []


class DiagnosisResponse(BaseModel):
    """Full diagnosis response."""
    case_id: str
    diagnosis: DiagnosisResult
    modality: str = "unknown"
    fusion_strategy: str = ""
    safety: SafetyInfo
    validation: ValidationInfo
    nlp_details: Optional[dict] = None
    cv_details: Optional[dict] = None
    report_path: Optional[str] = None
    gradcam_path: Optional[str] = None
    disclaimer: str = ""


class CaseHistoryItem(BaseModel):
    """Case history list item."""
    case_id: str
    created_at: str
    input_type: str
    primary_diagnosis: str = ""
    confidence: float = 0.0
    safety_status: str = ""


class CaseHistoryResponse(BaseModel):
    """Case history response."""
    total: int
    cases: list[CaseHistoryItem]


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    components: dict = {}


class IngestRequest(BaseModel):
    """Request to ingest clinical guidelines."""
    text: Optional[str] = None
    file_path: Optional[str] = None
