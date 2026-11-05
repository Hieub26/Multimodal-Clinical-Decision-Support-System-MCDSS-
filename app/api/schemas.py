"""
Pydantic schemas for API request/response models.
Provides strict validation, immutable default factories, Enums, and type safety for clinical DSS endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator


# --- Enums for Categorical Fields ---

class SeverityLevel(str, Enum):
    """Clinical severity levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class UrgencyLevel(str, Enum):
    """Action urgency levels."""
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENT = "emergent"
    EMERGENCY = "emergency"


class ModalityType(str, Enum):
    """Diagnostic input modality types."""
    NLP = "nlp"
    CV = "cv"
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class SafetyStatus(str, Enum):
    """Safety evaluation status."""
    APPROVED = "approved"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"
    FALLBACK = "fallback"
    UNKNOWN = "unknown"


class SystemHealthStatus(str, Enum):
    """System health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# --- Request & Response Models ---

class TextDiagnosisRequest(BaseModel):
    """Request for text-based diagnosis."""
    symptoms_text: str | None = Field(None, description="Patient symptoms description")
    clinical_question: str | None = Field(None, description="Clinical question to analyze")

    @model_validator(mode="after")
    def check_input_provided(self) -> "TextDiagnosisRequest":
        if not self.symptoms_text and not self.clinical_question:
            raise ValueError("At least one of 'symptoms_text' or 'clinical_question' must be provided.")
        return self


class MultimodalDiagnosisRequest(BaseModel):
    """Request metadata for multimodal diagnosis."""
    symptoms_text: str | None = Field(None, description="Patient symptoms description")
    clinical_question: str | None = Field(None, description="Clinical question to analyze")
    # image file is sent as UploadFile in form data


class DiagnosisResult(BaseModel):
    """Individual diagnosis output with score validations."""
    primary_diagnosis: str = Field("", description="Primary diagnostic hypothesis")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Diagnostic confidence score (0 to 1)")
    explanation: str = Field("", description="Clinical reasoning explanation")
    question_answer: str = Field("", description="Direct answer to clinical question if provided")
    severity: SeverityLevel = Field(SeverityLevel.UNKNOWN, description="Clinical severity level")
    urgency: UrgencyLevel = Field(UrgencyLevel.ROUTINE, description="Action urgency level")
    recommended_actions: list[str] = Field(default_factory=list, description="List of recommended clinical actions")


class SafetyInfo(BaseModel):
    """Safety assessment information."""
    status: SafetyStatus = Field(SafetyStatus.UNKNOWN, description="Safety check status")
    is_approved: bool = Field(False, description="Whether decision passed safety review")
    risk_factors: list[str] = Field(default_factory=list, description="Identified risk factors or red flags")
    message: str = Field("", description="Detailed safety review message")


class ValidationInfo(BaseModel):
    """Guideline validation information."""
    guideline_consistent: bool = Field(False, description="Whether diagnosis aligns with guidelines")
    support_score: float = Field(0.0, ge=0.0, le=1.0, description="Guideline support score (0 to 1)")
    notes: list[str] = Field(default_factory=list, description="Validation notes and warnings")
    method: str = Field("", description="How the result was reached: llm_judge, token_matching, skipped or no_guidelines")
    judge_model: str | None = Field(None, description="Model that judged guideline support, when an LLM judge was used")


class DiagnosisResponse(BaseModel):
    """Full diagnosis response."""
    case_id: str = Field(..., description="Unique case identifier")
    diagnosis: DiagnosisResult
    modality: ModalityType = Field(ModalityType.UNKNOWN, description="Diagnostic modality")
    fusion_strategy: str = Field("", description="Multimodal fusion strategy applied")
    safety: SafetyInfo
    validation: ValidationInfo
    nlp_details: dict[str, Any] | None = Field(None, description="Detailed NLP breakdown")
    cv_details: dict[str, Any] | None = Field(None, description="Detailed CV breakdown")
    vlm_safety_review: dict[str, Any] | None = Field(None, description="VLM safety review details")
    uncertainty: dict[str, Any] | None = Field(None, description="Uncertainty estimation details")
    severity_score: float | None = Field(None, ge=0.0, le=1.0, description="Overall severity score (0 to 1)")
    severity_breakdown: dict[str, Any] | None = Field(None, description="Severity score breakdown")
    report_path: str | None = Field(None, description="Path to generated clinical report")
    gradcam_path: str | None = Field(None, description="Path to Grad-CAM heatmap image")
    disclaimer: str = Field("", description="Medical decision support disclaimer")


class CaseHistoryItem(BaseModel):
    """Case history list item."""
    case_id: str = Field(..., description="Unique case identifier")
    created_at: datetime = Field(..., description="Timestamp of case creation")
    input_type: ModalityType | str = Field(ModalityType.UNKNOWN, description="Modality input type")
    primary_diagnosis: str = Field("", description="Primary diagnosis title")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score (0 to 1)")
    safety_status: SafetyStatus | str = Field(SafetyStatus.UNKNOWN, description="Safety status")


class CaseHistoryResponse(BaseModel):
    """Case history response."""
    total: int = Field(..., ge=0, description="Total number of cases")
    cases: list[CaseHistoryItem] = Field(default_factory=list, description="List of historical cases")


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: SystemHealthStatus = Field(SystemHealthStatus.HEALTHY, description="Overall health status")
    version: str = Field("1.0.0", description="API version")
    components: dict[str, Any] = Field(default_factory=dict, description="Component health details")


class IngestRequest(BaseModel):
    """Request to ingest clinical guidelines."""
    text: str | None = Field(None, description="Guideline text content")
    file_path: str | None = Field(None, description="Path to guideline document file")

    @model_validator(mode="after")
    def check_text_or_file(self) -> "IngestRequest":
        if not self.text and not self.file_path:
            raise ValueError("At least one of 'text' or 'file_path' must be provided.")
        return self
