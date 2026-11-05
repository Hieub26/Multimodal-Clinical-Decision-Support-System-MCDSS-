"""
API Dependencies: shared dependency injection for auth, DB sessions,
and common service instances.
"""

import secrets
from functools import lru_cache
from fastapi import Header, HTTPException
from app.config import settings
from app.core.nlp.vector_store import VectorStore
from app.core.nlp.rag_engine import RAGEngine
from app.core.nlp.text_preprocessor import TextPreprocessor
from app.core.nlp.jev_judge import JevJudge
from app.core.nlp.question_understanding import QuestionUnderstanding
from app.core.nlp.clinical_fallback import ClinicalFallbackEngine
from app.core.cv.cv_model import MedicalCVModel
from app.core.cv.image_preprocessor import ImagePreprocessor
from app.core.cv.cv_postprocessor import CVPostprocessor
from app.core.cv.uncertainty_analyzer import UncertaintyAnalyzer
from app.core.fusion.fusion_engine import FusionEngine
from app.core.validation.guideline_validator import GuidelineValidator
from app.core.safety.safety_controller import SafetyController
from app.utils.report_generator import ReportGenerator
from app.db.database import DatabaseManager, CaseRepository, db_manager, case_repository


@lru_cache()
def get_vector_store() -> VectorStore:
    """Singleton VectorStore instance shared across all components."""
    return VectorStore()


@lru_cache()
def get_jev_judge() -> JevJudge:
    """Singleton Jev judge (inactive without a TypeSafe API key)."""
    return JevJudge()


@lru_cache()
def get_text_preprocessor() -> TextPreprocessor:
    """Singleton TextPreprocessor instance."""
    return TextPreprocessor(context_judge=get_jev_judge())


@lru_cache()
def get_question_understanding() -> QuestionUnderstanding:
    """Singleton QuestionUnderstanding instance."""
    return QuestionUnderstanding()


@lru_cache()
def get_clinical_fallback_engine() -> ClinicalFallbackEngine:
    """Singleton ClinicalFallbackEngine instance."""
    return ClinicalFallbackEngine(preprocessor=get_text_preprocessor())


@lru_cache()
def get_rag_engine() -> RAGEngine:
    """Singleton RAG engine instance using Dependency Injection."""
    return RAGEngine(
        vector_store=get_vector_store(),
        text_preprocessor=get_text_preprocessor(),
        question_understanding=get_question_understanding(),
        fallback_engine=get_clinical_fallback_engine(),
    )


@lru_cache()
def get_image_preprocessor() -> ImagePreprocessor:
    """Singleton ImagePreprocessor instance."""
    return ImagePreprocessor()


@lru_cache()
def get_cv_postprocessor() -> CVPostprocessor:
    """Singleton CVPostprocessor instance."""
    return CVPostprocessor()


@lru_cache()
def get_uncertainty_analyzer() -> UncertaintyAnalyzer:
    """Singleton UncertaintyAnalyzer instance."""
    return UncertaintyAnalyzer()


@lru_cache()
def get_cv_model() -> MedicalCVModel:
    """Singleton CV model instance using Dependency Injection."""
    return MedicalCVModel(
        image_preprocessor=get_image_preprocessor(),
        postprocessor=get_cv_postprocessor(),
        uncertainty_analyzer=get_uncertainty_analyzer(),
    )


@lru_cache()
def get_fusion_engine() -> FusionEngine:
    """Singleton Fusion engine instance."""
    return FusionEngine()


@lru_cache()
def get_guideline_validator() -> GuidelineValidator:
    """Singleton Guideline validator instance."""
    return GuidelineValidator(vector_store=get_vector_store())


@lru_cache()
def get_safety_controller() -> SafetyController:
    """Singleton Safety controller instance."""
    return SafetyController(context_judge=get_jev_judge())


@lru_cache()
def get_report_generator() -> ReportGenerator:
    """Singleton Report generator instance."""
    return ReportGenerator()


def _key_matches(provided: str | None, expected: str) -> bool:
    return secrets.compare_digest((provided or "").encode(), expected.encode())


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Guard for endpoints that create or return patient cases.

    Enforced only when API_KEY is configured, so local development keeps
    working without credentials. The admin key is accepted as well: whoever
    may change the guideline library may also use the system.
    """
    expected = settings.api_key.get_secret_value()
    if not expected:
        return
    admin = settings.admin_api_key.get_secret_value()
    if _key_matches(x_api_key, expected) or (admin and _key_matches(x_api_key, admin)):
        return
    raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def require_admin_key(x_api_key: str | None = Header(default=None)) -> None:
    """Guard for endpoints that change server state.

    Enforced only when ADMIN_API_KEY is configured, so local development
    keeps working without credentials.
    """
    expected = settings.admin_api_key.get_secret_value()
    if expected and not _key_matches(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def get_db_manager() -> DatabaseManager:
    """Dependency injection provider for DatabaseManager."""
    return db_manager


def get_case_repository() -> CaseRepository:
    """Dependency injection provider for CaseRepository."""
    return case_repository
