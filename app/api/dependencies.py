"""
API Dependencies: shared dependency injection for auth, DB sessions,
and common service instances.
"""

from functools import lru_cache
from app.core.nlp.vector_store import VectorStore
from app.core.nlp.rag_engine import RAGEngine
from app.core.cv.cv_model import MedicalCVModel
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
def get_rag_engine() -> RAGEngine:
    """Singleton RAG engine instance."""
    return RAGEngine(vector_store=get_vector_store())


@lru_cache()
def get_cv_model() -> MedicalCVModel:
    """Singleton CV model instance."""
    return MedicalCVModel()


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
    return SafetyController()


@lru_cache()
def get_report_generator() -> ReportGenerator:
    """Singleton Report generator instance."""
    return ReportGenerator()


def get_db_manager() -> DatabaseManager:
    """Dependency injection provider for DatabaseManager."""
    return db_manager


def get_case_repository() -> CaseRepository:
    """Dependency injection provider for CaseRepository."""
    return case_repository
