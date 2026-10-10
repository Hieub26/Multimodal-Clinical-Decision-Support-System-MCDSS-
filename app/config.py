"""
Centralized configuration for the Multimodal Clinical Decision Support System.
Uses Pydantic Settings for environment-based configuration.
"""

from pathlib import Path
from typing import Any
from pydantic import Field, PositiveInt, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Project Paths (Consistent Path types) ---
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    storage_dir: Path = BASE_DIR / "storage"
    models_dir: Path = BASE_DIR / "models"
    chroma_persist_dir: Path = BASE_DIR / "data" / "chroma_db"
    cv_model_path: Path = BASE_DIR / "models" / "best_densenet121_chestxray14.pth"
    cv_model_meta_path: Path = BASE_DIR / "models" / "best_densenet121_chestxray14_meta.json"
    log_dir: Path = BASE_DIR / "storage" / "logs"
    image_storage_dir: Path = BASE_DIR / "storage" / "images"
    report_storage_dir: Path = BASE_DIR / "storage" / "reports"

    # --- App Info & Metadata ---
    app_title: str = "Multimodal Clinical Decision Support System"
    app_description: str = (
        "An AI-powered clinical decision support system combining NLP-based "
        "symptom analysis (RAG with ChromaDB) and medical image classification "
        "(PyTorch + Grad-CAM) for comprehensive diagnostic suggestions."
    )
    app_version: str = "1.0.0"
    api_prefix: str = "/api"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"

    # --- Server & CORS ---
    fastapi_host: str = "0.0.0.0"
    fastapi_port: PositiveInt = Field(default=8008, gt=0, le=65535)
    streamlit_port: PositiveInt = Field(default=8501, gt=0, le=65535)
    environment: str = "development"
    cors_origins: list[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8008",
        "http://127.0.0.1:8008",
    ]
    # When set, state-changing endpoints (guideline ingestion) require this
    # value in the X-API-Key header. Empty = no check (local development).
    admin_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key required by administrative endpoints"
    )
    max_upload_size_mb: PositiveInt = Field(default=20, gt=0)

    # --- LLM Provider ---
    gemini_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for Gemini LLM/VLM services"
    )
    llm_model_name: str = "gemini-2.5-flash"

    # --- Embedding Model ---
    embedding_model_name: str = "pritamdeka/S-BioBert-snli-multinli-stsb"

    # --- ChromaDB ---
    chroma_collection_name: str = "clinical_guidelines"

    # --- CV Model ---
    cv_num_classes: PositiveInt = Field(default=14, gt=0)
    cv_image_size: PositiveInt = Field(default=224, gt=0)
    cv_logit_temperature: float = Field(default=1.0, gt=0.0)
    cv_top_k: PositiveInt = Field(default=5, gt=0)
    cv_threshold_relaxation: float = Field(default=0.84, ge=0.0, le=1.0)
    cv_negative_screen_ratio_threshold: float = Field(default=0.65, ge=0.0, le=1.0)

    # Immutable tuple for class names
    cv_class_names: tuple[str, ...] = (
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    )

    # --- Safety & Confidence Thresholds (Validation with Field bounds) ---
    confidence_threshold_cv: float = Field(default=0.30, ge=0.0, le=1.0)
    confidence_threshold_nlp: float = Field(default=0.70, ge=0.0, le=1.0)
    safety_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    vlm_safety_enabled: bool = True
    vlm_safety_model: str = "gemini-2.5-flash"

    # --- Database ---
    # Holds the database password, so it is left out of the settings repr
    # (which ends up in tracebacks and test failure output).
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/clinical_dss",
        description="Async SQLAlchemy database connection URL",
        repr=False,
    )
    db_pool_min_size: PositiveInt = Field(default=2, gt=0)
    db_pool_max_size: PositiveInt = Field(default=10, gt=0)

    # --- Logging ---
    log_level: str = "INFO"

    # Configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @property
    def gemini_api_key_str(self) -> str:
        """Helper property to retrieve the raw string value of gemini_api_key."""
        return self.gemini_api_key.get_secret_value()

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories: tuple[Path, ...] = (
            self.data_dir,
            self.data_dir / "guidelines",
            self.data_dir / "chroma_db",
            self.data_dir / "sample_images",
            self.storage_dir,
            self.image_storage_dir,
            self.report_storage_dir,
            self.log_dir,
            self.models_dir,
        )
        for d in directories:
            d.mkdir(parents=True, exist_ok=True)


# Singleton settings instance
settings = Settings()
settings.ensure_directories()
