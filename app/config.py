"""
Centralized configuration for the Multimodal Clinical Decision Support System.
Uses Pydantic Settings for environment-based configuration.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


# Load .env file
load_dotenv()

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Project Paths ---
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    storage_dir: Path = BASE_DIR / "storage"
    models_dir: Path = BASE_DIR / "models"
    


    # --- LLM Provider ---
    gemini_api_key: str = ""
    llm_model_name: str = "gemini-2.5-flash"

    # --- Embedding Model ---
    embedding_model_name: str = "pritamdeka/S-BioBert-snli-multinli-stsb"

    # --- ChromaDB ---
    chroma_persist_dir: str = str(BASE_DIR / "data" / "chroma_db")
    chroma_collection_name: str = "clinical_guidelines"

    # --- CV Model ---
    cv_model_path: str = str(BASE_DIR / "models" / "best_densenet121_chestxray14.pth")
    cv_model_meta_path: str = str(BASE_DIR / "models" / "best_densenet121_chestxray14_meta.json")
    cv_num_classes: int = 14
    cv_image_size: int = 224
    cv_logit_temperature: float = 1.0
    cv_top_k: int = 5
    cv_threshold_relaxation: float = 0.84
    cv_class_names: list[str] = [
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
    ]

    # --- Safety Thresholds ---
    confidence_threshold_cv: float = 0.30
    confidence_threshold_nlp: float = 0.70
    safety_threshold: float = 0.15

    # --- Database ---
    database_url: str = "postgresql://postgres:hieu2611@localhost:5432/clinical_dss"

    # --- Server ---
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8008
    streamlit_port: int = 8501

    # --- Logging ---
    log_level: str = "INFO"
    log_dir: str = str(BASE_DIR / "storage" / "logs")

    # --- Image Storage ---
    image_storage_dir: str = str(BASE_DIR / "storage" / "images")
    report_storage_dir: str = str(BASE_DIR / "storage" / "reports")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_directories(self):
        """Create all required directories if they don't exist."""
        dirs = [
            self.data_dir,
            self.data_dir / "guidelines",
            self.data_dir / "chroma_db",
            self.data_dir / "sample_images",
            self.storage_dir,
            Path(self.image_storage_dir),
            Path(self.report_storage_dir),
            Path(self.log_dir),
            self.models_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# Singleton settings instance
settings = Settings()
settings.ensure_directories()
