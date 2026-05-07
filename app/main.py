"""
FastAPI Application Entry Point.
Orchestrates all components of the Multimodal Clinical Decision Support System.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.config import settings
from app.db.database import init_db
from app.api.endpoints.nlp_router import router as nlp_router
from app.api.endpoints.cv_router import router as cv_router
from app.api.endpoints.report_router import router as report_router
from app.utils.logger import api_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    api_logger.info("=" * 60)
    api_logger.info("Starting Multimodal Clinical Decision Support System")
    api_logger.info("=" * 60)

    # Initialize database
    await init_db()
    api_logger.info("Database initialized")

    # Initialize guidelines on first run
    try:
        from app.api.dependencies import get_rag_engine
        rag = get_rag_engine()
        rag.initialize_guidelines()
        api_logger.info("Clinical guidelines loaded")
    except Exception as e:
        api_logger.warning(f"Guideline initialization skipped: {e}")

    # Ensure storage directories exist
    settings.ensure_directories()

    api_logger.info(f"Server ready on http://{settings.fastapi_host}:{settings.fastapi_port}")
    api_logger.info(f"API docs at http://localhost:{settings.fastapi_port}/docs")

    yield

    api_logger.info("Shutting down Clinical DSS")


# Create FastAPI application
app = FastAPI(
    title="Multimodal Clinical Decision Support System",
    description=(
        "An AI-powered clinical decision support system combining NLP-based "
        "symptom analysis (RAG with ChromaDB) and medical image classification "
        "(PyTorch + Grad-CAM) for comprehensive diagnostic suggestions."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for images
images_dir = Path(settings.image_storage_dir)
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=str(images_dir)), name="images")

# Include routers
app.include_router(nlp_router, prefix="/api")
app.include_router(cv_router, prefix="/api")
app.include_router(report_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "system": "Multimodal Clinical Decision Support System",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "text_diagnosis": "/api/nlp/diagnose",
            "image_diagnosis": "/api/cv/diagnose",
            "history": "/api/reports/history",
            "health": "/api/reports/health",
        },
    }
