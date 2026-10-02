"""
FastAPI Application Entry Point.
Orchestrates all components of the Multimodal Clinical Decision Support System.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.database import init_db, close_db
from app.api.endpoints.nlp_router import router as nlp_router
from app.api.endpoints.cv_router import router as cv_router
from app.api.endpoints.report_router import router as report_router
from app.utils.logger import api_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events with critical & non-critical error handling."""
    api_logger.info("=" * 60)
    api_logger.info(f"Starting {settings.app_title} (v{settings.app_version})")
    api_logger.info(f"Environment: {settings.environment}")
    api_logger.info("=" * 60)

    # --- Critical Startup Tasks ---
    # Failures here stop the application startup immediately.
    try:
        settings.ensure_directories()
        api_logger.info("Storage directories verified")

        await init_db()
        api_logger.info("Database initialized successfully")
    except Exception as e:
        api_logger.critical(f"Critical startup failure: {e}", exc_info=True)
        raise RuntimeError(f"Application startup aborted due to critical error: {e}") from e

    # --- Non-Critical Startup Tasks ---
    # Failures here log a warning and allow application startup to continue.
    # Model loading and embedding are blocking; keep them off the event loop.
    try:
        from app.api.dependencies import get_rag_engine
        rag = get_rag_engine()
        await run_in_threadpool(rag.initialize_guidelines)
        api_logger.info("Clinical guidelines loaded successfully")
    except Exception as e:
        api_logger.warning(f"Non-critical startup warning (Guideline initialization skipped): {e}")

    try:
        from app.api.dependencies import get_cv_model
        cv = get_cv_model()
        # trigger lazy-load now, not on first request
        await run_in_threadpool(lambda: cv.model)
        api_logger.info("CV model preloaded into memory successfully")
    except Exception as e:
        api_logger.warning(
            f"Non-critical startup warning (CV model unavailable — image "
            f"diagnosis will return 503): {e}"
        )

    api_logger.info(f"Server ready on http://{settings.fastapi_host}:{settings.fastapi_port}")
    api_logger.info(f"API docs at http://localhost:{settings.fastapi_port}{settings.docs_url}")

    yield

    # Graceful shutdown
    await close_db()
    api_logger.info(f"Shutting down {settings.app_title}")


# Create FastAPI application using settings
app = FastAPI(
    title=settings.app_title,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
)

# CORS middleware configured via environment settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uploaded images and Grad-CAM outputs are patient data: they are not served
# over HTTP. The Streamlit frontend reads them from the shared storage volume.

# Include routers using configurable API prefix
app.include_router(nlp_router, prefix=settings.api_prefix)
app.include_router(cv_router, prefix=settings.api_prefix)
app.include_router(report_router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "system": settings.app_title,
        "version": settings.app_version,
        "environment": settings.environment,
        "docs": settings.docs_url,
        "endpoints": {
            "text_diagnosis": f"{settings.api_prefix}/nlp/diagnose",
            "image_diagnosis": f"{settings.api_prefix}/cv/diagnose",
            "history": f"{settings.api_prefix}/reports/history",
            "health": f"{settings.api_prefix}/reports/health",
        },
    }
