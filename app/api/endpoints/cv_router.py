"""
CV Router: Image-based diagnosis endpoints.
"""

import io
import uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from app.api.schemas import (
    DiagnosisResponse, DiagnosisResult, SafetyInfo, ValidationInfo,
)
from app.api.dependencies import (
    get_cv_model, get_fusion_engine, get_guideline_validator,
    get_safety_controller, get_report_generator, get_rag_engine,
)
from app.config import settings
from app.core.cv.cv_model import ModelWeightsNotFoundError
from app.db.database import save_case
from app.utils.logger import api_logger

router = APIRouter(prefix="/cv", tags=["CV Diagnosis"])

# Accepted formats and the extension each is stored under. The extension
# comes from the decoded content, never from the client-supplied filename.
_IMAGE_EXTENSIONS = {"PNG": ".png", "JPEG": ".jpg", "BMP": ".bmp", "TIFF": ".tiff"}


async def _read_validated_image(image: UploadFile) -> tuple[bytes, str]:
    """Read an upload, enforcing the size limit and that it decodes as an image.

    Returns:
        (file content, extension to store it under)
    """
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await image.read(max_bytes + 1)
    if not content:
        raise HTTPException(400, "Uploaded image is empty")
    if len(content) > max_bytes:
        raise HTTPException(
            413, f"Image exceeds the {settings.max_upload_size_mb} MB upload limit"
        )

    try:
        with Image.open(io.BytesIO(content)) as img:
            image_format = img.format
            img.verify()
    except Exception:
        raise HTTPException(400, "Uploaded file is not a valid image")

    extension = _IMAGE_EXTENSIONS.get(image_format)
    if extension is None:
        raise HTTPException(
            415,
            f"Unsupported image format '{image_format}'. "
            f"Supported: {', '.join(_IMAGE_EXTENSIONS)}",
        )
    return content, extension


def _run_image_pipeline(
    image_path: str, symptoms_text: str | None, clinical_question: str | None
) -> tuple[dict, dict | None, dict, str, str]:
    """Blocking diagnosis pipeline (model inference, LLM calls, file I/O).

    Runs in a worker thread so it does not stall the event loop.
    """
    # CV diagnosis
    cv_model = get_cv_model()
    cv_result = cv_model.predict(image_path)

    # Optional NLP diagnosis for multimodal
    nlp_result = None
    modality = "image"
    if symptoms_text or clinical_question:
        rag = get_rag_engine()
        nlp_result = rag.diagnose_from_text(symptoms_text, clinical_question)
        modality = "multimodal"

    # Fusion
    fusion = get_fusion_engine()
    fused = fusion.fuse(nlp_diagnosis=nlp_result, cv_diagnosis=cv_result)

    # Validate
    validator = get_guideline_validator()
    validated = validator.validate(fused)

    # Safety check
    safety = get_safety_controller()
    safe_result = safety.evaluate(
        validated,
        image_path=image_path,
        symptoms_text=symptoms_text,
        clinical_question=clinical_question,
    )

    # Generate report
    report_gen = get_report_generator()
    report_path = report_gen.generate_clinical_report(
        safe_result, symptoms_text, clinical_question
    )
    return safe_result, nlp_result, cv_result, modality, report_path


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_image(
    image: UploadFile = File(...),
    symptoms_text: str = Form(None),
    clinical_question: str = Form(None),
):
    """
    Perform image-based (or multimodal) clinical diagnosis.
    Upload a medical image and optionally provide symptoms text.
    """
    # The client's file name can carry a patient's name: it is not logged
    api_logger.info("Image diagnosis request received")

    content, extension = await _read_validated_image(image)

    # Save uploaded image
    image_dir = Path(settings.image_storage_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    image_filename = f"upload_{uuid.uuid4().hex[:8]}{extension}"
    image_path = str(image_dir / image_filename)

    with open(image_path, "wb") as f:
        f.write(content)

    api_logger.info(f"Image saved: {image_path}")

    try:
        safe_result, nlp_result, cv_result, modality, report_path = (
            await run_in_threadpool(
                _run_image_pipeline, image_path, symptoms_text, clinical_question
            )
        )
    except ModelWeightsNotFoundError as e:
        Path(image_path).unlink(missing_ok=True)
        api_logger.error(f"CV model unavailable: {e}")
        raise HTTPException(503, "Image analysis is unavailable: CV model weights are not installed")
    except Exception:
        # Do not keep an upload that produced no case
        Path(image_path).unlink(missing_ok=True)
        raise

    case_id = str(uuid.uuid4())[:12]
    response = DiagnosisResponse(
        case_id=case_id,
        diagnosis=DiagnosisResult(
            primary_diagnosis=safe_result.get("primary_diagnosis", ""),
            confidence=safe_result.get("combined_confidence", 0),
            explanation=safe_result.get("explanation", ""),
            question_answer=safe_result.get("question_answer", ""),
            severity=safe_result.get("severity", "unknown"),
            urgency=safe_result.get("urgency", "routine"),
            recommended_actions=safe_result.get("recommended_actions", []),
        ),
        modality=modality,
        fusion_strategy=safe_result.get("fusion_strategy", "single"),
        safety=SafetyInfo(
            status=safe_result.get("safety_status", "unknown"),
            is_approved=safe_result.get("is_approved", False),
            risk_factors=safe_result.get("risk_factors", []),
            message=safe_result.get("safety_message", ""),
        ),
        validation=ValidationInfo(
            guideline_consistent=safe_result.get("is_guideline_consistent", False),
            support_score=safe_result.get("guideline_support_score", 0),
            notes=safe_result.get("validation_notes", []),
            method=safe_result.get("validation_method", ""),
            judge_model=safe_result.get("judge_model"),
        ),
        nlp_details=nlp_result,
        cv_details=cv_result,
        vlm_safety_review=safe_result.get("vlm_safety_review"),
        uncertainty=cv_result.get("uncertainty"),
        severity_score=safe_result.get("severity_score"),
        severity_breakdown=safe_result.get("severity_breakdown"),
        report_path=report_path,
        gradcam_path=cv_result.get("gradcam_path", ""),
        disclaimer="This is an AI-generated analysis for educational purposes only.",
    )

    # Save to database
    await save_case({
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc),
        "input_type": modality,
        "symptoms_text": symptoms_text,
        "clinical_question": clinical_question,
        "image_path": image_path,
        "diagnosis": safe_result,
        "primary_diagnosis": safe_result.get("primary_diagnosis", ""),
        "confidence": safe_result.get("combined_confidence", 0),
        "safety_status": safe_result.get("safety_status", "unknown"),
        "report_path": report_path,
    })

    return response
