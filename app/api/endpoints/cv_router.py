"""
CV Router: Image-based diagnosis endpoints.
"""

import uuid
import shutil
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException

from app.api.schemas import (
    DiagnosisResponse, DiagnosisResult, SafetyInfo, ValidationInfo,
)
from app.api.dependencies import (
    get_cv_model, get_fusion_engine, get_guideline_validator,
    get_safety_controller, get_report_generator, get_rag_engine,
)
from app.config import settings
from app.db.database import save_case
from app.utils.logger import api_logger

router = APIRouter(prefix="/cv", tags=["CV Diagnosis"])


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
    api_logger.info(f"Image diagnosis request: {image.filename}")

    # Save uploaded image
    image_dir = Path(settings.image_storage_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(image.filename).suffix or ".png"
    image_filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    image_path = str(image_dir / image_filename)

    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)

    api_logger.info(f"Image saved: {image_path}")

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
        "created_at": datetime.utcnow().isoformat(),
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
