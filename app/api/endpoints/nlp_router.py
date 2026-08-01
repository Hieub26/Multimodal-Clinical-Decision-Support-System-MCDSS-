"""
NLP Router: Text-based diagnosis and guideline management endpoints.
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import (
    TextDiagnosisRequest, DiagnosisResponse, DiagnosisResult,
    SafetyInfo, ValidationInfo, IngestRequest,
)
from app.api.dependencies import (
    get_rag_engine, get_fusion_engine, get_guideline_validator,
    get_safety_controller, get_report_generator,
)
from app.db.database import save_case
from app.utils.logger import api_logger

router = APIRouter(prefix="/nlp", tags=["NLP Diagnosis"])


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_text(request: TextDiagnosisRequest):
    """Perform text-based clinical diagnosis using RAG pipeline."""
    api_logger.info(f"Text diagnosis request received")

    if not request.symptoms_text and not request.clinical_question:
        raise HTTPException(400, "Provide symptoms_text or clinical_question")

    rag = get_rag_engine()
    fusion = get_fusion_engine()
    validator = get_guideline_validator()
    safety = get_safety_controller()
    report_gen = get_report_generator()

    # NLP diagnosis
    nlp_result = rag.diagnose_from_text(request.symptoms_text, request.clinical_question)

    # Fusion (single modality)
    fused = fusion.fuse(nlp_diagnosis=nlp_result)

    # Validate against guidelines
    validated = validator.validate(fused)

    # Safety check
    safe_result = safety.evaluate(
        validated,
        symptoms_text=request.symptoms_text,
        clinical_question=request.clinical_question,
    )

    # Generate report
    report_path = report_gen.generate_clinical_report(
        safe_result, request.symptoms_text, request.clinical_question
    )

    # Build response
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
        modality="text",
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
        report_path=report_path,
        disclaimer="This is an AI-generated analysis for educational purposes only.",
    )

    # Save to database
    await save_case({
        "case_id": case_id,
        "created_at": datetime.utcnow().isoformat(),
        "input_type": "text",
        "symptoms_text": request.symptoms_text,
        "clinical_question": request.clinical_question,
        "diagnosis": safe_result,
        "primary_diagnosis": safe_result.get("primary_diagnosis", ""),
        "confidence": safe_result.get("combined_confidence", 0),
        "safety_status": safe_result.get("safety_status", "unknown"),
        "report_path": report_path,
    })

    return response


@router.post("/ingest-guidelines")
async def ingest_guidelines(request: IngestRequest):
    """Ingest clinical guidelines into ChromaDB."""
    rag = get_rag_engine()

    if request.file_path:
        count = rag.vector_store.ingest_from_file(request.file_path)
    elif request.text:
        count = rag.vector_store.ingest_documents([request.text])
    else:
        raise HTTPException(400, "Provide text or file_path")

    return {"status": "success", "chunks_ingested": count}


@router.get("/guideline-count")
async def get_guideline_count():
    """Get the number of guideline chunks in the vector store."""
    rag = get_rag_engine()
    count = rag.vector_store.get_document_count()
    return {"count": count}
