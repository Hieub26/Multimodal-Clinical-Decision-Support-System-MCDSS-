"""
Report Router: Case history and report retrieval endpoints.
"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.api.schemas import (
    CaseHistoryItem, CaseHistoryResponse, HealthCheckResponse,
)
from app.db.database import get_case, get_all_cases, get_case_count
from app.api.dependencies import get_rag_engine, get_cv_model
from app.utils.logger import api_logger
from pathlib import Path

router = APIRouter(prefix="/reports", tags=["Reports & History"])


@router.get("/history", response_model=CaseHistoryResponse)
async def get_history(limit: int = 50, offset: int = 0):
    """Get case history with pagination."""
    cases = await get_all_cases(limit, offset)
    total = await get_case_count()

    items = [
        CaseHistoryItem(
            case_id=c["case_id"],
            created_at=c["created_at"],
            input_type=c["input_type"],
            primary_diagnosis=c.get("primary_diagnosis", ""),
            confidence=c.get("confidence", 0),
            safety_status=c.get("safety_status", ""),
        )
        for c in cases
    ]

    return CaseHistoryResponse(total=total, cases=items)


@router.get("/history/{case_id}")
async def get_case_detail(case_id: str):
    """Get detailed case information."""
    case = await get_case(case_id)
    if not case:
        raise HTTPException(404, f"Case {case_id} not found")

    # Parse diagnosis JSON
    result = dict(case)
    if result.get("diagnosis_json"):
        try:
            result["diagnosis_detail"] = json.loads(result["diagnosis_json"])
        except json.JSONDecodeError:
            result["diagnosis_detail"] = {}

    return result


@router.get("/download/{case_id}")
async def download_report(case_id: str):
    """Download the clinical report file for a case."""
    case = await get_case(case_id)
    if not case:
        raise HTTPException(404, f"Case {case_id} not found")

    report_path = case.get("report_path")
    if not report_path or not Path(report_path).exists():
        raise HTTPException(404, "Report file not found")

    return FileResponse(
        report_path,
        media_type="text/markdown",
        filename=Path(report_path).name,
    )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """System health check."""
    components = {
        "database": "healthy",
        "vector_store": "healthy",
    }

    try:
        rag = get_rag_engine()
        doc_count = rag.vector_store.get_document_count()
        components["vector_store_docs"] = doc_count
    except Exception as e:
        components["vector_store"] = f"error: {str(e)}"

    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        components=components,
    )
