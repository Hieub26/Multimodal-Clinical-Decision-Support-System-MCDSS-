"""
Report Router: Case history and report retrieval endpoints.
"""

import json
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.api.schemas import (
    CaseHistoryItem, CaseHistoryResponse, HealthCheckResponse,
)
from app.config import settings
from app.db.database import get_case, get_all_cases, get_case_count
from app.api.dependencies import get_rag_engine, get_db_manager, require_api_key
from pathlib import Path

router = APIRouter(prefix="/reports", tags=["Reports & History"])

# Stored cases are patient data: every endpoint that returns them takes this
# guard. The health check stays open for container probes.
_case_access = [Depends(require_api_key)]


@router.get("/history", response_model=CaseHistoryResponse, dependencies=_case_access)
async def get_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
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


@router.get("/history/{case_id}", dependencies=_case_access)
async def get_case_detail(case_id: str):
    """Get detailed case information."""
    case = await get_case(case_id)
    if not case:
        raise HTTPException(404, f"Case {case_id} not found")

    # Safely handle diagnosis JSON whether it's already a dict or string
    result = dict(case)
    diag_json = result.get("diagnosis_json")
    if isinstance(diag_json, dict):
        result["diagnosis_detail"] = diag_json
    elif isinstance(diag_json, str):
        try:
            result["diagnosis_detail"] = json.loads(diag_json)
        except (json.JSONDecodeError, TypeError):
            result["diagnosis_detail"] = {}
    else:
        result["diagnosis_detail"] = {}

    return result


@router.get("/download/{case_id}", dependencies=_case_access)
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
async def health_check(response: Response):
    """System health check. Returns 503 when the database is unreachable."""
    components = {}

    try:
        await get_db_manager().ping()
        components["database"] = "healthy"
    except Exception as e:
        components["database"] = f"error: {str(e)}"

    try:
        rag = get_rag_engine()
        components["vector_store_docs"] = await run_in_threadpool(
            rag.vector_store.get_document_count
        )
        components["vector_store"] = "healthy"
    except Exception as e:
        components["vector_store"] = f"error: {str(e)}"

    components["cv_model"] = (
        "healthy"
        if Path(settings.cv_model_path).exists()
        else "error: model weights not found"
    )

    # Without the database no diagnosis can be stored; the other components
    # only take one modality offline.
    if components["database"] != "healthy":
        status = "unhealthy"
        response.status_code = 503
    elif components["vector_store"] != "healthy" or components["cv_model"] != "healthy":
        status = "degraded"
    else:
        status = "healthy"

    return HealthCheckResponse(
        status=status,
        version=settings.app_version,
        components=components,
    )
