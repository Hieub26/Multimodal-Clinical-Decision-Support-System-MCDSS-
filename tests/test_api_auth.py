"""Access control of the endpoints that create or return patient cases."""

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import app.api.endpoints.nlp_router as nlp_module
import app.api.endpoints.report_router as report_module
from app.config import settings
from app.main import app

API_KEY = "client-key"
ADMIN_KEY = "admin-key"

CASE = {
    "case_id": "c1",
    "created_at": "2026-10-10T00:00:00Z",
    "input_type": "text",
    "primary_diagnosis": "Pneumonia",
    "confidence": 0.8,
    "safety_status": "approved",
    "diagnosis_json": {"primary_diagnosis": "Pneumonia"},
    "report_path": None,
}


@pytest.fixture
def client(monkeypatch):
    """The real application without its startup (no database, no models)."""
    async def all_cases(limit, offset):
        return [CASE]

    async def case_count():
        return 1

    async def one_case(case_id):
        return CASE if case_id == "c1" else None

    monkeypatch.setattr(report_module, "get_all_cases", all_cases)
    monkeypatch.setattr(report_module, "get_case_count", case_count)
    monkeypatch.setattr(report_module, "get_case", one_case)

    def no_pipeline(*args):
        raise AssertionError("the diagnosis pipeline ran for an unauthenticated request")

    monkeypatch.setattr(nlp_module, "_run_text_pipeline", no_pipeline)
    return TestClient(app)


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setattr(settings, "api_key", SecretStr(API_KEY))
    monkeypatch.setattr(settings, "admin_api_key", SecretStr(ADMIN_KEY))


CASE_REQUESTS = [
    ("get", "/api/reports/history", {}),
    ("get", "/api/reports/history/c1", {}),
    ("get", "/api/reports/download/c1", {}),
    ("post", "/api/nlp/diagnose", {"json": {"symptoms_text": "cough"}}),
    ("post", "/api/cv/diagnose", {"files": {"image": ("scan.png", b"not an image", "image/png")}}),
]


@pytest.mark.parametrize("method, path, kwargs", CASE_REQUESTS)
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": ""}, {"X-API-Key": "wrong"}])
def test_case_endpoints_reject_requests_without_the_key(client, keys, method, path, kwargs, headers):
    response = getattr(client, method)(path, headers=headers, **kwargs)
    assert response.status_code == 401


@pytest.mark.parametrize("key", [API_KEY, ADMIN_KEY])
def test_case_history_is_returned_with_the_client_or_the_admin_key(client, keys, key):
    headers = {"X-API-Key": key}

    listing = client.get("/api/reports/history", headers=headers)
    assert listing.status_code == 200
    assert listing.json()["cases"][0]["case_id"] == "c1"

    detail = client.get("/api/reports/history/c1", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["diagnosis_detail"] == {"primary_diagnosis": "Pneumonia"}


def test_without_a_configured_key_the_endpoints_stay_open(client):
    assert client.get("/api/reports/history").status_code == 200


def test_health_and_root_need_no_key(client, keys):
    assert client.get("/").status_code == 200
    # Reports the database as unreachable here, but is not an auth failure
    assert client.get("/api/reports/health").status_code != 401


def test_client_key_does_not_open_admin_endpoints(client, keys):
    response = client.post(
        "/api/nlp/ingest-guidelines",
        headers={"X-API-Key": API_KEY},
        json={"text": "A guideline"},
    )
    assert response.status_code == 401
