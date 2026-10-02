import asyncio
import io
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image
from pydantic import SecretStr

import app.api.dependencies as dependencies
from app.api.endpoints.cv_router import _read_validated_image
from app.api.endpoints.nlp_router import _resolve_guideline_file
from app.config import settings
from app.db.mapper import CaseMapper


def _upload(content: bytes, filename: str = "scan.png") -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_upload_extension_comes_from_content_not_filename():
    content, extension = asyncio.run(_read_validated_image(_upload(_png_bytes(), "evil.html")))
    assert extension == ".png"
    assert content == _png_bytes()


@pytest.mark.parametrize("content, status", [
    (b"", 400),
    (b"<html><script>alert(1)</script></html>", 400),
])
def test_upload_rejects_non_images(content, status):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(_read_validated_image(_upload(content)))
    assert exc.value.status_code == status


def test_upload_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(settings, "max_upload_size_mb", 1)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(_read_validated_image(_upload(b"\0" * (1024 * 1024 + 1))))
    assert exc.value.status_code == 413


def test_upload_rejects_unsupported_image_format():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buffer, format="GIF")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(_read_validated_image(_upload(buffer.getvalue())))
    assert exc.value.status_code == 415


@pytest.mark.parametrize("file_path", [
    "../../.env",
    "../../app/config.py",
    str(settings.base_dir / ".env.example"),
    "cardiology/missing.txt",
])
def test_ingest_path_is_confined_to_guidelines_dir(file_path):
    with pytest.raises(HTTPException) as exc:
        _resolve_guideline_file(file_path)
    assert exc.value.status_code == 400


def test_ingest_path_accepts_guideline_file():
    path, source = _resolve_guideline_file("cardiology/cardiovascular_diseases.txt")
    assert path.is_file()
    assert source == "cardiology/cardiovascular_diseases.txt"


def test_admin_key_enforced_only_when_configured(monkeypatch):
    dependencies.require_admin_key(None)  # not configured: open

    monkeypatch.setattr(settings, "admin_api_key", SecretStr("s3cret"))
    dependencies.require_admin_key("s3cret")
    for bad in (None, "", "wrong"):
        with pytest.raises(HTTPException) as exc:
            dependencies.require_admin_key(bad)
        assert exc.value.status_code == 401


def test_naive_timestamps_are_stored_as_utc():
    """asyncpg reads a naive datetime as local time for TIMESTAMPTZ."""
    naive = datetime(2026, 1, 1, 12, 0, 0)
    for created_at in (naive, naive.isoformat()):
        record = CaseMapper.to_record({"case_id": "c1", "created_at": created_at})
        assert record.created_at == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
