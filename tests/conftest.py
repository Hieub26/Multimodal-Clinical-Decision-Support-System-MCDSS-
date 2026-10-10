"""
Test configuration.

Runs before any app module is imported: points storage, logs and the vector
store at a temporary directory and blanks the API keys, so the suite never
touches real data or calls an external API.
"""

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="mcdss_tests_"))

os.environ.update({
    "GEMINI_API_KEY": "",
    "OPENAI_API_KEY": "",
    "TYPESAFE_API_KEY": "",
    "ADMIN_API_KEY": "",
    "STORAGE_DIR": str(_TMP / "storage"),
    "LOG_DIR": str(_TMP / "logs"),
    "IMAGE_STORAGE_DIR": str(_TMP / "storage" / "images"),
    "REPORT_STORAGE_DIR": str(_TMP / "storage" / "reports"),
    "CHROMA_PERSIST_DIR": str(_TMP / "chroma_db"),
})
