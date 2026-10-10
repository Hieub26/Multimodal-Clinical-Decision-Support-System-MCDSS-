"""
Where the backend is and how to authenticate to it.
"""

import os

# API_BASE_URL is set by Docker and by run.py; localhost is the fallback when
# Streamlit is started on its own.
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8008/api")


def auth_headers() -> dict[str, str]:
    """The X-API-Key header, when the backend is configured to require one.

    Reads the same API_KEY variable as the backend. The key stays in this
    server process: it is never sent to the browser.
    """
    key = os.environ.get("API_KEY", "")
    return {"X-API-Key": key} if key else {}


def describe_failure(status_code: int, body: str) -> str:
    """A message for a failed backend call."""
    if status_code == 401:
        return (
            "The backend rejected the request (401). It requires an API key: "
            "start the frontend with the same API_KEY as the backend."
        )
    return f"API Error {status_code}: {body}"
