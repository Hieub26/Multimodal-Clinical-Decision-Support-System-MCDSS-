"""Checks on the dependency and deployment files."""

import ast
import re
import sys

from app.config import BASE_DIR, Settings


def _pins(filename: str) -> dict[str, str]:
    pins = {}
    for line in (BASE_DIR / filename).read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==(\S+)", line.strip())
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def test_frontend_pins_match_the_backend_pins():
    backend = _pins("requirements.txt")
    frontend = _pins("requirements-frontend.txt")

    assert frontend, "requirements-frontend.txt lists no pinned package"
    for package, version in frontend.items():
        assert backend.get(package) == version, package


def test_frontend_imports_only_what_its_requirements_install():
    """The frontend image has no backend code and no backend dependencies."""
    installed = {"streamlit", "httpx", "PIL"}
    local = {"components", "api_client"}

    for path in (BASE_DIR / "streamlit_app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                assert (
                    top in installed or top in local or top in sys.stdlib_module_names
                ), f"{path.name} imports {module}"


def test_every_variable_in_env_example_is_a_setting_or_a_compose_variable():
    compose_only = {"POSTGRES_USER", "POSTGRES_PASSWORD", "BIND_ADDRESS"}
    fields = {name.upper() for name in Settings.model_fields}

    for line in (BASE_DIR / ".env.example").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]*)=", line)
        if match:
            assert match.group(1) in fields | compose_only, match.group(1)


def test_settings_repr_holds_no_secret():
    """The repr ends up in tracebacks and in test failure output."""
    configured = Settings(
        gemini_api_key="gemini-secret", openai_api_key="openai-secret",
        typesafe_api_key="typesafe-secret", api_key="client-secret",
        admin_api_key="admin-secret",
        database_url="postgresql://user:database-secret@host:5432/db",
    )
    shown = repr(configured) + str(configured)

    for secret in ("gemini-secret", "openai-secret", "typesafe-secret",
                   "client-secret", "admin-secret", "database-secret"):
        assert secret not in shown
