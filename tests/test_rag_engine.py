import pytest

from app.core.nlp.rag_engine import RAGEngine


@pytest.fixture(scope="module")
def engine():
    # Collaborators are not exercised by these tests
    return RAGEngine(
        vector_store=object(),
        text_preprocessor=object(),
        question_understanding=object(),
        fallback_engine=object(),
    )


@pytest.mark.parametrize("raw, expected", [
    (0.85, 0.85),
    ("0.85", 0.85),
    ("85%", 0.85),
    (85, 0.85),
    (None, 0.0),
    ("high", 0.0),
    (-1, 0.0),
    (float("nan"), 0.0),
])
def test_confidence_is_coerced_to_unit_interval(engine, raw, expected):
    assert engine._coerce_score(raw) == pytest.approx(expected)


def test_sanitize_fills_null_and_mistyped_fields(engine):
    result = engine._sanitize_llm_result({
        "primary_diagnosis": None,
        "confidence": "90%",
        "explanation": None,
        "recommended_actions": "See a doctor",
        "differential_diagnoses": [{"condition": "Asthma", "probability": "20%"}, "junk"],
    })

    assert result["primary_diagnosis"] == "Analysis completed"
    assert result["confidence"] == pytest.approx(0.9)
    assert result["explanation"] == ""
    assert result["recommended_actions"] == ["See a doctor"]
    assert result["supporting_evidence"] == []
    assert result["differential_diagnoses"] == [
        {"condition": "Asthma", "probability": pytest.approx(0.2)}
    ]


def test_sanitize_rejects_non_object(engine):
    with pytest.raises(ValueError):
        engine._sanitize_llm_result(["not", "an", "object"])


@pytest.mark.parametrize("severity, urgency, expected", [
    ("critical", "routine", "emergency"),   # raised to match severity
    ("high", "routine", "urgent"),
    ("high", "emergency", "emergency"),     # never lowered
    ("moderate", "urgent", "urgent"),
    ("low", "N/A", "routine"),              # invalid value sanitized
])
def test_urgency_is_raised_never_lowered(engine, severity, urgency, expected):
    result = engine._normalize_urgency({"severity": severity, "urgency": urgency})
    assert result["urgency"] == expected


# --- The LLM call: cache, cooldown after a failure -------------------------

ANSWER = (
    '{"primary_diagnosis": "Pneumonia", "confidence": 0.8, "explanation": "x",'
    ' "severity": "high", "urgency": "urgent"}'
)


class FakeLLM:
    """Stands in for genai.Client: replays the given responses in order."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = 0
        self.models = self

    def generate_content(self, **request):
        self.calls += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return type("Response", (), {"text": response})()


class FakeFallback:
    def __init__(self):
        self.calls = 0

    def diagnose(self, user_input, retrieved_docs):
        self.calls += 1
        return {"primary_diagnosis": "Fallback", "confidence": 0.2}


def _engine_with(*responses):
    engine = RAGEngine(
        vector_store=object(), text_preprocessor=object(),
        question_understanding=object(), fallback_engine=FakeFallback(),
    )
    engine._client = FakeLLM(*responses)
    return engine


def test_identical_request_is_answered_from_the_cache():
    engine = _engine_with(ANSWER, ANSWER)

    first = engine._generate_diagnosis("guidelines", "cough and fever")
    first["retrieved_documents"] = ["added by the caller"]
    second = engine._generate_diagnosis("guidelines", "cough and fever")

    assert engine._client.calls == 1
    assert second["primary_diagnosis"] == "Pneumonia"
    assert "retrieved_documents" not in second  # the caller's changes are not shared

    engine._generate_diagnosis("guidelines", "a different complaint")
    assert engine._client.calls == 2


def test_failed_call_falls_back_and_pauses_the_llm(monkeypatch):
    import app.core.nlp.rag_engine as module

    engine = _engine_with(TimeoutError("timed out"), ANSWER)

    assert engine._generate_diagnosis("guidelines", "cough")["primary_diagnosis"] == "Fallback"
    # During the cooldown the LLM is not called again
    assert engine._generate_diagnosis("guidelines", "fever")["primary_diagnosis"] == "Fallback"
    assert engine._client.calls == 1
    assert engine._fallback.calls == 2

    now = module.time.monotonic()
    monkeypatch.setattr(
        module.time, "monotonic", lambda: now + module._FAILURE_COOLDOWN_SECONDS + 1
    )
    assert engine._generate_diagnosis("guidelines", "cough")["primary_diagnosis"] == "Pneumonia"


def test_unusable_answer_falls_back_without_a_pause_or_a_cache_entry():
    engine = _engine_with('["not", "an", "object"]', ANSWER)

    assert engine._generate_diagnosis("guidelines", "cough")["primary_diagnosis"] == "Fallback"
    # Same prompt again: asked again, and this time answered
    assert engine._generate_diagnosis("guidelines", "cough")["primary_diagnosis"] == "Pneumonia"
    assert engine._client.calls == 2


def test_answer_recovered_from_malformed_json_is_not_cached():
    broken = '{"primary_diagnosis": "Pneumonia", "confidence": 0.8, "explanation": "a "quoted" word"}'
    engine = _engine_with(broken, broken)

    assert engine._generate_diagnosis("guidelines", "cough")["primary_diagnosis"] == "Pneumonia"
    engine._generate_diagnosis("guidelines", "cough")
    assert engine._client.calls == 2


def test_llm_client_is_given_the_configured_timeout(monkeypatch):
    import app.core.nlp.rag_engine as module
    from app.config import settings

    created = {}

    class Client:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(module.genai, "Client", Client)
    monkeypatch.setattr(settings, "gemini_api_key", type(settings.gemini_api_key)("key"))
    monkeypatch.setattr(settings, "llm_timeout_seconds", 12.5)
    engine = RAGEngine(
        vector_store=object(), text_preprocessor=object(),
        question_understanding=object(), fallback_engine=object(),
    )

    engine._ensure_llm()

    assert created["http_options"].timeout == 12500  # milliseconds
