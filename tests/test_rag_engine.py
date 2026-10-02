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
