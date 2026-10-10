"""Rule-based fallback diagnosis engine: ranking, hallmark symptoms, confidence."""

import os
import subprocess
import sys

import pytest

from app.config import BASE_DIR, settings
from app.core.nlp.clinical_fallback import (
    CONFIDENCE_BY_MARGIN, FALLBACK_CONFIDENCE_CAP, NO_SYMPTOM_CONFIDENCE_CAP,
    SINGLE_SYMPTOM_CONFIDENCE_CAP, ClinicalFallbackEngine,
)


# A plain number: an assertion that names `settings` makes pytest print the
# whole settings object when it fails
THRESHOLD = settings.confidence_threshold_nlp


@pytest.fixture(scope="module")
def engine():
    return ClinicalFallbackEngine()


def diagnose(engine, text):
    result = engine.diagnose(text, [])
    return result, result["fallback_reasoning"]


# ----------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------

def test_equal_scores_are_ranked_by_severity_then_by_profile_order(engine):
    # "dizziness" gives anemia (moderate) and stroke (critical) 2 points each
    _, reasoning = diagnose(engine, "I feel dizzy")
    assert reasoning["final_scores"]["anemia"] == reasoning["final_scores"]["stroke"]
    assert reasoning["top_disease"] == "stroke"

    # "fever" gives pneumonia, COVID-19 and tuberculosis (all high) 2 points each
    _, reasoning = diagnose(engine, "I have a fever")
    assert reasoning["top_disease"] == "pneumonia"


def test_diagnosis_does_not_depend_on_the_hash_seed():
    """The leading disease among equal scores once followed set iteration
    order, which changes with every interpreter start."""
    script = (
        "from app.core.nlp.clinical_fallback import ClinicalFallbackEngine as E;"
        "r = E().diagnose('fever, cough, headache, dizziness, nausea, fatigue and chills', []);"
        "print(r['primary_diagnosis'], r['confidence'], r['fallback_reasoning']['runner_up_disease'])"
    )
    outputs = set()
    for seed in ("1", "2", "3"):
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=BASE_DIR, capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": seed, "LOG_LEVEL": "ERROR"},
        )
        assert completed.returncode == 0, completed.stderr
        outputs.add(completed.stdout.strip().splitlines()[-1])
    assert len(outputs) == 1, outputs


# ----------------------------------------------------------------------
# Hallmark symptoms from the guideline library
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text, disease", [
    ("It burns when I pee and I have to pee all the time. My urine is cloudy.", "uti"),
    ("I'm always thirsty, I pee a lot more than usual and I've lost weight.", "diabetes"),
    ("I have an itchy rash with blisters on my arms.", "skin_condition"),
    ("My face is drooping and my speech is slurred.", "stroke"),
])
def test_hallmark_symptoms_lead_to_their_disease_with_confidence(engine, text, disease):
    result, reasoning = diagnose(engine, text)

    assert reasoning["top_disease"] == disease
    assert len(reasoning["supporting_symptoms"]) >= 2
    assert result["confidence"] >= THRESHOLD


def test_a_hallmark_symptom_leads_without_settling_a_shared_picture(engine):
    """Loss of smell puts COVID-19 first, but fever and cough fit pneumonia
    almost as well: named, not confident."""
    result, reasoning = diagnose(engine, "I have lost my sense of smell and I have a fever and a cough.")

    assert reasoning["top_disease"] == "covid19"
    assert reasoning["runner_up_disease"] == "pneumonia"
    assert result["confidence"] < THRESHOLD


def test_symptoms_shared_by_several_diseases_stay_below_the_threshold(engine):
    """Fever, cough and sputum fit pneumonia best, and tuberculosis and
    COVID-19 too: the engine names pneumonia without claiming to be sure."""
    result, reasoning = diagnose(engine, "I have had a cough, a fever and yellow phlegm for 3 days")

    assert reasoning["top_disease"] == "pneumonia"
    assert reasoning["supporting_symptoms"] == ["cough", "fever", "sputum"]
    assert result["confidence"] < THRESHOLD


def test_one_symptom_is_never_a_confident_diagnosis(engine):
    result, reasoning = diagnose(engine, "I have a rash")

    assert reasoning["top_disease"] == "skin_condition"
    assert reasoning["runner_up_disease"] is None  # nothing else scored at all
    assert result["confidence"] == SINGLE_SYMPTOM_CONFIDENCE_CAP
    assert result["confidence"] < THRESHOLD


def test_dyspnea_and_shortness_of_breath_are_one_symptom(engine):
    clinical, _ = diagnose(engine, "wheezing and dyspnea")
    everyday, _ = diagnose(engine, "wheezing and shortness of breath")

    assert clinical["fallback_reasoning"]["final_scores"] == everyday["fallback_reasoning"]["final_scores"]
    assert clinical["primary_diagnosis"] == everyday["primary_diagnosis"]


def test_nothing_recognized_gives_the_lowest_confidence(engine):
    result, reasoning = diagnose(engine, "I feel a bit off today")

    assert result["primary_diagnosis"] == "General clinical assessment needed"
    assert result["confidence"] == CONFIDENCE_BY_MARGIN[0][1]
    assert reasoning["supporting_symptoms"] == []


# ----------------------------------------------------------------------
# Confidence curve
# ----------------------------------------------------------------------

confidence = ClinicalFallbackEngine._confidence


def test_confidence_grows_with_the_margin_not_with_the_score():
    margins = [confidence(10, runner_up, 3) for runner_up in (10, 8, 6, 5, 4, 3, 2, 1, 0)]
    assert margins == sorted(margins)
    # The same margin at three sizes of score
    assert confidence(4, 1, 3) == confidence(12, 3, 3) == confidence(40, 10, 3)


@pytest.mark.parametrize("top, runner_up, expected", [
    (10, 10, 0.10),   # a tie
    (10, 5, 0.25),
    (10, 3, 0.70),    # margin 0.7: the threshold
    (10, 1, 0.75),
    (10, 0, 0.75),
])
def test_confidence_at_the_knots(top, runner_up, expected):
    assert confidence(top, runner_up, 3) == expected


def test_confidence_is_rounded_down_at_the_threshold():
    assert confidence(10, 3.01, 3) < 0.70


def test_confidence_caps():
    assert confidence(50, 0, 5) == FALLBACK_CONFIDENCE_CAP
    assert confidence(50, 0, 1) == SINGLE_SYMPTOM_CONFIDENCE_CAP
    assert confidence(50, 0, 0) == NO_SYMPTOM_CONFIDENCE_CAP
    assert NO_SYMPTOM_CONFIDENCE_CAP < SINGLE_SYMPTOM_CONFIDENCE_CAP < THRESHOLD
