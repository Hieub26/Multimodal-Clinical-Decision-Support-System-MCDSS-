"""What a CV flag is worth: validated precision, relaxed thresholds, severity."""

import numpy as np
import pytest

from app.core.cv.cv_postprocessor import MIN_PRECISION_FOR_CRITICAL, CVPostprocessor
from app.core.fusion.fusion_engine import FusionEngine

CLASS_NAMES = ["Edema", "Pneumothorax", "Effusion", "Nodule"]
# Validated on the model's validation set...
VALIDATION = [
    {"threshold": 0.92, "precision": 0.17, "recall": 0.38},
    {"threshold": 0.82, "precision": 0.35, "recall": 0.53},
    {"threshold": 0.76, "precision": 0.47, "recall": 0.63},
    {"threshold": 0.75, "precision": 0.27, "recall": 0.33},
]
# ...and relaxed for detection (CV_THRESHOLD_RELAXATION = 0.88)
THRESHOLDS = [round(v["threshold"] * 0.88, 4) for v in VALIDATION]


def postprocess(validation=VALIDATION, **scores):
    probs = np.full(len(CLASS_NAMES), 0.05)
    for name, score in scores.items():
        probs[CLASS_NAMES.index(name)] = score
    return CVPostprocessor().postprocess(probs, CLASS_NAMES, THRESHOLDS, validation)


def test_flag_carries_the_precision_measured_on_validation():
    result = postprocess(Effusion=0.85)

    assert result["confidence"] == pytest.approx(0.85)  # the model's score, unchanged
    assert result["reliability"]["class"] == "Effusion"
    assert result["reliability"]["validated_precision"] == 0.47
    assert result["reliability"]["validated_recall"] == 0.63
    assert result["reliability"]["above_validated_threshold"] is True
    assert "47% of the Effusion flags" in result["explanation"]

    detected = result["detected_predictions"][0]
    assert detected["validated_threshold"] == 0.76
    assert detected["above_validated_threshold"] is True


def test_low_precision_flag_is_not_an_emergency_by_itself():
    """Edema is an emergency if present, but 17% of its flags are real."""
    assert VALIDATION[0]["precision"] < MIN_PRECISION_FOR_CRITICAL
    result = postprocess(Edema=0.95)

    assert (result["severity"], result["urgency"]) == ("high", "urgent")
    assert result["reliability"]["confirmed_by_validation"] is False


def test_validated_flag_keeps_the_risk_of_its_class():
    result = postprocess(Pneumothorax=0.90)

    assert (result["severity"], result["urgency"]) == ("critical", "emergency")
    assert result["reliability"]["confirmed_by_validation"] is True


def test_flag_only_above_the_relaxed_threshold_says_so_and_is_capped():
    # Between the relaxed threshold (0.72) and the validated one (0.82)
    result = postprocess(Pneumothorax=0.78)

    assert result["finding_detected"] is True
    assert result["reliability"]["above_validated_threshold"] is False
    assert (result["severity"], result["urgency"]) == ("high", "urgent")
    assert "below the threshold the model was validated at (0.82)" in result["explanation"]


def test_cap_never_raises_a_milder_finding():
    result = postprocess(Nodule=0.70)  # relaxed-only flag of a moderate / routine class
    assert (result["severity"], result["urgency"]) == ("moderate", "routine")


def test_most_severe_confirmed_finding_decides():
    result = postprocess(Edema=0.95, Pneumothorax=0.90)
    assert (result["severity"], result["urgency"]) == ("critical", "emergency")


def test_negative_screen_states_how_much_the_model_misses():
    result = postprocess()

    assert result["negative_screen"] is True
    assert result["reliability"] is None
    assert "finds between 33% and 63% of the cases" in result["explanation"]


def test_without_validation_figures_nothing_is_claimed_or_capped():
    result = postprocess(validation=None, Edema=0.95)

    assert result["reliability"] is None
    assert "validated_threshold" not in result["detected_predictions"][0]
    assert (result["severity"], result["urgency"]) == ("critical", "emergency")


def test_fused_explanation_calls_the_cv_number_a_model_score_and_adds_the_note():
    cv = postprocess(Effusion=0.85)
    cv["uncertainty"] = {}
    nlp = {
        "primary_diagnosis": "Pleural effusion", "confidence": 0.8, "severity": "high",
        "urgency": "urgent", "explanation": "", "supporting_evidence": ["dyspnea"],
    }
    fused = FusionEngine().fuse(nlp_diagnosis=nlp, cv_diagnosis=cv)

    assert fused["fusion_strategy"] == "concordant"
    assert "(model score: 85%)" in fused["explanation"]
    assert "47% of the Effusion flags" in fused["explanation"]


def test_unconfirmed_image_flag_does_not_escalate_a_multimodal_case():
    """A mild text picture plus an Edema flag that is right 17% of the time
    must not come out as critical / emergency."""
    cv = postprocess(Edema=0.95)
    cv["uncertainty"] = {}
    nlp = {
        "primary_diagnosis": "Common cold", "confidence": 0.8, "severity": "low",
        "urgency": "routine", "explanation": "", "supporting_evidence": ["runny nose"],
    }
    fused = FusionEngine().fuse(nlp_diagnosis=nlp, cv_diagnosis=cv)

    assert fused["severity"] != "critical"
    assert fused["urgency"] != "emergency"
