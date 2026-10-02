import numpy as np
import pytest

from app.core.cv.cv_postprocessor import CVPostprocessor
from app.core.fusion.fusion_engine import FusionEngine

CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
]
THRESHOLDS = [0.5] * len(CLASS_NAMES)


def _cv_result(**probabilities) -> dict:
    probs = np.full(len(CLASS_NAMES), 0.02)
    for name, prob in probabilities.items():
        probs[CLASS_NAMES.index(name)] = prob
    result = CVPostprocessor().postprocess(probs, CLASS_NAMES, THRESHOLDS)
    result["uncertainty"] = {}
    return result


def _nlp_result(diagnosis: str, severity: str = "high", urgency: str = "urgent") -> dict:
    return {
        "primary_diagnosis": diagnosis,
        "confidence": 0.85,
        "severity": severity,
        "urgency": urgency,
        "supporting_evidence": ["evidence"],
        "explanation": "",
        "recommended_actions": [],
    }


@pytest.fixture(scope="module")
def fusion():
    return FusionEngine()


@pytest.mark.parametrize("nlp_severity, nlp_urgency", [
    ("critical", "emergency"),
    ("high", "urgent"),
    ("low", "routine"),
])
def test_fusion_never_deescalates(fusion, nlp_severity, nlp_urgency):
    """CV pneumothorax is critical/emergency; adding text must not lower that."""
    cv = _cv_result(Pneumothorax=0.95)
    assert (cv["severity"], cv["urgency"]) == ("critical", "emergency")

    fused = fusion.fuse(
        nlp_diagnosis=_nlp_result("Pneumothorax", nlp_severity, nlp_urgency),
        cv_diagnosis=cv,
    )
    assert (fused["severity"], fused["urgency"]) == ("critical", "emergency")


def test_fusion_keeps_nlp_severity_when_image_is_clear(fusion):
    fused = fusion.fuse(
        nlp_diagnosis=_nlp_result("Acute myocardial infarction", "critical", "emergency"),
        cv_diagnosis=_cv_result(),
    )
    assert (fused["severity"], fused["urgency"]) == ("critical", "emergency")


@pytest.mark.parametrize("diagnosis", [
    "Lung adenocarcinoma",              # contains "no"
    "Non-ST elevation myocardial infarction",
    "Infectious mononucleosis",
])
def test_negative_screen_is_never_concordant(fusion, diagnosis):
    cv = _cv_result()
    assert cv["finding_detected"] is False

    fused = fusion.fuse(nlp_diagnosis=_nlp_result(diagnosis), cv_diagnosis=cv)
    assert fused["fusion_strategy"] != "concordant"


@pytest.mark.parametrize("cv_probs, diagnosis, expected", [
    ({"Mass": 0.9}, "Massive pulmonary embolism", "discordant"),
    ({"Mass": 0.9}, "Lung mass", "concordant"),
    ({"Nodule": 0.9}, "Pulmonary nodules", "concordant"),
    ({"Pleural_Thickening": 0.9}, "Pleural thickening", "concordant"),
    ({"Pneumonia": 0.9}, "Lobar consolidation", "concordant"),  # ontology mapping
])
def test_strategy_uses_whole_word_matching(fusion, cv_probs, diagnosis, expected):
    fused = fusion.fuse(
        nlp_diagnosis=_nlp_result(diagnosis), cv_diagnosis=_cv_result(**cv_probs)
    )
    assert fused["fusion_strategy"] == expected
