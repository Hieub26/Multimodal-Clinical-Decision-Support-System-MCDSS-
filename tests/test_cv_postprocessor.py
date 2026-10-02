import numpy as np

from app.core.cv.cv_postprocessor import CVPostprocessor
from app.core.cv.image_preprocessor import ImagePreprocessor
from app.core.fusion.fusion_engine import FusionEngine
from app.core.safety.safety_controller import SafetyController
from PIL import Image

CLASS_NAMES = ["Emphysema", "Infiltration", "Mass"]
THRESHOLDS = [0.80, 0.45, 0.70]


def test_decision_follows_detected_class_not_argmax():
    """Emphysema has the highest probability but is below its own threshold;
    the reported finding is Infiltration."""
    probs = np.array([0.75, 0.50, 0.05])
    result = CVPostprocessor().postprocess(probs, CLASS_NAMES, THRESHOLDS)

    assert result["predicted_class"] == "Infiltration"
    assert result["primary_class_index"] == CLASS_NAMES.index("Infiltration")
    assert result["decision_threshold"] == 0.45
    assert result["confidence"] >= result["decision_threshold"]


def test_detected_finding_is_not_flagged_low_confidence():
    probs = np.array([0.75, 0.50, 0.05])
    cv = CVPostprocessor().postprocess(probs, CLASS_NAMES, THRESHOLDS)
    cv["uncertainty"] = {}

    fused = FusionEngine().fuse(cv_diagnosis=cv)
    result = SafetyController().evaluate({**fused, "is_guideline_consistent": True})

    assert not any(r.startswith("Low confidence") for r in result["risk_factors"])


def test_no_finding_uses_argmax_class():
    probs = np.array([0.75, 0.30, 0.05])
    result = CVPostprocessor().postprocess(probs, CLASS_NAMES, THRESHOLDS)

    assert result["finding_detected"] is False
    assert result["primary_class_index"] == 0
    assert result["positive_decision_threshold"] == 0.80


def test_content_box_marks_unpadded_region():
    """A 400x200 image is letterboxed into 224x224: content is the middle band."""
    preprocessed = ImagePreprocessor(image_size=224).preprocess(Image.new("RGB", (400, 200)))

    assert preprocessed.content_box == (0, 56, 224, 168)
    assert tuple(preprocessed.tensor.shape) == (1, 3, 224, 224)
