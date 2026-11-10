"""DenseNet + CLEAR image ensemble: score combination, figures, fallback."""

import json

import numpy as np
import pytest
from PIL import Image

from app.core.cv import cv_model as cv_module
from app.core.cv.cv_model import MedicalCVModel
from app.core.cv.image_ensemble import (
    ClassCalibration,
    ImageEnsemble,
    combined_probability,
    densenet_logit,
)

CAL = ClassCalibration(
    prompt="pleural effusion", densenet_mean=-4.0, densenet_std=2.0,
    clear_mean=0.01, clear_std=0.02, slope=3.0, intercept=-2.0, threshold=0.6,
    precision=0.5, recall=0.8,
)


def test_combination_rises_with_either_score():
    base = combined_probability(0.05, 0.01, CAL)
    assert combined_probability(0.5, 0.01, CAL) > base
    assert combined_probability(0.05, 0.05, CAL) > base
    assert 0.0 < base < 1.0


def test_extreme_probabilities_stay_finite():
    assert np.isfinite(densenet_logit(0.0)) and np.isfinite(densenet_logit(1.0))
    assert 0.0 <= combined_probability(1.0, 10.0, CAL) <= 1.0


def test_only_calibrated_classes_change():
    ensemble = ImageEnsemble({"Effusion": CAL}, "Open-I")
    names = ["Atelectasis", "Effusion"]
    out = ensemble.combine(np.array([0.3, 0.2]), names, {"Effusion": 0.04})
    assert out[0] == pytest.approx(0.3)
    assert out[1] == pytest.approx(combined_probability(0.2, 0.04, CAL))

    assert ensemble.thresholds(names, [0.5, 0.7]) == [0.5, 0.6]
    metrics = ensemble.validation_metrics(names, [0.5, 0.7], None)
    assert metrics[1] == {"threshold": 0.6, "precision": 0.5, "recall": 0.8, "measured_on": "Open-I"}
    assert metrics[0]["precision"] is None


def test_calibration_must_name_classes_the_model_has(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"measured_on": "x", "classes": {"Effusion": CAL.__dict__}}))
    ensemble = ImageEnsemble.from_file(path)
    assert ensemble.prompts == {"Effusion": "pleural effusion"}
    with pytest.raises(ValueError):
        ensemble.check_classes(["Atelectasis"])


class FakeScorer:
    def __init__(self, fail=False):
        self.fail = fail

    def score(self, image):
        if self.fail:
            raise RuntimeError("CLEAR down")
        return {"Effusion": 0.08}  # far above the calibration mean: a strong CLEAR vote


@pytest.fixture
def model(monkeypatch):
    cv = MedicalCVModel()
    k = list(cv.class_names).index("Effusion")
    probs = np.full(len(cv.class_names), 0.01, dtype=np.float32)
    probs[k] = 0.10  # below DenseNet's own threshold
    monkeypatch.setattr(cv, "densenet_probabilities", lambda tensor: (probs.copy(), None))
    cv._grad_cam_visualizer = type("V", (), {"generate_and_save": lambda *a, **k: "cam.png"})()
    cv.ensemble = ImageEnsemble({"Effusion": CAL}, "Open-I")
    return cv


def test_clear_vote_can_raise_a_flag(model):
    model.clear_scorer = FakeScorer()
    result = model.predict(Image.new("L", (64, 64)))
    assert result["image_models"] == ["DenseNet-121", "CLEAR"]
    assert result["combined_classes"] == ["Effusion"]
    assert "Effusion" in result["predicted_class"]
    assert result["class_thresholds"]["Effusion"] == pytest.approx(0.6)
    assert result["reliability"]["measured_on"] == "Open-I"


def test_clear_failure_falls_back_to_densenet_figures(model):
    model.clear_scorer = FakeScorer(fail=True)
    result = model.predict(Image.new("L", (64, 64)))
    assert result["image_models"] == ["DenseNet-121"]
    assert result["combined_classes"] == []
    k = list(model.class_names).index("Effusion")
    assert result["class_thresholds"]["Effusion"] == pytest.approx(model.densenet_thresholds[k])
    assert "Effusion" not in result["predicted_class"]


def test_disabled_by_default():
    assert cv_module.settings.cv_clear_enabled is False
    assert MedicalCVModel().ensemble is None
