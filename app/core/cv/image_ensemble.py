"""
Image ensemble: DenseNet-121 probabilities combined with CLEAR zero-shot scores.

For each class listed in the calibration file (models/clear_ensemble_calibration.json,
written by evaluation/clear_calibration.py from Open-I), the two scores are put
on a common scale with statistics of the calibration images, averaged, and
mapped to a 0-1 score with a logistic fit. The threshold and the precision and
recall that come with each class were measured on Open-I, not ChestX-ray14.
Classes not listed keep the DenseNet probability and its own figures.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EPS = 1e-6


@dataclass(frozen=True)
class ClassCalibration:
    """Calibration of the combined score for one class."""

    prompt: str
    densenet_mean: float
    densenet_std: float
    clear_mean: float
    clear_std: float
    slope: float
    intercept: float
    threshold: float
    precision: float | None = None
    recall: float | None = None


def densenet_logit(probability: float) -> float:
    """Logit of a DenseNet probability, clipped away from 0 and 1."""
    p = min(max(float(probability), _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def combined_value(densenet_probability: float, clear_score: float, cal: ClassCalibration) -> float:
    """Mean of the two standardized scores (before the logistic map)."""
    z_densenet = (densenet_logit(densenet_probability) - cal.densenet_mean) / max(cal.densenet_std, _EPS)
    z_clear = (float(clear_score) - cal.clear_mean) / max(cal.clear_std, _EPS)
    return 0.5 * (z_densenet + z_clear)


def combined_probability(densenet_probability: float, clear_score: float, cal: ClassCalibration) -> float:
    """Combined 0-1 score of one class."""
    x = cal.slope * combined_value(densenet_probability, clear_score, cal) + cal.intercept
    return 1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0)))


class ImageEnsemble:
    """Per-class combination of DenseNet and CLEAR, from a calibration file."""

    def __init__(self, classes: dict[str, ClassCalibration], measured_on: str):
        if not classes:
            raise ValueError("The calibration lists no class to combine")
        self.classes = classes
        self.measured_on = measured_on

    @classmethod
    def from_file(cls, path: str | Path) -> "ImageEnsemble":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = {
            name: ClassCalibration(**{k: entry[k] for k in ClassCalibration.__dataclass_fields__ if k in entry})
            for name, entry in data["classes"].items()
        }
        return cls(classes, data.get("measured_on", "Open-I"))

    @property
    def prompts(self) -> dict[str, str]:
        """CLEAR prompt per combined class."""
        return {name: cal.prompt for name, cal in self.classes.items()}

    def check_classes(self, class_names) -> None:
        missing = sorted(set(self.classes) - set(class_names))
        if missing:
            raise ValueError(f"Calibrated classes not produced by the CV model: {missing}")

    def combine(self, probs: np.ndarray, class_names, clear_scores: dict[str, float]) -> np.ndarray:
        """DenseNet probabilities with the combined classes replaced."""
        out = np.array(probs, dtype=np.float64, copy=True)
        for i, name in enumerate(class_names):
            cal = self.classes.get(name)
            if cal is not None:
                out[i] = combined_probability(out[i], clear_scores[name], cal)
        return out.astype(np.float32)

    def thresholds(self, class_names, base: list[float]) -> list[float]:
        return [
            self.classes[name].threshold if name in self.classes else float(base[i])
            for i, name in enumerate(class_names)
        ]

    def validation_metrics(self, class_names, base: list[float], base_metrics: list[dict] | None) -> list[dict]:
        """Figures for every class: the calibration's for combined classes,
        the DenseNet metadata's (or none) for the others."""
        metrics = []
        for i, name in enumerate(class_names):
            cal = self.classes.get(name)
            if cal is not None:
                metrics.append({
                    "threshold": cal.threshold,
                    "precision": cal.precision,
                    "recall": cal.recall,
                    "measured_on": self.measured_on,
                })
            elif base_metrics:
                metrics.append(base_metrics[i])
            else:
                metrics.append({"threshold": float(base[i]), "precision": None, "recall": None})
        return metrics
