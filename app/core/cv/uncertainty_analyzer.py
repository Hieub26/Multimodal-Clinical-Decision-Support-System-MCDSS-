"""
Prediction Uncertainty Analyzer for CV model outputs.
Combines margin-to-threshold and entropy analysis for robust clinical uncertainty assessment.
"""

import numpy as np
from app.utils.logger import cv_logger


class UncertaintyAnalyzer:
    """Computes prediction uncertainty via margin-to-threshold + entropy analysis.

    Combines two complementary uncertainty signals:
    - Margin-to-threshold: clinically meaningful per-class decision proximity
    - Entropy: information-theoretic measure of prediction diffuseness
    """

    def compute(
        self,
        probs: np.ndarray,
        class_names: list[str],
        class_thresholds: list[float],
    ) -> dict:
        """Compute prediction uncertainty metrics.

        Args:
            probs: Array of per-class probabilities from sigmoid output.
            class_names: List of class label names.
            class_thresholds: Per-class decision thresholds.

        Returns:
            Dictionary with uncertainty metrics and classification.
        """
        margins = []
        near_threshold_classes = []

        for i, prob in enumerate(probs):
            threshold = class_thresholds[i]
            margin = float(prob) - float(threshold)  # positive = above threshold
            margins.append({
                "class": class_names[i],
                "probability": round(float(prob), 4),
                "threshold": round(float(threshold), 4),
                "margin": round(margin, 4),
            })
            # Near-threshold = within ±0.10 of decision boundary
            if abs(margin) < 0.10:
                near_threshold_classes.append(class_names[i])

        # Min absolute margin = most uncertain prediction
        abs_margins = [abs(m["margin"]) for m in margins]
        min_margin = float(min(abs_margins)) if abs_margins else 1.0

        # Entropy computation (per-class binary entropy, then aggregate)
        # For multi-label sigmoid outputs, each class is an independent
        # Bernoulli, so we compute binary entropy per class.
        eps = 1e-7
        binary_entropies = []
        for p in probs:
            p_clamped = float(np.clip(p, eps, 1 - eps))
            h = -(p_clamped * np.log2(p_clamped)
                  + (1 - p_clamped) * np.log2(1 - p_clamped))
            binary_entropies.append(float(h))

        entropy_mean = float(np.mean(binary_entropies)) if binary_entropies else 0.0
        entropy_max = float(np.max(binary_entropies)) if binary_entropies else 0.0

        # Granular entropy: separate positive (above-threshold) classes
        # from overall entropy for clinical usefulness.
        positive_entropies = [
            binary_entropies[i] for i, prob in enumerate(probs)
            if float(prob) >= float(class_thresholds[i])
        ]
        entropy_positive = (
            float(np.mean(positive_entropies)) if positive_entropies else 0.0
        )

        # Uncertainty level: combine margin + entropy for robust assessment
        uncertainty_level = self._classify_uncertainty(
            min_margin, near_threshold_classes, entropy_mean, entropy_max
        )

        uncertainty = {
            "min_margin": round(min_margin, 4),
            "near_threshold_classes": near_threshold_classes,
            "near_threshold_count": len(near_threshold_classes),
            "entropy_mean": round(entropy_mean, 4),
            "entropy_max": round(entropy_max, 4),
            "entropy_positive": round(entropy_positive, 4),
            "uncertainty_level": uncertainty_level,
            "class_margins": margins,
        }

        cv_logger.info(
            f"Uncertainty analysis: level={uncertainty_level}, "
            f"min_margin={min_margin:.4f}, "
            f"entropy_mean={entropy_mean:.4f}, entropy_max={entropy_max:.4f}, "
            f"entropy_positive={entropy_positive:.4f}, "
            f"near_threshold={near_threshold_classes}"
        )
        return uncertainty

    @staticmethod
    def _classify_uncertainty(
        min_margin: float,
        near_threshold_classes: list,
        entropy_mean: float,
        entropy_max: float,
    ) -> str:
        """Classify uncertainty level using both margin and entropy signals."""
        # High: very close to decision boundary OR high entropy
        if (min_margin < 0.05 or len(near_threshold_classes) >= 3
                or entropy_max > 0.90 or entropy_mean > 0.60):
            return "high"
        # Moderate: somewhat close to boundary OR moderate entropy
        if (min_margin < 0.10 or len(near_threshold_classes) >= 1
                or entropy_max > 0.70 or entropy_mean > 0.40):
            return "moderate"
        return "low"
