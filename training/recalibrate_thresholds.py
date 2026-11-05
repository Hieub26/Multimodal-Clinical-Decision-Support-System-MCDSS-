"""Re-pick the threshold of classes whose F1-optimal threshold flags too many films.

    python training/recalibrate_thresholds.py <test_predictions.npz> <splits.csv> <meta.json>

For a common, noisily labelled finding the F1-optimal threshold buys recall
by flagging most films: Infiltration (24% of test films) was flagged on 60%.
For each class in CAP_AT_PREVALENCE the threshold becomes the lowest one at
which no more films are flagged than carry the finding.

The training run kept no validation scores, so the threshold is chosen on one
half of the test patients (A) and its precision and recall are reported on
the other half (B); the two halves share no patient. AUROC does not depend on
the threshold and is unchanged. The metadata file is updated in place.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CAP_AT_PREVALENCE = ["Infiltration"]
SEED = 0

predictions, splits, meta_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
data = np.load(predictions, allow_pickle=True)
scores, labels, names = data["scores"], data["labels"], data["image_index"]
patients = pd.read_csv(splits).set_index("Image Index").loc[names, "Patient ID"].to_numpy()

unique = np.unique(patients)
np.random.default_rng(SEED).shuffle(unique)
half_a = np.isin(patients, unique[: len(unique) // 2])
half_b = ~half_a

meta = json.loads(meta_path.read_text(encoding="utf-8"))
overrides = meta.setdefault("threshold_overrides", {})


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    flagged = p >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn, "flag_rate": float(flagged.mean())}


for label in CAP_AT_PREVALENCE:
    c = meta["labels"].index(label)
    y_a, p_a = labels[half_a, c], scores[half_a, c]
    prevalence = float(y_a.mean())
    # Lowest threshold whose flag rate on half A is at most the prevalence:
    # just above the (k+1)-th highest score, k = films that carry the finding
    allowed = int(np.floor(prevalence * len(p_a)))
    highest_first = np.sort(p_a)[::-1]
    threshold = float(np.nextafter(highest_first[allowed], np.float32(1.0)))
    assert (p_a >= threshold).sum() <= allowed

    before = meta["thresholds"][c]
    meta["thresholds"][c] = threshold
    measured = metrics(labels[half_b, c], scores[half_b, c], threshold)
    entry = next(m for m in meta["threshold_metrics"] if m["label"] == label)
    entry.clear()
    entry.update({"label": label, **measured})
    overrides[label] = {
        "rule": "lowest threshold flagging no more films than carry the finding",
        "previous_threshold": before,
        "chosen_on": "test patients, half A",
        "measured_on": "test patients, half B (no patient shared with A)",
        "prevalence_half_a": prevalence,
        "images": {"half_a": int(half_a.sum()), "half_b": int(half_b.sum())},
        "seed": SEED,
    }
    print(f"{label}: threshold {before:.3f} -> {threshold:.3f}; on half B "
          f"precision {measured['precision']:.2f}, recall {measured['recall']:.2f}, "
          f"films flagged {measured['flag_rate']:.0%} (prevalence {labels[half_b, c].mean():.0%})")

meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
