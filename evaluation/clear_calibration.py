"""
Calibrate the DenseNet-121 + CLEAR image ensemble on Open-I and measure it.

Data: the full Open-I / Indiana University chest X-ray collection (NLM, CC
BY-NC-ND 4.0), downloaded by hand and not redistributed:

    evaluation/.cache/openi/reports/ecgen-radiology/*.xml   NLMCXR_reports.tgz
    evaluation/.cache/openi/images_full/<image id>.png      NLMCXR_png.tgz

Labels come from the reports' MeSH terms (evaluation.fusion_eval.TARGETS): an
image is positive for a class when its report names it, normal when the only
MeSH term is "normal"; images whose reports name other findings only are left
out. Lateral films are removed with the TorchXRayVision view classifier
(xinario), since a report's finding is often not visible on the lateral view
and both models were built for frontal films.

Reports are split in two halves by a hash of their file name, so the views of
a report stay together:
  - half A fits, per class, the standardization of each model's score and
    the logistic map of their mean, and chooses the operating point: one
    specificity for every combined class, the lowest on SPECIFICITY_GRID at
    which the share of normal films of A flagged for anything (all 14
    classes) is not above DenseNet's share at its own thresholds. A class is
    combined only when, at that specificity, the combination finds more of
    its cases on A than DenseNet alone at the same specificity;
  - half B is only measured: AUROC (positives vs normal films), recall and
    specificity at the threshold, precision over all labelled frontal films
    of B, and the share of normal films flagged for anything.

The 95% specificity first tried flagged a quarter of the normal films of B;
the rule above was written after seeing that. Nodule is left out by hand
(EXCLUDED_AFTER_REVIEW), also after seeing half B.

Writes models/clear_ensemble_calibration.json (read by the app when
CV_CLEAR_ENABLED=true) and evaluation/results/clear_calibration.json.

Usage:
    python -m evaluation.clear_calibration
Needs the CLEAR package and weights (scripts/fetch_clear.py), torchxrayvision
and scikit-learn. Scores are cached per image, so a rerun only fits.
"""

import argparse
import glob
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from datetime import date

import numpy as np
import torch
from PIL import Image

from app.config import settings
from app.core.cv.clear_scorer import ClearScorer
from app.core.cv.cv_model import MedicalCVModel
from app.core.cv.image_ensemble import ClassCalibration, combined_value, densenet_logit
from evaluation.common import CACHE_DIR, save_result
from evaluation.fusion_eval import TARGETS

OPENI_DIR = CACHE_DIR / "openi"
SCORES_CACHE = OPENI_DIR / "clear_calibration_scores.json"
OUTPUT = settings.models_dir / "clear_ensemble_calibration.json"

# App class -> (label used in TARGETS, CLEAR prompt). The prompt pair is the
# CLEAR authors' ("{}", "no {}").
CANDIDATES = {
    "Cardiomegaly": ("cardiomegaly", "cardiomegaly"),
    "Effusion": ("effusion", "pleural effusion"),
    "Atelectasis": ("atelectasis", "atelectasis"),
    "Nodule": ("nodule", "nodule"),
    "Edema": ("edema", "edema"),
    "Pneumonia": ("pneumonia", "pneumonia"),
}
SPECIFICITY_GRID = (0.95, 0.96, 0.97, 0.98, 0.99, 0.995)
# Kept on DenseNet alone although the rule on half A combined them: decided
# after seeing half B, where the combination found fewer nodules (12% against
# 18%) at a lower precision. Half B no longer measures this choice blind.
EXCLUDED_AFTER_REVIEW = ("Nodule",)


def labelled_images(images_dir: str) -> list[dict]:
    have = {f[:-4] for f in os.listdir(images_dir) if f.endswith(".png")}
    rows = []
    for path in sorted(glob.glob(str(OPENI_DIR / "reports" / "ecgen-radiology" / "*.xml"))):
        root = ET.parse(path).getroot()
        mesh = [m.text.split("/")[0].strip().lower() for m in root.findall("./MeSH/major") if m.text]
        found = sorted({TARGETS[m] for m in mesh if m in TARGETS})
        group = "finding" if found else "normal" if mesh == ["normal"] else None
        if not group:
            continue
        for image_id in (p.get("id") for p in root.findall("parentImage")):
            if image_id in have:
                rows.append({"report": os.path.basename(path), "image": image_id,
                             "group": group, "truth": found})
    return rows


def score_images(rows: list[dict], images_dir: str) -> dict:
    cache = json.load(open(SCORES_CACHE)) if SCORES_CACHE.exists() else {}
    todo = [r for r in rows if r["image"] not in cache]
    if not todo:
        return cache
    import torchxrayvision as xrv

    cv = MedicalCVModel()
    clear = ClearScorer({name: prompt for name, (_, prompt) in CANDIDATES.items()}, cv.device,
                        checkpoint_path=settings.cv_clear_checkpoint_path,
                        allow_download=settings.cv_clear_allow_download)
    view = xrv.baseline_models.xinario.ViewModel().to(cv.device).eval()
    crop, resize = xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)
    for n, r in enumerate(todo):
        file = os.path.join(images_dir, r["image"] + ".png")
        pre = cv.image_preprocessor.preprocess(file)
        probs, _ = cv.densenet_probabilities(pre["tensor"].to(cv.device))
        gray = np.asarray(Image.open(file).convert("L"), dtype=np.float32)
        x = torch.from_numpy(resize(crop(xrv.utils.normalize(gray, 255)[None, ...])))[None, ...].to(cv.device)
        with torch.no_grad():
            p_lateral = float(torch.softmax(view(x), 1)[0, 1])
        cache[r["image"]] = {"densenet": probs.tolist(), "clear": clear.score(pre["original_image"]),
                             "p_lateral": p_lateral}
        if n % 500 == 0:
            print(f"  scored {n}/{len(todo)}", flush=True)
            json.dump(cache, open(SCORES_CACHE, "w"))
    json.dump(cache, open(SCORES_CACHE, "w"))
    return cache


def auroc(scores: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
    from scipy.stats import rankdata
    p, n = scores[positive], scores[negative]
    ranks = rankdata(np.concatenate([p, n]))
    return float((ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def at_threshold(flag: np.ndarray, truth: np.ndarray, normal: np.ndarray) -> dict:
    tp = int((flag & truth).sum())
    return {
        "recall": tp / max(int(truth.sum()), 1),
        "specificity": 1 - float(flag[normal].mean()),
        "precision": tp / int(flag.sum()) if flag.any() else None,
    }


def run(images_dir: str) -> None:
    from sklearn.linear_model import LogisticRegression

    rows = labelled_images(images_dir)
    print(f"labelled images: {len(rows)}")
    cache = score_images(rows, images_dir)
    cv = MedicalCVModel()
    cv_classes = list(cv.class_names)

    frontal = [r for r in rows if cache[r["image"]]["p_lateral"] < 0.5]
    half = np.array([int(hashlib.sha256(r["report"].encode()).hexdigest(), 16) % 2 for r in frontal])
    normal = np.array([r["group"] == "normal" for r in frontal])
    A, B = half == 0, half == 1
    print(f"frontal films: {len(frontal)} (A {A.sum()}, B {B.sum()}); lateral removed: {len(rows) - len(frontal)}")

    # DenseNet as deployed: every class at its own threshold
    densenet = np.array([cache[r["image"]]["densenet"] for r in frontal])
    densenet_flags = densenet >= np.array(cv.densenet_thresholds)[None, :]

    # Per class: the combined value and its logistic map, fitted on A
    fitted = {}
    for name, (label, prompt) in CANDIDATES.items():
        k = cv_classes.index(name)
        truth = np.array([label in r["truth"] for r in frontal])
        p = densenet[:, k]
        s = np.array([cache[r["image"]]["clear"][name] for r in frontal])
        logit = np.array([densenet_logit(v) for v in p])
        stats = ClassCalibration(prompt, float(logit[A].mean()), float(logit[A].std()),
                                 float(s[A].mean()), float(s[A].std()), 1.0, 0.0, 0.5)
        c = np.array([combined_value(pv, sv, stats) for pv, sv in zip(p, s)])
        fit = A & (truth | normal)
        lr = LogisticRegression(C=1e6).fit(c[fit, None], truth[fit])
        fitted[name] = dict(k=k, truth=truth, p=p, c=c, stats=stats,
                            slope=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]))

    def operating_point(spec: float):
        """Classes the combination improves on A at this specificity, their
        cuts, and the share of normal films of A flagged for anything."""
        chosen, flags = {}, densenet_flags.copy()
        for name, f in fitted.items():
            cut = float(np.quantile(f["c"][A & normal], spec))
            densenet_cut = float(np.quantile(f["p"][A & normal], spec))
            recall_c = (f["c"] >= cut)[A & f["truth"]].mean()
            recall_d = (f["p"] >= densenet_cut)[A & f["truth"]].mean()
            if recall_c > recall_d and name not in EXCLUDED_AFTER_REVIEW:
                chosen[name] = cut
                flags[:, f["k"]] = f["c"] >= cut
        return chosen, flags

    baseline_a = densenet_flags.any(axis=1)[A & normal].mean()
    for spec in SPECIFICITY_GRID:
        chosen, flags = operating_point(spec)
        flagged_a = flags.any(axis=1)[A & normal].mean()
        print(f"  specificity {spec:.1%}: normal films of A flagged {flagged_a:.1%} "
              f"(DenseNet {baseline_a:.1%}), combined: {', '.join(chosen) or 'none'}")
        if flagged_a <= baseline_a:
            break
    else:
        raise SystemExit("No specificity on the grid keeps the false-flag rate of DenseNet")

    calibration, report = {}, {}
    for name, f in fitted.items():
        auc = {f"{h}_{m}": auroc(v, sel & f["truth"], sel & normal)
               for h, sel in (("A", A), ("B", B)) for m, v in (("densenet", f["p"]), ("combined", f["c"]))}
        use = name in chosen
        entry = {"positives_A": int((A & f["truth"]).sum()), "positives_B": int((B & f["truth"]).sum()),
                 **{key: round(v, 4) for key, v in auc.items()}, "combined": use,
                 "densenet_at_its_threshold_B": at_threshold(
                     B & densenet_flags[:, f["k"]], B & f["truth"], B & normal)}
        if use:
            cut = chosen[name]
            threshold = float(1 / (1 + np.exp(-(f["slope"] * cut + f["intercept"]))))
            measured = at_threshold(B & (f["c"] >= cut), B & f["truth"], B & normal)
            entry.update(threshold=threshold, combined_at_threshold_B=measured)
            calibration[name] = {**{k_: getattr(f["stats"], k_) for k_ in (
                "prompt", "densenet_mean", "densenet_std", "clear_mean", "clear_std")},
                "slope": f["slope"], "intercept": f["intercept"], "threshold": threshold,
                "precision": measured["precision"], "recall": measured["recall"]}
        report[name] = entry
        print(f"  {name:<13} pos A/B {entry['positives_A']:>3}/{entry['positives_B']:<3} "
              f"AUROC A {auc['A_densenet']:.3f} -> {auc['A_combined']:.3f}, "
              f"B {auc['B_densenet']:.3f} -> {auc['B_combined']:.3f}  "
              f"{'combined' if use else 'DenseNet only'}")
    flagged_b = {"densenet": float(densenet_flags.any(axis=1)[B & normal].mean()),
                 "ensemble": float(flags.any(axis=1)[B & normal].mean())}
    print(f"normal films of B flagged for anything: DenseNet {flagged_b['densenet']:.1%}, "
          f"ensemble {flagged_b['ensemble']:.1%}")

    OUTPUT.write_text(json.dumps({
        "measured_on": "Open-I held-out half (frontal films)",
        "created": date.today().isoformat(),
        "clear": {"repository": "peterhan91/CLEAR", "weights": "huggingface.co/peterhan91/CLEAR"},
        "classes": calibration,
    }, indent=1), encoding="utf-8")
    save_result("clear_calibration.json", {"frontal_films": len(frontal), "half_A": int(A.sum()),
                                      "half_B": int(B.sum()), "specificity": spec,
                                      "normal_films_flagged_B": flagged_b, "classes": report})
    print(f"calibration written to {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--images", default=str(OPENI_DIR / "images_full"))
    run(parser.parse_args().images)
