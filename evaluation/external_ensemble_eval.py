"""
The DenseNet + CLEAR ensemble on collections it was not calibrated on. The
ensemble keeps the thresholds set on Open-I (models/clear_ensemble_calibration.json),
so these runs test whether the combination and its thresholds carry over.

nih    ChestX-ray14 (NIH), official patient-disjoint test split: the DenseNet's
       own test set (none of its patients were used to train or tune it).
       CLEAR was not trained on it. Labels are text-mined from the reports.
       A random sample of test patients is enough:
           evaluation/.cache/nih/Data_Entry_2017_v2020.csv
           evaluation/.cache/nih/images/<Image Index>.png
vindr  VinDr-CXR (two hospitals in Vietnam), the training set of the Kaggle
       VinBigData competition: every film read by three radiologists. Neither
       model was trained on it. A class is present when at least two of the
       three readers marked it; a film is normal when all three found nothing.
       PNGs from the original DICOMs, 512 px on the long side (Kaggle
       xhlulu/vinbigdata-chest-xray-png-512px-original-ratio):
           evaluation/.cache/vindr/train.csv.zip
           evaluation/.cache/vindr/png/train/<image_id>.png

For each class: AUROC (films with the finding against all others) and, at each
system's thresholds, recall and precision; the share of normal films flagged
for anything; and a paired bootstrap (patients, or films for VinDr) of the
mean AUROC difference over the combined classes.

Usage:
    python -m evaluation.external_ensemble_eval nih
    python -m evaluation.external_ensemble_eval vindr
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from app.config import settings
from app.core.cv.clear_scorer import ClearScorer
from app.core.cv.cv_model import MedicalCVModel
from app.core.cv.image_ensemble import ImageEnsemble
from evaluation.common import CACHE_DIR, save_result

REPORTED = ["Cardiomegaly", "Effusion", "Atelectasis", "Edema", "Pneumonia", "Nodule"]


def auroc(scores, positive):
    r = rankdata(scores)
    n1 = positive.sum()
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(scores) - n1)))


def load_nih():
    root = CACHE_DIR / "nih"
    images = sorted(f for f in os.listdir(root / "images") if f.endswith(".png"))
    entry = pd.read_csv(root / "Data_Entry_2017_v2020.csv").set_index("Image Index").loc[images]
    labels = entry["Finding Labels"].str.split("|")
    truth = {c: labels.apply(lambda l, c=c: c in l).to_numpy() for c in REPORTED}
    normal = labels.apply(lambda l: l == ["No Finding"]).to_numpy()
    return [str(root / "images" / i) for i in images], truth, normal, entry["Patient ID"].to_numpy()


VINDR_CLASSES = {"Cardiomegaly": "Cardiomegaly", "Effusion": "Pleural effusion",
                 "Atelectasis": "Atelectasis", "Nodule": "Nodule/Mass"}


def load_vindr():
    root = CACHE_DIR / "vindr"
    readers = (pd.read_csv(root / "train.csv.zip")
               .groupby(["image_id", "class_name"]).rad_id.nunique().unstack(fill_value=0))
    ids = [i for i in readers.index if (root / "png" / "train" / f"{i}.png").exists()]
    readers = readers.loc[ids]
    truth = {c: (readers[v] >= 2).to_numpy() for c, v in VINDR_CLASSES.items()}
    normal = (readers["No finding"] == 3).to_numpy()
    return [str(root / "png" / "train" / f"{i}.png") for i in ids], truth, normal, np.array(ids)


def score(paths, cache_path, ensemble, cv):
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    todo = [p for p in paths if os.path.basename(p) not in cache]
    if todo:
        clear = ClearScorer({**ensemble.prompts, "Nodule": "nodule"}, cv.device,
                            checkpoint_path=settings.cv_clear_checkpoint_path,
                            allow_download=settings.cv_clear_allow_download)
        for n, path in enumerate(todo):
            pre = cv.image_preprocessor.preprocess(path)
            probs, _ = cv.densenet_probabilities(pre["tensor"].to(cv.device))
            cache[os.path.basename(path)] = {"densenet": probs.tolist(), "clear": clear.score(pre["original_image"])}
            if n % 500 == 0:
                print(f"  scored {n}/{len(todo)}", flush=True)
                json.dump(cache, open(cache_path, "w"))
        json.dump(cache, open(cache_path, "w"))
    return cache


def run(collection: str) -> None:
    cv = MedicalCVModel()
    ensemble = ImageEnsemble.from_file(settings.cv_clear_calibration_path)
    names = list(cv.class_names)
    paths, truth, normal, groups_of = {"nih": load_nih, "vindr": load_vindr}[collection]()
    cache = score(paths, CACHE_DIR / collection / "ensemble_scores.json", ensemble, cv)

    keys = [os.path.basename(p) for p in paths]
    densenet = np.array([cache[k]["densenet"] for k in keys])
    combined = np.array([ensemble.combine(densenet[j], names, cache[k]["clear"]) for j, k in enumerate(keys)])
    flags_d = densenet >= np.array(cv.densenet_thresholds)
    flags_e = combined >= np.array(ensemble.thresholds(names, cv.densenet_thresholds))

    print(f"{collection}: {len(paths)} films, {len(set(groups_of))} patients/films, {normal.sum()} normal")
    print(f"  {'class':<13}{'n':>5} | AUROC DenseNet -> ensemble | recall | precision")
    report = {}
    for c in [c for c in REPORTED if c in truth]:
        k, pos = names.index(c), truth[c]

        def at(flags):
            tp = int((flags[:, k] & pos).sum())
            return {"recall": tp / max(int(pos.sum()), 1),
                    "precision": tp / int(flags[:, k].sum()) if flags[:, k].any() else None}

        a_d, a_e = auroc(densenet[:, k], pos), auroc(combined[:, k], pos)
        d, e = at(flags_d), at(flags_e)
        report[c] = {"positives": int(pos.sum()), "combined": c in ensemble.classes, "auroc_densenet": a_d,
                     "auroc_ensemble": a_e, "densenet": d, "ensemble": e}
        f = lambda x: "  - " if x is None else f"{x:4.0%}"  # noqa: E731
        print(f"  {c:<13}{int(pos.sum()):>5} | {a_d:.3f} -> {a_e:.3f}{'' if c in ensemble.classes else ' (not combined)'}"
              f" | {f(d['recall'])} -> {f(e['recall'])} | {f(d['precision'])} -> {f(e['precision'])}")
    flagged = {"densenet": float(flags_d.any(axis=1)[normal].mean()),
               "ensemble": float(flags_e.any(axis=1)[normal].mean())}
    print(f"normal films flagged for anything: DenseNet {flagged['densenet']:.1%}, ensemble {flagged['ensemble']:.1%}")

    main = [c for c in report if c in ensemble.classes]
    groups = [np.where(groups_of == g)[0] for g in np.unique(groups_of)]
    rng, diffs = np.random.default_rng(0), []
    for _ in range(2000):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        vals = [auroc(combined[idx, names.index(c)], truth[c][idx]) - auroc(densenet[idx, names.index(c)], truth[c][idx])
                for c in main if 0 < truth[c][idx].sum() < len(idx)]
        diffs.append(np.mean(vals))
    point = float(np.mean([report[c]["auroc_ensemble"] - report[c]["auroc_densenet"] for c in main]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    print(f"mean AUROC difference over {', '.join(main)}: {point:+.3f} [{lo:+.3f}, {hi:+.3f}] (95% CI)")

    save_result(f"{collection}_ensemble.json", {
        "films": len(paths), "normal": int(normal.sum()), "classes": report, "normal_flagged": flagged,
        "mean_auroc_difference": {"classes": main, "point": point, "ci95": [float(lo), float(hi)]}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("collection", choices=["nih", "vindr"])
    run(parser.parse_args().collection)
