# %% [markdown]
# # DenseNet-121 on NIH ChestX-ray14: training and a held-out test
#
# One run produces everything the app and the README need:
#
# * `best_densenet121_chestxray14.pth`: the weights (the state_dict the app loads)
# * `best_densenet121_chestxray14_meta.json`: labels, thresholds and the
#   validation and **test** figures the app reports with each flag
# * `splits.csv`, `history.csv`, `test_predictions.npz`: what the numbers
#   were computed from
#
# How the data is used:
#
# * **Test**: the official `test_list.txt` (patient-disjoint from
#   `train_val_list.txt`). It is touched once, after training, with the
#   checkpoint and thresholds already fixed.
# * **Validation**: 10% of the patients of `train_val_list.txt`, grouped by
#   `Patient ID`, so no patient is in two splits. It picks the epoch, drives
#   the learning-rate schedule and sets the per-class thresholds.
# * **Train**: the remaining patients.
#
# Kaggle setup: add the dataset **NIH Chest X-rays** (`nih-chest-xrays/data`),
# GPU T4 or P100, Internet off is fine, then **Save Version -> Save & Run All**
# so the outputs are kept. Expected time on a T4: about 20 minutes to cache the
# images, then about 4 minutes per epoch.

# %% Settings
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2

SEED = 42
IMG_SIZE = 224              # the app resizes to this (CV_IMAGE_SIZE)
VAL_FRACTION = 0.10         # of train_val patients
BATCH_SIZE = 64
EPOCHS = 20
EARLY_STOP_PATIENCE = 4     # epochs without a better validation AUROC
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
MAX_TRAIN_HOURS = 8.0       # leaves time for the test pass in a 12-hour session
BOOTSTRAP_ROUNDS = 1000

# Order and names must match app/config.py (cv_class_names)
LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

OUT = Path("/kaggle/working")
CACHE = Path("/tmp/cxr14_cache")   # not in the saved output
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")
print("torch", torch.__version__)

# %% Locate the dataset
entry_csv = glob.glob("/kaggle/input/**/Data_Entry_2017.csv", recursive=True)
assert entry_csv, "Data_Entry_2017.csv not found: add the dataset nih-chest-xrays/data"
DATA = Path(entry_csv[0]).parent
print("dataset:", DATA)

df = pd.read_csv(DATA / "Data_Entry_2017.csv")
paths = {Path(p).name: p for p in glob.glob(str(DATA / "images_*" / "images" / "*.png"))}
df["path"] = df["Image Index"].map(paths)
missing = df["path"].isna().sum()
assert missing == 0, f"{missing} images listed in the CSV were not found"

findings = df["Finding Labels"].str.split("|")
unknown = set(f for fs in findings for f in fs) - set(LABELS) - {"No Finding"}
assert not unknown, f"unexpected labels: {unknown}"
Y_ALL = np.array([[label in fs for label in LABELS] for fs in findings], dtype=np.float32)
print(f"{len(df):,} images, {df['Patient ID'].nunique():,} patients")

# %% Splits: official test list, patient-grouped validation
test_names = set((DATA / "test_list.txt").read_text().split())
trainval_names = set((DATA / "train_val_list.txt").read_text().split())
assert not test_names & trainval_names

is_test = df["Image Index"].isin(test_names).to_numpy()
trainval_idx = np.flatnonzero(~is_test)
test_idx = np.flatnonzero(is_test)

gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
tr, va = next(gss.split(trainval_idx, groups=df["Patient ID"].to_numpy()[trainval_idx]))
train_idx, val_idx = trainval_idx[tr], trainval_idx[va]

patients = df["Patient ID"].to_numpy()
p_train, p_val, p_test = (set(patients[i]) for i in (train_idx, val_idx, test_idx))
assert not p_train & p_val, "a patient is in train and validation"
assert not (p_train | p_val) & p_test, "a patient is in train/validation and test"

split = np.full(len(df), "", dtype=object)
split[train_idx], split[val_idx], split[test_idx] = "train", "val", "test"
df["split"] = split
df[["Image Index", "Patient ID", "split"]].to_csv(OUT / "splits.csv", index=False)
for name, idx, pats in (("train", train_idx, p_train), ("val", val_idx, p_val), ("test", test_idx, p_test)):
    print(f"{name:<5} {len(idx):>6,} images  {len(pats):>6,} patients  "
          f"positives per class: {Y_ALL[idx].sum(0).astype(int).tolist()}")

# %% Cache every image once at IMG_SIZE (grayscale, uint8)
# Same resize as app/core/cv/image_preprocessor.py: thumbnail with LANCZOS,
# then black padding to a square (NIH images are square, so no padding).
def load_resized(path: str) -> np.ndarray:
    image = Image.open(path).convert("L")
    image.thumbnail((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    canvas.paste(image, ((IMG_SIZE - image.width) // 2, (IMG_SIZE - image.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


cache_file = CACHE / f"images_{IMG_SIZE}.npy"
done_flag = CACHE / "complete"
if not done_flag.exists():
    from concurrent.futures import ProcessPoolExecutor

    started = time.time()
    images = np.lib.format.open_memmap(
        cache_file, mode="w+", dtype=np.uint8, shape=(len(df), IMG_SIZE, IMG_SIZE)
    )
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        for i, array in enumerate(pool.map(load_resized, df["path"], chunksize=256)):
            images[i] = array
            if i % 20000 == 0:
                print(f"  cached {i:,}/{len(df):,} ({time.time() - started:.0f}s)")
    images.flush()
    del images
    done_flag.touch()
    print(f"cache built in {(time.time() - started) / 60:.1f} min")
IMAGES = np.load(cache_file, mmap_mode="r")

# %% Data loading
# No horizontal flip: it would put the heart on the right. Small rotations,
# shifts and zoom stand in for positioning differences between studies.
train_tf = v2.Compose([
    v2.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class CXR(Dataset):
    def __init__(self, indices: np.ndarray, transform):
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        row = self.indices[i]
        gray = torch.from_numpy(np.array(IMAGES[row]))          # H x W uint8
        image = gray.unsqueeze(0).expand(3, -1, -1).contiguous()  # the app feeds RGB
        return self.transform(image), torch.from_numpy(Y_ALL[row])


def loader(indices, transform, shuffle):
    return DataLoader(
        CXR(indices, transform), batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        drop_last=shuffle,
    )


train_loader = loader(train_idx, train_tf, shuffle=True)
val_loader = loader(val_idx, eval_tf, shuffle=False)
test_loader = loader(test_idx, eval_tf, shuffle=False)

# %% Model
# Plain BCE, no pos_weight: heavy positive weights push every score up, and
# the app reports the score next to each flag. Rare classes are handled by
# the per-class thresholds instead.
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
model = model.to(DEVICE).to(memory_format=torch.channels_last)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")


@torch.no_grad()
def predict(data_loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    for images, targets in data_loader:
        images = images.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = model(images)
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def class_aucs(y: np.ndarray, p: np.ndarray) -> list[float]:
    return [float(roc_auc_score(y[:, c], p[:, c])) for c in range(y.shape[1])]


# %% Train
history = []
best_auc, best_epoch, stale = -1.0, -1, 0
started = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_start, running, seen = time.time(), 0.0, 0
    for images, targets in train_loader:
        images = images.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            loss = criterion(model(images), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * len(images)
        seen += len(images)

    val_scores, val_labels = predict(val_loader)
    aucs = class_aucs(val_labels, val_scores)
    val_auc = float(np.mean(aucs))
    scheduler.step(val_auc)
    history.append({
        "epoch": epoch, "train_loss": running / seen, "val_mean_auroc": val_auc,
        "lr": optimizer.param_groups[0]["lr"], "minutes": (time.time() - epoch_start) / 60,
        **{f"val_auroc_{l}": a for l, a in zip(LABELS, aucs)},
    })
    pd.DataFrame(history).to_csv(OUT / "history.csv", index=False)
    print(f"epoch {epoch:>2}  loss {running / seen:.4f}  val AUROC {val_auc:.4f}  "
          f"lr {optimizer.param_groups[0]['lr']:.1e}  {(time.time() - epoch_start) / 60:.1f} min")

    if val_auc > best_auc:
        best_auc, best_epoch, stale = val_auc, epoch, 0
        torch.save(model.state_dict(), OUT / "best_densenet121_chestxray14.pth")
    else:
        stale += 1
        if stale >= EARLY_STOP_PATIENCE:
            print("early stop")
            break
    if (time.time() - started) / 3600 > MAX_TRAIN_HOURS:
        print("time budget reached")
        break

print(f"best epoch {best_epoch}, validation mean AUROC {best_auc:.4f}")

# %% Thresholds from validation (best checkpoint), F1-optimal per class
model.load_state_dict(torch.load(OUT / "best_densenet121_chestxray14.pth", map_location=DEVICE))
val_scores, val_labels = predict(val_loader)


def metrics_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    flagged = p >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# A common, noisily labelled finding gets an F1-optimal threshold that flags
# most films (Infiltration: 60% of films for a 24% prevalence). For these
# classes the threshold is raised until no more films are flagged than carry
# the finding.
CAP_AT_PREVALENCE = {"Infiltration"}

thresholds, val_threshold_metrics = [], []
for c, label in enumerate(LABELS):
    precision, recall, cuts = precision_recall_curve(val_labels[:, c], val_scores[:, c])
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    threshold = float(cuts[int(np.argmax(f1))])
    if label in CAP_AT_PREVALENCE:
        allowed = int(np.floor(val_labels[:, c].mean() * len(val_scores)))
        cap = float(np.nextafter(np.sort(val_scores[:, c])[::-1][allowed], np.float32(1.0)))
        threshold = max(threshold, cap)
    thresholds.append(threshold)
    val_threshold_metrics.append({"label": label, **metrics_at(val_labels[:, c], val_scores[:, c], threshold)})

# %% Test: one pass, checkpoint and thresholds fixed
test_scores, test_labels = predict(test_loader)
np.savez_compressed(OUT / "test_predictions.npz", scores=test_scores, labels=test_labels,
                    image_index=df["Image Index"].to_numpy()[test_idx])

test_aucs = class_aucs(test_labels, test_scores)
test_aps = [float(average_precision_score(test_labels[:, c], test_scores[:, c])) for c in range(len(LABELS))]
test_threshold_metrics = [
    {"label": label, **metrics_at(test_labels[:, c], test_scores[:, c], thresholds[c])}
    for c, label in enumerate(LABELS)
]

# 95% CI by resampling patients (their images move together)
rng = np.random.default_rng(SEED)
test_patients = patients[test_idx]
unique_patients, inverse = np.unique(test_patients, return_inverse=True)
rows_of = [np.flatnonzero(inverse == k) for k in range(len(unique_patients))]
boot_means, boot_class = [], []
for _ in range(BOOTSTRAP_ROUNDS):
    rows = np.concatenate([rows_of[k] for k in rng.integers(0, len(unique_patients), len(unique_patients))])
    y, p = test_labels[rows], test_scores[rows]
    per_class = [roc_auc_score(y[:, c], p[:, c]) if 0 < y[:, c].sum() < len(y) else np.nan
                 for c in range(len(LABELS))]
    boot_class.append(per_class)
    boot_means.append(np.nanmean(per_class))
boot_class = np.array(boot_class)
ci = lambda values: [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(
        abs(y[which == b].mean() - p[which == b].mean()) * (which == b).mean()
        for b in range(bins) if (which == b).any()
    ))


calibration = {
    label: {"brier": float(np.mean((test_scores[:, c] - test_labels[:, c]) ** 2)),
            "ece": expected_calibration_error(test_labels[:, c], test_scores[:, c])}
    for c, label in enumerate(LABELS)
}

print(f"\nTEST mean AUROC {np.mean(test_aucs):.4f}  95% CI {ci(boot_means)}")
report = pd.DataFrame({
    "label": LABELS, "test_auroc": test_aucs,
    "ci_low": [ci(boot_class[:, c])[0] for c in range(len(LABELS))],
    "ci_high": [ci(boot_class[:, c])[1] for c in range(len(LABELS))],
    "test_ap": test_aps,
    "threshold": thresholds,
    "test_precision": [m["precision"] for m in test_threshold_metrics],
    "test_recall": [m["recall"] for m in test_threshold_metrics],
    "test_ece": [calibration[l]["ece"] for l in LABELS],
})
report.to_csv(OUT / "test_report.csv", index=False)
print(report.round(3).to_string(index=False))

# %% Metadata for the app
# threshold_metrics holds the TEST precision and recall at the thresholds
# chosen on validation: that is what the app shows next to a flag.
meta = {
    "model": "densenet121",
    "num_classes": len(LABELS),
    "labels": LABELS,
    "img_size": IMG_SIZE,
    "mean": IMAGENET_MEAN,
    "std": IMAGENET_STD,
    "loss": "BCEWithLogitsLoss (no pos_weight)",
    "epoch": best_epoch,
    "best_auc": best_auc,                       # validation, kept for the app's log line
    "thresholds": thresholds,                   # chosen on validation
    "threshold_metrics": test_threshold_metrics,
    "threshold_metrics_split": "test",
    "val_class_aucs": dict(zip(LABELS, class_aucs(val_labels, val_scores))),
    "val_threshold_metrics": val_threshold_metrics,
    "class_aucs": dict(zip(LABELS, test_aucs)),  # test
    "test_mean_auroc": float(np.mean(test_aucs)),
    "test_mean_auroc_ci95": ci(boot_means),
    "test_class_auroc_ci95": {l: ci(boot_class[:, c]) for c, l in enumerate(LABELS)},
    "test_average_precision": dict(zip(LABELS, test_aps)),
    "test_calibration": calibration,
    "split": {
        "method": "official test_list.txt; validation = 10% of train_val patients (GroupShuffleSplit by Patient ID)",
        "seed": SEED,
        "images": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "patients": {"train": len(p_train), "val": len(p_val), "test": len(p_test)},
        "patient_overlap": 0,
    },
    "training": {
        "pretrained": "ImageNet (torchvision IMAGENET1K_V1)",
        "optimizer": f"AdamW lr={LR} weight_decay={WEIGHT_DECAY}",
        "scheduler": "ReduceLROnPlateau on validation mean AUROC (factor 0.5, patience 1)",
        "batch_size": BATCH_SIZE,
        "epochs_run": len(history),
        "augmentation": "RandomAffine(degrees=7, translate=0.05, scale=0.95-1.05); no flip",
        "resize": "thumbnail LANCZOS to 224 + black padding (as in the app)",
        "amp": DEVICE.type == "cuda",
    },
}
(OUT / "best_densenet121_chestxray14_meta.json").write_text(json.dumps(meta, indent=2))
print("\nwritten:", sorted(p.name for p in OUT.iterdir()))
