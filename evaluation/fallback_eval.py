"""
Measure the rule-based fallback diagnosis engine on a public benchmark.

The benchmark is gretelai/symptom_to_diagnosis (Apache-2.0): 1,065 patient-
style symptom descriptions labeled with one of 22 diagnoses. It was not
written for this project and the engine was not built on it. It is downloaded
on first use into evaluation/.cache and is not redistributed here.

Each text goes through the path the application takes without an LLM:
preprocessing, guideline retrieval, ClinicalFallbackEngine. Two questions:

  - How often is the engine's first diagnosis the labeled one?
  - Of the diagnoses whose confidence passes the gate (and would be shown as
    approved), how many are right, and how many right ones does it hold back?

Usage:
    python -m evaluation.fallback_eval --split dev
    python -m evaluation.fallback_eval --split test

Splits: "dev" is the benchmark's train file, used to fit the confidence
formula; "test" is its test file, not run until the formula was final.
No API is called.
"""

import argparse
import json
import urllib.request
from collections import defaultdict

from app.config import settings
from app.core.nlp.clinical_fallback import ClinicalFallbackEngine
from app.core.nlp.rag_engine import RAGEngine
from app.core.nlp.text_preprocessor import TextPreprocessor
from app.core.nlp.vector_store import VectorStore
from evaluation.common import CACHE_DIR, save_result

DATASET_REVISION = "722cfb0e11f8ae37339c7f573b5e10429b94df49"
DATASET_URL = (
    "https://huggingface.co/datasets/gretelai/symptom_to_diagnosis/resolve/"
    f"{DATASET_REVISION}/{{file}}"
)
SPLIT_FILES = {"dev": "train.jsonl", "test": "test.jsonl"}

# How each benchmark label relates to what the engine can output. Written
# before the engine was run on the benchmark.
#
# Labels the engine has an entry for, and the entry that counts as correct.
# The engine has a single entry for skin disease, so that entry is the right
# answer for the three skin conditions.
IN_SCOPE = {
    "pneumonia": "pneumonia",
    "bronchial asthma": "asthma",
    "diabetes": "diabetes",
    "hypertension": "hypertension",
    "gastroesophageal reflux disease": "gerd",
    "urinary tract infection": "uti",
    "psoriasis": "skin_condition",
    "fungal infection": "skin_condition",
    "impetigo": "skin_condition",
}
# Conditions the engine has no entry for: whatever it names is wrong, and a
# confident answer is the failure this benchmark is here to catch.
OUT_OF_SCOPE = {
    "arthritis", "cervical spondylosis", "common cold", "dengue", "jaundice",
    "malaria", "migraine", "peptic ulcer disease", "typhoid", "varicose veins",
}
# Rash-dominated conditions without an entry: "Dermatological Condition"
# would be a defensible answer, so these are not scored.
NOT_SCORED = {"chicken pox", "allergy", "drug reaction"}


class NoLLMEngine(RAGEngine):
    """The text pipeline with the LLM switched off, whatever keys are set."""

    def _ensure_llm(self):
        self._client = None


def load_benchmark(split: str) -> list[dict]:
    directory = CACHE_DIR / "symptom_to_diagnosis"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SPLIT_FILES[split]
    if not path.exists():
        urllib.request.urlretrieve(DATASET_URL.format(file=path.name), path)
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            row = json.loads(line)
            rows.append({"index": index, "text": row["input_text"], "label": row["output_text"]})
    unknown = {r["label"] for r in rows} - set(IN_SCOPE) - OUT_OF_SCOPE - NOT_SCORED
    assert not unknown, f"labels without a mapping: {unknown}"
    return rows


def diagnose_all(rows: list[dict]) -> list[dict]:
    store = VectorStore()
    store.sync_guidelines(settings.data_dir / "guidelines")
    preprocessor = TextPreprocessor()
    engine = NoLLMEngine(
        vector_store=store,
        text_preprocessor=preprocessor,
        fallback_engine=ClinicalFallbackEngine(preprocessor=preprocessor),
    )
    threshold = settings.confidence_threshold_nlp

    items = []
    for row in rows:
        diagnosis = engine.diagnose_from_text(row["text"])
        reasoning = diagnosis.get("fallback_reasoning") or {}
        top = reasoning.get("top_disease")
        label = row["label"]
        group = (
            "in_scope" if label in IN_SCOPE
            else "out_of_scope" if label in OUT_OF_SCOPE
            else "not_scored"
        )
        items.append({
            "index": row["index"],
            "label": label,
            "group": group,
            "top_disease": top,
            "top_score": reasoning.get("top_score", 0),
            "runner_up_score": reasoning.get("runner_up_score", 0),
            "n_supporting_symptoms": len(reasoning.get("supporting_symptoms") or []),
            "confidence": diagnosis["confidence"],
            "shown": diagnosis["confidence"] >= threshold,
            "correct": group == "in_scope" and top == IN_SCOPE[label],
        })
    return items


def summarize(items: list[dict]) -> dict:
    scored = [i for i in items if i["group"] != "not_scored"]
    in_scope = [i for i in scored if i["group"] == "in_scope"]
    out_of_scope = [i for i in scored if i["group"] == "out_of_scope"]
    shown = [i for i in scored if i["shown"]]

    def share(part: list, whole: list) -> float | None:
        return round(len(part) / len(whole), 4) if whole else None

    per_label = defaultdict(lambda: {"n": 0, "correct": 0, "shown": 0, "shown_correct": 0})
    for i in scored:
        entry = per_label[i["label"]]
        entry["n"] += 1
        entry["correct"] += i["correct"]
        entry["shown"] += i["shown"]
        entry["shown_correct"] += i["shown"] and i["correct"]

    bins = defaultdict(lambda: [0, 0])
    for i in scored:
        low = min(int(i["confidence"] * 10), 9) / 10
        bins[low][0] += i["correct"]
        bins[low][1] += 1

    return {
        "n_scored": len(scored),
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(out_of_scope),
        "top1_accuracy_in_scope": share([i for i in in_scope if i["correct"]], in_scope),
        "gate": {
            "threshold": settings.confidence_threshold_nlp,
            "shown": len(shown),
            "shown_correct": len([i for i in shown if i["correct"]]),
            "precision_of_shown": share([i for i in shown if i["correct"]], shown),
            "in_scope_shown_and_correct": share(
                [i for i in in_scope if i["shown"] and i["correct"]], in_scope
            ),
            "out_of_scope_shown": share([i for i in out_of_scope if i["shown"]], out_of_scope),
        },
        "per_label": dict(per_label),
        "accuracy_by_confidence": {
            f"{low:.1f}-{low + 0.1:.1f}": {"n": n, "accuracy": round(correct / n, 4)}
            for low, (correct, n) in sorted(bins.items())
        },
    }


def run(split: str) -> dict:
    rows = load_benchmark(split)
    items = diagnose_all(rows)
    summary = summarize(items)
    gate = summary["gate"]

    print(f"\nFallback engine [{split}]: {len(rows)} texts, {summary['n_scored']} scored "
          f"({summary['n_in_scope']} with a condition the engine knows, "
          f"{summary['n_out_of_scope']} with one it does not)")
    print(f"  first diagnosis is the labeled one, conditions it knows: "
          f"{summary['top1_accuracy_in_scope']:.1%}")
    print(f"  gate at confidence >= {gate['threshold']:.2f}: {gate['shown']} shown, "
          f"{gate['shown_correct']} of them correct"
          + (f" ({gate['precision_of_shown']:.1%})" if gate["shown"] else ""))
    print(f"    correct diagnoses shown, of conditions it knows: {gate['in_scope_shown_and_correct']:.1%}")
    print(f"    unknown conditions shown with some diagnosis:   {gate['out_of_scope_shown']:.1%}")
    print("\n  accuracy by confidence:")
    for band, cell in summary["accuracy_by_confidence"].items():
        print(f"    {band}   n={cell['n']:<4} accuracy={cell['accuracy']:.1%}")
    print("\n  per label (n, first diagnosis correct, shown, shown and correct):")
    for label, cell in sorted(summary["per_label"].items()):
        scope = "known  " if label in IN_SCOPE else "unknown"
        print(f"    {scope} {label:<34} {cell['n']:>3} {cell['correct']:>4} {cell['shown']:>4} {cell['shown_correct']:>4}")

    result = {"split": split, "benchmark": "gretelai/symptom_to_diagnosis",
              "revision": DATASET_REVISION, "n_texts": len(rows), **summary, "items": items}
    save_result(f"fallback_{split}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=list(SPLIT_FILES))
    run(parser.parse_args().split)


if __name__ == "__main__":
    main()
