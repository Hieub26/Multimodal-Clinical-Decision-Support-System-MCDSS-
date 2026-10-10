"""
Compare the rule-based clinical context logic with Jev on labeled data.

Three tasks, each run through the same code paths the application uses:

  context   Is a mention a current finding of the patient?
            TextPreprocessor with and without the Jev judge.
  gate      Should the red-flag gate escalate this text?
            SafetyController.patient_text_flags with and without Jev.
  negex     Is a concept negated? The public NegEx annotation set
            (Chapman et al.), which neither system was built on.

Usage:
    python -m evaluation.run_eval context --split heldout
    python -m evaluation.run_eval gate --split heldout
    python -m evaluation.run_eval negex --split test
    python -m evaluation.run_eval all

Splits: "dev" was used to write the rules and to tune Jev's questions and
thresholds; "heldout" / "test" were not looked at while doing either.
Jev responses are cached in evaluation/.cache, so a re-run costs nothing.
"""

import argparse
import re
import urllib.request
from collections import Counter, defaultdict

from app.core.nlp.jev_judge import DENIED
from app.core.nlp.text_preprocessor import (
    TextPreprocessor, is_negated, normalize_contractions,
)
from app.core.safety.safety_controller import SafetyController
from evaluation.common import (
    CACHE_DIR, CachedJevJudge, binary_metrics, load_jsonl, print_metrics,
    run_parallel, save_result,
)

NEGEX_URL = (
    "https://raw.githubusercontent.com/mongoose54/negex/master/"
    "negex.python/Annotations-1-120.txt"
)
# Reports 1-20 were used while tuning Jev's questions; the rest are the test set
NEGEX_DEV_REPORTS = 20
# Annotated concepts that are themselves a negative or normal finding
NEGATIVE_CONCEPT = re.compile(
    r"\b(no|non\w*|not|without|negative|normal\w*|unremarkable|free|clear|afebrile|"
    r"atraumatic|asymptomatic|anicteric|nontender|den(y|ies|ied)|absent|intact|stable|"
    r"well|benign|regular|supple|soft|within normal limits)\b"
)


# ----------------------------------------------------------------------
# Task: mention context
# ----------------------------------------------------------------------

def run_context(split: str, judge: CachedJevJudge) -> dict:
    items = load_jsonl(f"context_{split}.jsonl")
    systems = {
        "rules": TextPreprocessor(),
        "jev": TextPreprocessor(context_judge=judge),
    }

    def extract(item):
        return {name: tp.preprocess(item["text"]) for name, tp in systems.items()}

    outputs = run_parallel(extract, items)
    judge.save()

    pairs = defaultdict(list)
    by_kind = defaultdict(lambda: defaultdict(list))
    rows = []
    for item, output in zip(items, outputs):
        for mention in item["mentions"]:
            truth = mention["kind"] == "present"
            row = {"id": item["id"], "text": item["text"], "mention": mention["mention"],
                   "kind": mention["kind"]}
            for name in systems:
                predicted = mention["mention"] in output[name]["extracted_symptoms"]
                pairs[name].append((truth, predicted))
                by_kind[mention["kind"]][name].append(truth == predicted)
                row[name] = predicted
                row[f"{name}_context"] = output[name]["mention_contexts"].get(mention["mention"])
            rows.append(row)

    metrics = {name: binary_metrics(p) for name, p in pairs.items()}
    print_metrics(f"Mention context [{split}]", metrics, "mention is a current finding")
    print("\n  accuracy by kind of mention:")
    kinds = {}
    for kind, per_system in sorted(by_kind.items()):
        kinds[kind] = {name: round(sum(v) / len(v), 4) for name, v in per_system.items()}
        n = len(next(iter(per_system.values())))
        print(f"    {kind:<14} n={n:<4}" + "".join(f" {name}={acc:.3f}" for name, acc in kinds[kind].items()))

    errors = [r for r in rows if r["rules"] != (r["kind"] == "present") or r["jev"] != (r["kind"] == "present")]
    print(f"\n  disagreements with the label ({len(errors)}):")
    for r in errors:
        truth = r["kind"] == "present"
        mark = ("R" if r["rules"] != truth else "-") + ("J" if r["jev"] != truth else "-")
        print(f"    [{mark}] {r['id']} {r['mention']:<20} label={r['kind']:<13} "
              f"rules={r['rules_context']:<13} jev={r['jev_context']:<13} | {r['text']}")

    result = {"split": split, "metrics": metrics, "accuracy_by_kind": kinds, "items": rows}
    save_result(f"context_{split}.json", result)
    return result


# ----------------------------------------------------------------------
# Task: red-flag gate
# ----------------------------------------------------------------------

def run_gate(split: str, judge: CachedJevJudge) -> dict:
    items = load_jsonl(f"gate_{split}.jsonl")
    controllers = {
        "rules": SafetyController(),
        "rules + jev signs": SafetyController(context_judge=judge, clear_history_flags=False),
        "rules + jev signs + clearing": SafetyController(context_judge=judge, clear_history_flags=True),
    }

    def assess(item):
        flags = {
            name: controller.patient_text_flags(item.get("symptoms", ""), item.get("question", ""))
            for name, controller in controllers.items()
        }
        return flags

    outputs = run_parallel(assess, items)
    judge.save()

    rows = []
    for item, flags in zip(items, outputs):
        hybrid = flags["rules + jev signs + clearing"]
        row = {
            "id": item["id"], "label": item["label"], "category": item["category"],
            "text": item.get("symptoms") or item.get("question"),
            "rules": flags["rules"].any,
            "jev signs only": bool(flags["rules + jev signs"].emergency_signs),
            "rules + jev signs": flags["rules + jev signs"].any,
            "rules + jev signs + clearing": hybrid.any,
            "rule_keywords": flags["rules"].keywords,
            "jev_signs": {k: round(v, 3) for k, v in flags["rules + jev signs"].emergency_signs.items()},
            "max_signal": round(max(flags["rules + jev signs"].signals.values()), 3),
            "cleared": hybrid.cleared,
        }
        rows.append(row)

    system_names = ["rules", "jev signs only", "rules + jev signs", "rules + jev signs + clearing"]
    clear = [r for r in rows if r["label"] != "ambiguous"]
    metrics = {
        name: binary_metrics([(r["label"] == "escalate", r[name]) for r in clear])
        for name in system_names
    }
    print_metrics(f"Red-flag gate [{split}], clear cases only", metrics, "should escalate")

    print("\n  escalation rate by category (clear cases):")
    categories = {}
    for category in sorted({r["category"] for r in clear}):
        subset = [r for r in clear if r["category"] == category]
        label = subset[0]["label"]
        categories[category] = {"label": label, "n": len(subset)}
        cells = []
        for name in system_names:
            rate = sum(r[name] for r in subset) / len(subset)
            categories[category][name] = round(rate, 4)
            cells.append(f"{rate:>6.2f}")
        print(f"    {category:<24}{label:<12} n={len(subset):<4}" + " ".join(cells))
    print("    (columns: " + " | ".join(system_names) + ")")

    print("\n  wrong on a clear case:")
    for r in clear:
        truth = r["label"] == "escalate"
        wrong = [name for name in system_names if r[name] != truth]
        if wrong:
            marks = "".join("x" if name in wrong else "." for name in system_names)
            print(f"    [{marks}] {r['id']} {r['label']:<12} max={r['max_signal']:.2f} "
                  f"kw={r['rule_keywords']} cleared={r['cleared']} | {r['text'][:110]}")

    print("\n  ambiguous cases (no right answer; shown for behaviour only):")
    for r in rows:
        if r["label"] == "ambiguous":
            decisions = " ".join("FLAG" if r[name] else "pass" for name in system_names)
            print(f"    {r['id']} {decisions} max={r['max_signal']:.2f} | {r['text'][:90]}")

    result = {"split": split, "metrics": metrics, "by_category": categories, "items": rows}
    save_result(f"gate_{split}.json", result)
    return result


# ----------------------------------------------------------------------
# Task: public NegEx benchmark
# ----------------------------------------------------------------------

def _load_negex() -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "negex_annotations.txt"
    if not path.exists():
        urllib.request.urlretrieve(NEGEX_URL, path)
    rows = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        report, concept, sentence, label = line.split("\t")
        rows.append({
            "row": index,
            "report": int(report),
            "concept": " ".join(concept.lower().split()),
            # The file marks the concept in upper case inside the sentence.
            # Lower-casing removes that hint for both systems.
            "sentence": " ".join(sentence.strip().strip('"').lower().split()),
            "negated": label.strip() == "Negated",
        })
    return rows


def run_negex(split: str, judge: CachedJevJudge) -> dict:
    rows = _load_negex()
    rows = [
        r for r in rows
        if (r["report"] <= NEGEX_DEV_REPORTS) == (split == "dev")
    ]

    located = []
    for r in rows:
        text = normalize_contractions(r["sentence"])
        match = re.search(r"\s+".join(re.escape(t) for t in r["concept"].split()), text)
        if match:
            located.append((r, text, match))
    print(f"\nNegEx [{split}]: {len(rows)} annotations, concept located in {len(located)}")

    negex_judge = CachedJevJudge(cache_name="jev_negex.json")

    def ask(entry):
        r = entry[0]
        return negex_judge.judge_mentions(r["sentence"], [r["concept"]])[r["concept"]]

    judgments = run_parallel(ask, located)
    negex_judge.save()

    pairs = {"rules": [], "jev": []}
    items = []
    for (r, text, match), judgment in zip(located, judgments):
        rule_negated = is_negated(text, match.start(), match.end())
        jev_negated = judgment.context == DENIED
        pairs["rules"].append((r["negated"], rule_negated))
        pairs["jev"].append((r["negated"], jev_negated))
        items.append({"row": r["row"], "report": r["report"], "concept": r["concept"],
                      "negated": r["negated"], "rules": rule_negated, "jev": jev_negated,
                      "jev_context": judgment.context})

    metrics = {name: binary_metrics(p) for name, p in pairs.items()}
    print_metrics(f"NegEx annotations [{split}]", metrics, "concept is negated")
    both_wrong = sum(1 for i in items if i["rules"] != i["negated"] and i["jev"] != i["negated"])
    print(f"  wrong in both: {both_wrong} | jev contexts: {dict(Counter(i['jev_context'] for i in items))}")
    print(f"  jev usage: {negex_judge.cached_usage()}")

    # Post hoc, written after reading the errors of the first test run: the
    # annotation marks "afebrile" or "lungs are clear" as affirmed (the phrase
    # is stated), while Jev answers that the patient does not have the
    # finding. Without those concepts the two systems are asked the same thing.
    plain = [i for i in items if not NEGATIVE_CONCEPT.search(i["concept"])]
    post_hoc = {name: binary_metrics([(i["negated"], i[name]) for i in plain]) for name in pairs}
    print_metrics(
        f"NegEx annotations [{split}], post hoc: without the {len(items) - len(plain)} "
        "concepts that are themselves a negative or normal finding",
        post_hoc, "concept is negated",
    )

    result = {"split": split, "n_annotations": len(rows), "n_located": len(located),
              "metrics": metrics,
              "post_hoc_without_negative_concepts": {"n_removed": len(items) - len(plain),
                                                     "metrics": post_hoc},
              "items": items, "usage": negex_judge.cached_usage()}
    save_result(f"negex_{split}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["context", "gate", "negex", "all"])
    parser.add_argument("--split", default=None, help="dev | heldout (context, gate); dev | test (negex)")
    args = parser.parse_args()

    judge = CachedJevJudge()
    if args.task in ("context", "all"):
        run_context(args.split or "heldout", judge)
    if args.task in ("gate", "all"):
        run_gate(args.split or "heldout", judge)
    if args.task in ("negex", "all"):
        split = args.split if args.split in ("dev", "test") else "test"
        run_negex(split, judge)
    if args.task != "negex":
        print(f"\njev usage (context + gate cache): {judge.cached_usage()}")


if __name__ == "__main__":
    main()
