"""
Measure the multimodal fusion end to end on a public chest X-ray collection.

The collection is Open-I / Indiana University Chest X-ray (NLM, CC BY-NC-ND
4.0): radiology reports with the images they describe, indexed with MeSH terms
by the collection's curators. It was not written for this project. The
archives are downloaded by hand into evaluation/.cache/openi and are not
redistributed here:

    openi/reports/ecgen-radiology/*.xml     NLMCXR_reports.tgz
    openi/images/<image id>.png             NLMCXR_png.tgz (first N images)

For every report whose first image is on disk, the application's path is run
without an LLM: the "indication" field of the report (why the film was taken)
is the patient text, the image goes through the DenseNet-121, the two go
through FusionEngine. Three systems are scored against the report's MeSH
terms:

  - image only : the CV model's own diagnosis
  - text only  : the rule-based engine on the indication
  - fused      : FusionEngine.fuse(text, image)

Ground truth: the case has a target finding when its MeSH terms name one of
TARGETS; it is normal when the only MeSH term is "normal". Cases with other
abnormalities only (calcified granuloma, scoliosis, ...) are left out: the
CV model cannot name them and the report does not say they are absent.

Questions:
  1. Of the cases with a target finding, how often is the system's diagnosis
     one of the findings present?
  2. Of the normal cases, how often is a finding (a false alarm) reported?
  3. Does the confidence rank right diagnoses above wrong ones (AUROC)?
  4. How often does fusion change the outcome relative to the image alone?

Usage:
    python -m evaluation.fusion_eval --limit 500
No API is called.
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET

from app.config import settings
from app.api.dependencies import get_cv_model
from app.core.fusion.fusion_engine import FusionEngine
from app.core.nlp.clinical_fallback import ClinicalFallbackEngine
from app.core.nlp.rag_engine import RAGEngine
from app.core.nlp.text_preprocessor import TextPreprocessor
from app.core.nlp.vector_store import VectorStore
from evaluation.common import CACHE_DIR, save_result
from evaluation.fallback_eval import NoLLMEngine

OPENI_DIR = CACHE_DIR / "openi"

# MeSH major term -> the CV class that names it. Written before the model was
# run on this collection.
TARGETS = {
    "cardiomegaly": "cardiomegaly",
    "pleural effusion": "effusion",
    "pulmonary atelectasis": "atelectasis",
    "nodule": "nodule",
    "pulmonary edema": "edema",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "pulmonary emphysema": "emphysema",
    "pulmonary fibrosis": "fibrosis",
    "hernia, hiatal": "hernia",
}


def load_reports(limit: int) -> list[dict]:
    """Reports whose first image is on disk, in image-file order."""
    images = OPENI_DIR / "images"
    rows = []
    for path in sorted((OPENI_DIR / "reports" / "ecgen-radiology").glob("*.xml"), key=lambda p: int(p.stem)):
        root = ET.parse(path).getroot()
        image_ids = [p.get("id") for p in root.findall("parentImage")]
        image = next((images / f"{i}.png" for i in image_ids if (images / f"{i}.png").exists()), None)
        if image is None:
            continue
        indication = next(
            (a.text or "" for a in root.iter("AbstractText") if a.get("Label") == "INDICATION"), ""
        ).strip()
        mesh = [m.text.split("/")[0].strip().lower() for m in root.findall("./MeSH/major") if m.text]
        found = sorted({TARGETS[m] for m in mesh if m in TARGETS})
        if found:
            group = "finding"
        elif mesh == ["normal"]:
            group = "normal"
        else:
            continue
        # XXXX marks text the collection anonymised
        indication = re.sub(r"\s*X{3,}\s*", " ", indication).strip()
        rows.append({"report": path.stem, "image": str(image), "indication": indication,
                     "group": group, "truth": found})
        if len(rows) >= limit:
            break
    return rows


def names_a_present_finding(diagnosis: str, truth: list[str]) -> bool:
    """Whether a diagnosis names one of the findings the report lists."""
    engine = FusionEngine
    text = (diagnosis or "").lower()
    for finding in truth:
        terms = [finding] + engine.ONTOLOGY_MAPPING.get(finding, [])
        if any(engine._mentions(text, term) for term in terms):
            return True
    return False


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    positives = [s for s, l in zip(scores, labels) if l]
    negatives = [s for s, l in zip(scores, labels) if not l]
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return round(wins / (len(positives) * len(negatives)), 4)


LLM_CACHE = CACHE_DIR / "fusion_gemini.json"


def evaluate(rows: list[dict], llm: bool = False) -> list[dict]:
    """Run the three systems on every row.

    With llm=True the text is analysed by the real generator (Gemini), one
    case at a time, each answer kept on disk. The run stops at the first case
    the generator could not answer (quota, timeout): the rules would answer
    in its place, and their result must not be reported as Gemini's.
    """
    store = VectorStore()
    store.sync_guidelines(settings.data_dir / "guidelines")
    preprocessor = TextPreprocessor()
    engine_class = RAGEngine if llm else NoLLMEngine
    rag = engine_class(
        vector_store=store, text_preprocessor=preprocessor,
        fallback_engine=ClinicalFallbackEngine(preprocessor=preprocessor),
    )
    cv_model = get_cv_model()
    fusion = FusionEngine()
    cache = json.loads(LLM_CACHE.read_text(encoding="utf-8")) if llm and LLM_CACHE.exists() else {}
    stopped = False

    items = []
    for n, row in enumerate(rows, 1):
        cv = cv_model.predict(row["image"])
        nlp = None
        if row["indication"] and llm:
            if row["report"] in cache:
                nlp = cache[row["report"]]
            elif stopped:
                break
            else:
                nlp = rag.diagnose_from_text(row["indication"])
                if "fallback_reasoning" in (nlp.get("nlp_details") or nlp):
                    print(f"  generator unavailable at report {row['report']}; stopping", flush=True)
                    stopped = True
                    break
                cache[row["report"]] = nlp
                LLM_CACHE.write_text(json.dumps(cache, ensure_ascii=True, default=str), encoding="utf-8")
        elif row["indication"]:
            nlp = rag.diagnose_from_text(row["indication"])
        elif llm:
            continue
        fused = fusion.fuse(nlp_diagnosis=nlp, cv_diagnosis=cv)

        detected = bool(cv.get("finding_detected"))
        cv_name = cv.get("predicted_class", "") if detected else ""
        nlp_name = (nlp or {}).get("primary_diagnosis", "")
        fused_name = fused.get("primary_diagnosis", "")
        fused_detected = detected or (
            fused.get("agreement") in ("complementary", "concordant", "discordant")
            and fused.get("combined_confidence", 0) >= settings.confidence_threshold_nlp
            and fused_name not in ("", cv.get("predicted_class", ""))
        )
        items.append({
            **{k: row[k] for k in ("report", "indication", "group", "truth")},
            "cv": {"name": cv_name, "detected": detected, "confidence": cv.get("confidence", 0)},
            "nlp": {"name": nlp_name, "confidence": (nlp or {}).get("confidence", 0)} if nlp else None,
            "fused": {"name": fused_name, "confidence": fused.get("combined_confidence", 0),
                      "strategy": fused.get("fusion_strategy"), "severity": fused.get("severity"),
                      "reports_a_finding": fused_detected},
        })
        if n % 50 == 0:
            print(f"  {n}/{len(rows)}", flush=True)
    return items


def summarize(items: list[dict]) -> dict:
    sick = [i for i in items if i["group"] == "finding"]
    normal = [i for i in items if i["group"] == "normal"]

    def right(item, system):
        entry = item[system]
        if entry is None:
            return False
        if system == "cv" and not entry["detected"]:
            return False
        return names_a_present_finding(entry["name"], item["truth"])

    def share(part, whole):
        return round(len(part) / len(whole), 4) if whole else None

    systems = {}
    for system in ("cv", "nlp", "fused"):
        scored = [i for i in sick if i[system] is not None]
        if system == "nlp":
            false_alarm = [i for i in normal if i["nlp"] and names_a_present_finding(i["nlp"]["name"], list(TARGETS.values()))]
        elif system == "cv":
            false_alarm = [i for i in normal if i["cv"]["detected"]]
        else:
            false_alarm = [i for i in normal if i["fused"]["reports_a_finding"]]
        conf_items = [i for i in sick if i[system] is not None]
        systems[system] = {
            "cases_with_output": len(scored),
            "diagnosis_is_a_present_finding": share([i for i in sick if right(i, system)], sick),
            "false_alarm_rate_on_normal": share(false_alarm, normal),
            "confidence_auroc": auroc(
                [i[system]["confidence"] for i in conf_items], [right(i, system) for i in conf_items]
            ),
        }

    both = [i for i in sick if i["nlp"] is not None]
    strategies = {}
    for i in items:
        if i["nlp"] is not None:
            strategies[i["fused"]["strategy"]] = strategies.get(i["fused"]["strategy"], 0) + 1

    return {
        "n_cases": len(items), "n_with_finding": len(sick), "n_normal": len(normal),
        "n_with_indication": sum(1 for i in items if i["nlp"] is not None),
        "systems": systems,
        "fusion_strategies_when_text_present": strategies,
        "fusion_changed_diagnosis_vs_image": len([i for i in both if i["fused"]["name"] != i["cv"]["name"]]),
        "cases_with_text_and_finding": len(both),
        "target_prevalence": {t: sum(t in i["truth"] for i in sick) for t in sorted(set(TARGETS.values()))},
    }


def run(limit: int, llm: bool = False) -> dict:
    rows = load_reports(limit)
    print(f"Fusion on Open-I: {len(rows)} reports with an image on disk"
          + (" (text analysed by the real generator)" if llm else ""))
    items = evaluate(rows, llm)
    summary = summarize(items)
    print(f"\n{summary['n_with_finding']} with a target finding, {summary['n_normal']} normal, "
          f"{summary['n_with_indication']} with an indication text")
    print(f"  {'system':<8}{'finding named':>15}{'false alarm':>13}{'conf AUROC':>12}")
    for name, s in summary["systems"].items():
        fa = s["false_alarm_rate_on_normal"]
        au = s["confidence_auroc"]
        print(f"  {name:<8}{s['diagnosis_is_a_present_finding']:>15.1%}"
              f"{(f'{fa:.1%}' if fa is not None else '-'):>13}{(f'{au:.3f}' if au is not None else '-'):>12}")
    print(f"  fusion strategies (text present): {summary['fusion_strategies_when_text_present']}")
    print(f"  fused diagnosis differs from the image's: {summary['fusion_changed_diagnosis_vs_image']} "
          f"of {summary['cases_with_text_and_finding']} cases with a finding and a text")
    save_result("fusion_openi_llm.json" if llm else "fusion_openi.json", {"dataset": "Open-I / IU chest X-ray", "limit": limit,
                                      "targets": TARGETS, **summary, "items": items})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--llm", action="store_true",
                        help="analyse the text with Gemini until its quota is spent")
    args = parser.parse_args()
    run(args.limit, args.llm)


if __name__ == "__main__":
    main()
