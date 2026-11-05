"""
Compare guideline judges on labeled (diagnosis, retrieved passages) pairs.

Each pair is a diagnosis label and the three passages GuidelineValidator
retrieved for it from the guideline library. The label says whether those
passages support the diagnosis, and was written before any judge was run.

Systems:
  tokens           GuidelineValidator._validate_with_tokens (the fallback)
  gemini           GeminiJudge: the judge before it was separated from the
                   generator, and still the judge without an OpenAI key
  openai_original  the OpenAI judge model given the Gemini judge's prompt, to
                   separate what the model changes from what the
                   instructions change
  openai           OpenAIJudge, as the application runs it

Usage:
    python -m evaluation.judge_eval --split heldout
    python -m evaluation.judge_eval --split dev --systems tokens,gemini,openai_original,openai
    python -m evaluation.judge_eval --split dev --systems tokens,gemini,openai --cached-only gemini

Splits: "dev" was used to write the judge instructions; "heldout" was not run
until they were final. Responses are cached in evaluation/.cache, so a re-run
costs nothing and needs no API key.

The free tier of the Gemini API allows a few dozen requests per day (20 for
gemini-2.5-flash when this was written), and every Gemini call made here
comes out of the allowance the application itself needs. --cached-only lists
the systems that must not send anything; the cache fills up over several days.
"""

import argparse
import hashlib
import json
import re
import threading
import time
from collections import defaultdict

from pydantic import BaseModel

from app.core.validation.guideline_judge import (
    GEMINI_JUDGE_PROMPT, JUDGE_INSTRUCTIONS, GeminiJudge, JudgeVerdict,
    OpenAIJudge, gemini_judge_prompt,
)
from app.core.validation.guideline_validator import (
    VALIDATION_THRESHOLD, GuidelineValidator,
)
from evaluation.common import (
    CACHE_DIR, binary_metrics, load_jsonl, print_metrics, run_parallel,
    save_result,
)

# OpenAI list price for gpt-6-luna, USD per million tokens
OPENAI_USD_PER_MILLION = {"input_tokens": 0.10, "output_tokens": 0.50}

RATE_LIMIT_RETRIES = 6
RATE_LIMIT_WAIT_SECONDS = 20.0


class ResponseStore:
    """Judge responses on disk, shared by every system in a run."""

    def __init__(self, name: str = "judge_responses.json"):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._path = CACHE_DIR / name
        self._lock = threading.Lock()
        self._records: dict[str, dict] = (
            json.loads(self._path.read_text(encoding="utf-8")) if self._path.exists() else {}
        )
        self.live_calls: dict[str, int] = defaultdict(int)

    def get(self, key: str) -> dict | None:
        with self._lock:
            return self._records.get(key)

    def put(self, key: str, record: dict) -> None:
        with self._lock:
            self._records[key] = record
            self.live_calls[record["system"]] += 1
            self._path.write_text(json.dumps(self._records, ensure_ascii=True), encoding="utf-8")

    def records(self, keys: set[str]) -> list[dict]:
        with self._lock:
            return [self._records[k] for k in keys if k in self._records]


class CachedJudge:
    """Mixin for a GuidelineJudge: responses are stored on disk, calls are
    measured, and a failed call stops the run.

    In the application a failed judge falls back to token matching. In an
    evaluation that would silently report token matching as the judge.
    """

    system = ""   # name of the compared system, also the cache namespace
    prompt = ""   # part of the cache key: changed instructions are a new request

    def __init__(self, store: ResponseStore, cached_only: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._store = store
        # Grade only what is already on disk and send nothing
        self._cached_only = cached_only
        self.keys: set[str] = set()

    @property
    def available(self) -> bool:
        # A cached run needs no key
        return True

    def judge(self, diagnosis: str, passages: list[str]) -> dict | None:
        """The verdict, or None for a pair that is not cached in a cached-only run."""
        if self._cached_only and self._store.get(self._key(diagnosis, passages)) is None:
            return None
        verdict = super().judge(diagnosis, passages)
        if verdict is None:
            raise RuntimeError(
                f"{self.system}: the judge call failed (missing API key, quota, "
                "or the API is unreachable) and the response is not cached"
            )
        return verdict

    def _key(self, diagnosis: str, passages: list[str]) -> str:
        payload = json.dumps(
            [self.system, self.model, getattr(self, "_reasoning_effort", ""),
             hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(), diagnosis, passages],
            sort_keys=True, ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _request(self, diagnosis: str, passages: list[str]) -> tuple[JudgeVerdict, dict]:
        key = self._key(diagnosis, passages)
        self.keys.add(key)
        hit = self._store.get(key)
        if hit is not None:
            return JudgeVerdict(**hit["verdict"]), hit["usage"]

        for attempt in range(RATE_LIMIT_RETRIES):
            started = time.perf_counter()
            try:
                verdict, usage = super()._request(diagnosis, passages)
                break
            except Exception as e:
                rate_limited = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                # A per-minute limit clears by waiting. A daily quota does
                # not: the API then asks to retry in hours.
                wait = re.search(r"retryDelay'?: '?(\d+)", str(e))
                daily_quota = bool(wait) and int(wait.group(1)) > RATE_LIMIT_WAIT_SECONDS * 3
                if not rate_limited or daily_quota or attempt == RATE_LIMIT_RETRIES - 1:
                    raise
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
        elapsed = time.perf_counter() - started
        self._store.put(key, {
            "system": self.system,
            "model": self.model,
            "verdict": verdict.model_dump(),
            "usage": usage,
            "seconds": round(elapsed, 3),
        })
        return verdict, usage


class PromptedVerdict(BaseModel):
    """The JSON object GEMINI_JUDGE_PROMPT asks for."""
    is_consistent: bool
    support_score: float
    reasoning: str
    negation_detected: bool
    synonym_matches: list[str]


class OriginalPromptOpenAIJudge(OpenAIJudge):
    """The OpenAI judge model given the Gemini judge's prompt."""

    def _request(self, diagnosis: str, passages: list[str]) -> tuple[JudgeVerdict, dict]:
        response = self._get_client().responses.parse(
            model=self.model,
            input=[{"role": "user", "content": gemini_judge_prompt(diagnosis, passages)}],
            text_format=PromptedVerdict,
            reasoning={"effort": self._reasoning_effort},
            store=False,
        )
        prompted = response.output_parsed
        if prompted is None:
            raise ValueError("the model returned no verdict (refusal or empty output)")
        verdict = JudgeVerdict(**prompted.model_dump(exclude={"is_consistent"}))
        return verdict, {
            "input_tokens": response.usage.input_tokens or 0,
            "output_tokens": response.usage.output_tokens or 0,
        }


class CachedGeminiJudge(CachedJudge, GeminiJudge):
    system, prompt = "gemini", GEMINI_JUDGE_PROMPT


class CachedOriginalPromptOpenAIJudge(CachedJudge, OriginalPromptOpenAIJudge):
    system, prompt = "openai_original", GEMINI_JUDGE_PROMPT


class CachedOpenAIJudge(CachedJudge, OpenAIJudge):
    system, prompt = "openai", JUDGE_INSTRUCTIONS


JUDGES = {
    "gemini": CachedGeminiJudge,
    "openai_original": CachedOriginalPromptOpenAIJudge,
    "openai": CachedOpenAIJudge,
}
WORKERS = {"gemini": 2, "openai_original": 4, "openai": 4}


def _usage(store: ResponseStore, judge: CachedJudge) -> dict:
    """Latency and tokens over the responses this run used."""
    records = store.records(judge.keys)
    seconds = sorted(r["seconds"] for r in records)
    tokens = {
        kind: sum(r["usage"].get(kind, 0) for r in records)
        for kind in ("input_tokens", "output_tokens")
    }
    usage = {
        "model": judge.model,
        "requests": len(records),
        "live_calls_this_run": store.live_calls[judge.system],
        **tokens,
        "latency_median_s": seconds[len(seconds) // 2] if seconds else None,
        "latency_p95_s": seconds[int(len(seconds) * 0.95)] if seconds else None,
    }
    if judge.provider == "openai":
        usage["cost_usd"] = round(
            sum(tokens[kind] / 1_000_000 * price for kind, price in OPENAI_USD_PER_MILLION.items()), 5
        )
    return usage


def _score_table(
    items: list[dict], scores: dict[str, dict], refused: dict[str, set]
) -> tuple[dict, dict, dict]:
    """Metrics, correct-per-category and errors of every system over `items`.

    `refused` holds, per system, the items it did not pass whatever the score.
    """
    metrics, by_category, errors = {}, {}, {}
    for system, graded in scores.items():
        pairs, per_category, wrong = [], defaultdict(lambda: [0, 0]), []
        for item in items:
            score = graded[item["id"]]
            if score is None:
                continue
            truth = item["label"] == "supported"
            predicted = (
                score >= VALIDATION_THRESHOLD
                and item["id"] not in refused.get(system, ())
            )
            pairs.append((truth, predicted))
            per_category[item["category"]][0] += truth == predicted
            per_category[item["category"]][1] += 1
            if truth != predicted:
                wrong.append({"id": item["id"], "diagnosis": item["diagnosis"],
                              "label": item["label"], "score": round(score, 2)})
        metrics[system] = binary_metrics(pairs)
        by_category[system] = {c: f"{ok}/{n}" for c, (ok, n) in per_category.items()}
        errors[system] = wrong
    return metrics, by_category, errors


def run_judges(split: str, systems: list[str], cached_only: set[str] = frozenset()) -> dict:
    items = load_jsonl(f"judge_{split}.jsonl")
    store = ResponseStore()
    validator = GuidelineValidator(vector_store=object())

    scores: dict[str, dict[str, float | None]] = {}
    refused: dict[str, set] = {}
    usage = {}
    for system in systems:
        if system == "tokens":
            matched = {
                item["id"]: validator._validate_with_tokens(item["diagnosis"], item["passages"])
                for item in items
            }
            scores[system] = {i: result[1] for i, result in matched.items()}
            # Token matching does not pass a label whose stated measurement
            # it cannot check, however many words match
            refused[system] = {i for i, result in matched.items() if not result[0]}
            continue
        judge = JUDGES[system](store, cached_only=system in cached_only)

        def grade(item, judge=judge):
            verdict = judge.judge(item["diagnosis"], [p["text"] for p in item["passages"]])
            return verdict["support_score"] if verdict else None

        graded = run_parallel(grade, items, workers=WORKERS[system])
        if all(score is None for score in graded):
            print(f"\n{system}: no pair of this split is cached, nothing to report")
            continue
        scores[system] = {item["id"]: score for item, score in zip(items, graded)}
        usage[system] = _usage(store, judge)

    clear = [item for item in items if item["label"] != "ambiguous"]
    metrics, by_category, errors = _score_table(clear, scores, refused)
    print_metrics(
        f"Guideline judge [{split}], clear cases only", metrics,
        "the passages support the diagnosis",
    )
    print("  FN = a supported diagnosis sent to review; FP = an unsupported diagnosis passed as consistent")

    # A system that could not grade every pair is only comparable on the
    # pairs that every system graded.
    common = [item for item in clear if all(scores[s][item["id"]] is not None for s in scores)]
    common_metrics = None
    if len(common) < len(clear):
        common_metrics = _score_table(common, scores, refused)[0]
        print_metrics(
            f"Guideline judge [{split}], the {len(common)} of {len(clear)} clear cases graded by every system",
            common_metrics, "the passages support the diagnosis",
        )

    print("\n  correct per category:")
    categories = list(dict.fromkeys(item["category"] for item in clear))
    print(f"  {'system':<16}" + "".join(f"{c[:13]:>14}" for c in categories))
    for system in scores:
        print(f"  {system:<16}" + "".join(f"{by_category[system].get(c, '-'):>14}" for c in categories))
    for system, wrong in errors.items():
        for e in wrong:
            print(f"  wrong [{system}] {e['id']} {e['diagnosis']!r}: {e['label']}, scored {e['score']}")

    def shown(score: float | None) -> str:
        return "n/a" if score is None else f"{score:.2f}"

    ambiguous = [item for item in items if item["label"] == "ambiguous"]
    if ambiguous:
        print("\n  ambiguous cases (not scored), support score per system:")
        for item in ambiguous:
            row = "  ".join(f"{system}={shown(scores[system][item['id']])}" for system in scores)
            print(f"  {item['id']} {item['diagnosis']!r}: {row}")
    for system, u in usage.items():
        print(f"  usage [{system}]: {u}")

    result = {
        "split": split,
        "threshold": VALIDATION_THRESHOLD,
        "n_items": len(items),
        "n_clear": len(clear),
        "metrics": metrics,
        "n_graded_by_every_system": len(common),
        "metrics_graded_by_every_system": common_metrics,
        "correct_per_category": by_category,
        "errors": errors,
        "usage": usage,
        "items": [
            {"id": item["id"], "diagnosis": item["diagnosis"], "category": item["category"],
             "label": item["label"],
             "scores": {
                 system: None if scores[system][item["id"]] is None
                 else round(scores[system][item["id"]], 3)
                 for system in scores
             }}
            for item in items
        ],
    }
    save_result(f"judge_{split}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="heldout", choices=["dev", "heldout"])
    parser.add_argument(
        "--systems", default="tokens,openai",
        help="comma-separated: tokens, gemini, openai_original, openai "
             "(gemini uses the free-tier daily allowance, see above)",
    )
    parser.add_argument(
        "--cached-only", default="",
        help="comma-separated systems that grade only the pairs already in the "
             "cache and send no request (for a provider whose quota is used up)",
    )
    args = parser.parse_args()

    def names(value: str) -> list[str]:
        return [s.strip() for s in value.split(",") if s.strip()]

    run_judges(args.split, names(args.systems), set(names(args.cached_only)))


if __name__ == "__main__":
    main()
