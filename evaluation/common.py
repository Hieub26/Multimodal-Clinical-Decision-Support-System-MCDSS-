"""
Shared pieces of the evaluation scripts: dataset loading, a Jev judge that
caches every response on disk, and metric and reporting helpers.

The cache makes a re-run free and reproducible: the same request is never
sent twice, and the published numbers can be recomputed without an API key.
"""

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.core.nlp.jev_judge import JevJudge, _plain_answers

EVAL_DIR = Path(__file__).resolve().parent
DATASET_DIR = EVAL_DIR / "datasets"
CACHE_DIR = EVAL_DIR / ".cache"
RESULTS_DIR = EVAL_DIR / "results"

# Jev pricing (USD per million input tokens; output tokens are free)
USD_PER_MILLION_INPUT_TOKENS = 0.042


def load_jsonl(name: str) -> list[dict]:
    path = DATASET_DIR / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class CachedJevJudge(JevJudge):
    """JevJudge whose responses are stored on disk and whose calls are measured."""

    def __init__(self, cache_name: str = "jev_responses.json", **kwargs):
        super().__init__(**kwargs)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._disk_path = CACHE_DIR / cache_name
        self._disk_lock = threading.Lock()
        self._disk: dict[str, dict] = (
            json.loads(self._disk_path.read_text(encoding="utf-8"))
            if self._disk_path.exists() else {}
        )
        self.live_calls = 0
        self.input_tokens = 0
        self.latencies: list[float] = []

    @property
    def available(self) -> bool:
        # A cached run needs no key
        return True

    def _ask(self, state: dict, questions: dict[str, dict]) -> dict[str, dict]:
        # In the application a failed call falls back to the rules. In an
        # evaluation that would silently report rule results as Jev's.
        answers = super()._ask(state, questions)
        if answers is None:
            raise RuntimeError(
                "Jev call failed during evaluation (missing TYPESAFE_API_KEY, "
                "or the API is unreachable) and the response is not cached"
            )
        return answers

    def _request(self, state: dict, questions: dict[str, dict]) -> dict[str, dict]:
        payload = json.dumps([self.model, state, questions], sort_keys=True, ensure_ascii=True)
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with self._disk_lock:
            hit = self._disk.get(key)
        if hit is not None:
            return hit["answers"]

        started = time.perf_counter()
        response = self._get_client().system_one(state=state, questions=questions)
        elapsed = time.perf_counter() - started
        answers = _plain_answers(response)
        record = {
            "answers": answers,
            "input_tokens": response.usage.input_tokens or 0,
            "seconds": round(elapsed, 4),
            "model": response.model,
        }
        with self._disk_lock:
            self._disk[key] = record
            self.live_calls += 1
            self.input_tokens += record["input_tokens"]
            self.latencies.append(elapsed)
        return answers

    def save(self) -> None:
        with self._disk_lock:
            self._disk_path.write_text(
                json.dumps(self._disk, ensure_ascii=True), encoding="utf-8"
            )

    def cached_usage(self) -> dict:
        """Tokens and latency over everything in the cache."""
        with self._disk_lock:
            records = list(self._disk.values())
        seconds = sorted(r["seconds"] for r in records)
        tokens = sum(r["input_tokens"] for r in records)
        return {
            "requests": len(records),
            "input_tokens": tokens,
            "cost_usd": round(tokens / 1_000_000 * USD_PER_MILLION_INPUT_TOKENS, 5),
            "latency_median_ms": round(seconds[len(seconds) // 2] * 1000) if seconds else None,
            "latency_p95_ms": round(seconds[int(len(seconds) * 0.95)] * 1000) if seconds else None,
        }


def run_parallel(function, items: list, workers: int = 8) -> list:
    """Map with a small pool (the public endpoint is rate limited)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items))


def binary_metrics(pairs: list[tuple[bool, bool]]) -> dict:
    """Metrics for (truth, prediction) pairs; the positive class is True."""
    tp = sum(1 for truth, pred in pairs if truth and pred)
    tn = sum(1 for truth, pred in pairs if not truth and not pred)
    fp = sum(1 for truth, pred in pairs if not truth and pred)
    fn = sum(1 for truth, pred in pairs if truth and not pred)
    total = len(pairs)

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision and recall else None
    )
    return {
        "n": total,
        "accuracy": ratio(tp + tn, total),
        "sensitivity": recall,            # recall of the positive class
        "specificity": ratio(tn, tn + fp),
        "precision": precision,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def print_metrics(title: str, metrics: dict[str, dict], positive: str) -> None:
    print(f"\n{title}  (positive class: {positive})")
    print(f"  {'system':<34}{'n':>5}{'acc':>8}{'sens':>8}{'spec':>8}{'prec':>8}{'F1':>8}{'FN':>5}{'FP':>5}")
    for name, m in metrics.items():
        cells = [m["accuracy"], m["sensitivity"], m["specificity"], m["precision"], m["f1"]]
        shown = "".join(f"{c:>8.3f}" if c is not None else f"{'-':>8}" for c in cells)
        print(f"  {name:<34}{m['n']:>5}{shown}{m['fn']:>5}{m['fp']:>5}")


def save_result(name: str, payload: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
