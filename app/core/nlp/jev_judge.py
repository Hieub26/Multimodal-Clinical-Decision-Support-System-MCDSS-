"""
Clinical context judgments from Jev, TypeSafe's System One model.

Code finds the candidate mentions and owns every decision. Jev only returns
typed probabilities for narrow questions about the patient's text:

  * what the text says about each mention (a current finding, denied, past,
    someone else's, or only asked about), and
  * whether the text describes specific emergency warning signs.

Every public method returns None when Jev is disabled, has no API key, or the
call fails. Callers then fall back to the rule-based logic, so the service
keeps working without the external API.
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence

try:
    from typesafe_sdk import RetryPolicy, TypeSafeClient
    HAS_TYPESAFE = True
except ImportError:
    HAS_TYPESAFE = False
    RetryPolicy = TypeSafeClient = None

from app.config import settings
from app.core.nlp.text_preprocessor import (
    DENIED, HYPOTHETICAL, OTHER_PERSON, PAST, PRESENT,
)
from app.utils.logger import describe_error, nlp_logger

# Who "the patient" is. Stated in every question, because the text may be
# written by the patient or by someone describing them.
_PATIENT_DEFINITION = (
    "The patient is the person whose health the text describes: the writer, "
    "or the person the writer is describing, such as their child or parent."
)

# --- What the text says about one mention ---------------------------------
MENTION_CONTEXT_OPTIONS = {
    PRESENT: (
        "The text reports it as a current or recent problem of the patient: "
        "happening now, ongoing, an active condition, or an acute event from "
        "today or the last few days."
    ),
    DENIED: (
        "The text says the patient does not have it: denied, absent, "
        "negative or ruled out."
    ),
    PAST: (
        "The patient had it earlier and it is over: the text says it "
        "resolved, healed or went away, or it is only part of their medical "
        "history from weeks, months or years ago."
    ),
    OTHER_PERSON: (
        "It belongs to someone who is not the patient, such as the medical "
        "history of a relative, friend or coworker."
    ),
    HYPOTHETICAL: (
        "Nobody is reported to have it: it is only asked about, feared, "
        "possible, to be prevented or ruled out, or conditional."
    ),
}

# --- Emergency warning signs -----------------------------------------------
# One narrow yes/no question per sign. Each asks about something happening to
# the patient now, so a relative's history or a denied symptom should score
# low. Thresholds and the decision live in SafetyController.
_IS_AN_EMERGENCY = (
    "The text reports it as happening to the patient now or within the last "
    "day, including when the writer only suspects it."
)
_NOT_AN_EMERGENCY = (
    "It is denied or absent, is only past history, happened to someone who "
    "is not the patient, or is only asked about."
)

EMERGENCY_SIGNS = {
    # Asks only whether the deficits are there. Whether they are old news is
    # a second question (SIGN_EXCLUSIONS): Jev reads literally, and with
    # "sudden" or an exception clause in this one it scored "had a stroke
    # last year and now cannot move her arm" as not an emergency.
    "stroke_signs": (
        "Does the text report that the patient has, or may be having, a "
        "stroke: face drooping, weakness, numbness or paralysis on one side "
        "of the body, an arm or leg they cannot move, slurred or lost "
        "speech, or loss of vision?"
    ),
    "cardiac_signs": (
        "Does the text report that the patient has, or may be having, a "
        "heart attack or cardiac arrest now: chest pain, pressure or "
        "tightness that is severe, crushing, squeezing, spreading to the "
        "arm, jaw or back, or that comes with sweating, nausea or "
        "breathlessness?"
    ),
    "breathing_emergency": (
        "Does the text report that the patient is now struggling to breathe: "
        "unable to speak in full sentences, gasping, choking, turning blue, "
        "or a child working hard to breathe?"
    ),
    "severe_allergic_reaction": (
        "Does the text report that the patient now has swelling of the face, "
        "eyes, lips, tongue or throat, or a closing throat, after food, a "
        "sting or a medicine?"
    ),
    "serious_bleeding": (
        "Does the text report that the patient has bleeding that is heavy or "
        "has not stopped, is vomiting or coughing up blood, or has black "
        "tarry stools?"
    ),
    "unresponsive": (
        "Does the text report that the patient has collapsed, cannot be "
        "woken, is unconscious, limp or floppy, or is not responding or not "
        "breathing normally?"
    ),
    "seizure": (
        "Does the text report that the patient is having a seizure or "
        "convulsion now, or had one within the last day?"
    ),
    "self_harm_or_poisoning": (
        "Does the text report that the patient plans or wants to end their "
        "life or harm themselves, or has taken an overdose or swallowed "
        "something poisonous?"
    ),
    "head_emergency": (
        "Does the text report that the patient now has a sudden, extremely "
        "severe headache, or a head injury followed by vomiting, drowsiness "
        "or confusion?"
    ),
    "serious_infection": (
        "Does the text report signs of meningitis or sepsis in the patient "
        "now, such as fever with a stiff neck, a rash that does not fade, "
        "confusion or a very fast heart rate, or a fever in a baby under "
        "three months old?"
    ),
    "other_emergency": (
        "Does the text report any other problem in the patient that needs "
        "emergency medical care right now, such as sudden severe abdominal "
        "pain with a rigid abdomen, or a pregnant patient with a severe "
        "headache and visual disturbance?"
    ),
}

# A second literal question for a sign whose wording can also match something
# that is not an emergency. The sign is dropped only when Jev is confident of
# the exclusion (EXCLUSION_MIN_PROBABILITY); when it is unsure, the sign
# stands.
SIGN_EXCLUSIONS = {
    "stroke_signs": {
        "question": (
            "Does the text say that the patient's weakness, numbness, "
            "paralysis, drooping, speech or vision problem has been there "
            "for a long time and has not changed?"
        ),
        "true": (
            "The text describes the problem as long-standing, stable or "
            "unchanged, for example since an earlier stroke, since birth or "
            "for years."
        ),
        "false": (
            "The text does not say that, or describes the problem as new, "
            "sudden, returning or getting worse, or mentions no such problem."
        ),
    },
}
EXCLUSION_MIN_PROBABILITY = 0.8

_CACHE_SIZE = 256

# After a failed call Jev is skipped for this long. Without it every request
# during an outage would wait for the timeout on each of its Jev calls before
# falling back to the rules.
_FAILURE_COOLDOWN_SECONDS = 30.0


@dataclass(frozen=True)
class MentionJudgment:
    """Jev's reading of one mention."""
    context: str                      # the most likely MENTION_CONTEXT_OPTIONS key
    probabilities: dict[str, float]   # probability of every option
    confidence: float                 # how concentrated the distribution is

    @property
    def present(self) -> float:
        """Probability that the patient has it now."""
        return self.probabilities.get(PRESENT, 0.0)


class JevJudge:
    """Asks Jev narrow, typed questions about a patient's text."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        enabled: bool | None = None,
    ):
        self._api_key = (
            api_key if api_key is not None
            else settings.typesafe_api_key.get_secret_value()
        )
        self.model = model or settings.jev_model
        self._timeout = timeout or settings.jev_timeout_seconds
        self._enabled = settings.jev_enabled if enabled is None else enabled
        self._client = None
        self._client_lock = threading.Lock()
        # Identical requests within a diagnosis (and retries of the same
        # text) are answered from memory.
        self._cache: OrderedDict[str, dict | None] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._skip_until = 0.0  # time.monotonic() until which calls are skipped
        if self._enabled and self._api_key and not HAS_TYPESAFE:
            nlp_logger.warning(
                "TYPESAFE_API_KEY is set but the typesafe-sdk package is not "
                "installed: Jev is disabled and the rule-based logic is used"
            )
        nlp_logger.info(
            f"JevJudge configured (model={self.model}, available={self.available})"
        )

    @property
    def available(self) -> bool:
        """True when Jev can be called at all."""
        return bool(self._enabled and self._api_key and HAS_TYPESAFE)

    # ------------------------------------------------------------------
    # Judgments
    # ------------------------------------------------------------------

    def judge_mentions(
        self, text: str, mentions: Sequence[str]
    ) -> dict[str, MentionJudgment] | None:
        """What the text says about each mention, in one request.

        Returns:
            {mention: MentionJudgment}, or None when Jev is unavailable.
        """
        mentions = list(dict.fromkeys(mentions))
        if not text or not mentions:
            return {}

        questions = {
            f"m{i}": {
                "type": "choice",
                "instructions": {
                    "question": f"What does `text` say about `mentions[{i}]`?",
                    "patient": _PATIENT_DEFINITION,
                    "if_mentioned_more_than_once": (
                        "Answer for the mention that describes the patient's "
                        "current state."
                    ),
                },
                "criteria": MENTION_CONTEXT_OPTIONS,
            }
            for i in range(len(mentions))
        }
        answers = self._ask({"text": text, "mentions": mentions}, questions)
        if answers is None:
            return None

        judgments = {}
        for i, mention in enumerate(mentions):
            answer = answers.get(f"m{i}")
            if not answer or "choice" not in answer:
                return None
            judgments[mention] = MentionJudgment(
                context=answer["choice"],
                probabilities=answer["probabilities"],
                confidence=answer["confidence"],
            )
        return judgments

    def emergency_signals(
        self, symptoms: str, question: str = ""
    ) -> dict[str, float] | None:
        """Probability of each emergency warning sign in the patient's text.

        Returns:
            {sign: probability}, or None when Jev is unavailable.
        """
        if not (symptoms or question):
            return {}

        questions = {
            sign: {
                "type": "noul",
                "instructions": {"question": instruction, "patient": _PATIENT_DEFINITION},
                "criteria": {
                    "true": _IS_AN_EMERGENCY,
                    "false": _NOT_AN_EMERGENCY,
                },
            }
            for sign, instruction in EMERGENCY_SIGNS.items()
        }
        for sign, exclusion in SIGN_EXCLUSIONS.items():
            questions[f"{sign}_excluded"] = {
                "type": "noul",
                "instructions": {
                    "question": exclusion["question"],
                    "patient": _PATIENT_DEFINITION,
                },
                "criteria": {"true": exclusion["true"], "false": exclusion["false"]},
            }
        state = {"symptoms": symptoms or "", "clinical_question": question or ""}
        answers = self._ask(state, questions)
        if answers is None:
            return None

        signals = {}
        for sign in EMERGENCY_SIGNS:
            answer = answers.get(sign)
            if not answer or "noul" not in answer:
                return None
            signals[sign] = answer["noul"]
        for sign in SIGN_EXCLUSIONS:
            excluded = answers.get(f"{sign}_excluded", {}).get("noul", 0.0)
            if excluded >= EXCLUSION_MIN_PROBABILITY:
                signals[sign] = 0.0
        return signals

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _ask(self, state: dict, questions: dict[str, dict]) -> dict[str, dict] | None:
        """Cached request; None on any failure."""
        if not self.available:
            return None

        key = repr((self.model, state, questions))
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        if time.monotonic() < self._skip_until:
            return None

        try:
            answers = self._request(state, questions)
        except Exception as e:
            # Jev is optional: whatever went wrong (network, SDK, an
            # unexpected response) the diagnosis continues on the rules.
            # Not cached, and Jev is skipped for a while so that an outage
            # costs one timeout instead of one per call.
            self._skip_until = time.monotonic() + _FAILURE_COOLDOWN_SECONDS
            nlp_logger.warning(
                f"Jev request failed ({describe_error(e)}); using the "
                f"rule-based logic for the next {_FAILURE_COOLDOWN_SECONDS:.0f}s"
            )
            return None

        with self._cache_lock:
            self._cache[key] = answers
            while len(self._cache) > _CACHE_SIZE:
                self._cache.popitem(last=False)
        return answers

    def _request(self, state: dict, questions: dict[str, dict]) -> dict[str, dict]:
        """One Jev call. Returns {question id: plain answer dict}."""
        response = self._get_client().system_one(state=state, questions=questions)
        return _plain_answers(response)

    def _get_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = TypeSafeClient(
                        api_key=self._api_key,
                        model=self.model,
                        timeout=self._timeout,
                        # The diagnosis request is waiting: no retries, a
                        # failure falls straight back to the rules.
                        retry=RetryPolicy(max_retries=0),
                    )
        return self._client


def _plain_answers(response: Any) -> dict[str, dict]:
    """Reduce an SDK response to JSON-serializable answer dicts."""
    answers = {}
    for qid, answer in response.answers.items():
        if hasattr(answer, "noul"):
            answers[qid] = {"noul": float(answer.noul)}
        elif hasattr(answer, "choice"):
            answers[qid] = {
                "choice": answer.choice,
                "probabilities": {k: float(v) for k, v in answer.probabilities.items()},
                "confidence": float(answer.confidence),
            }
    return answers
