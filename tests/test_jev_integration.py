"""Jev integration, exercised with a fake judge: no network, no API key."""

import pytest

import app.core.nlp.jev_judge as jev_module
from app.core.nlp.jev_judge import (
    EMERGENCY_SIGNS, MENTION_CONTEXT_OPTIONS, JevJudge,
)
from app.core.nlp.text_preprocessor import TextPreprocessor
from app.core.safety.safety_controller import SafetyController

APPROVABLE = {
    "primary_diagnosis": "Common cold",
    "combined_confidence": 0.9,
    "modality": "nlp",
    "severity": "low",
    "urgency": "routine",
    "is_guideline_consistent": True,
    "explanation": "",
}


class FakeJudge(JevJudge):
    """Answers from canned values instead of calling the API."""

    def __init__(self, contexts=None, signals=None, exclusions=None, fail=False):
        super().__init__(api_key="test-key", enabled=True)
        self.contexts = contexts or {}      # mention -> (context, probability of "present")
        self.signals = signals or {}        # sign -> probability
        self.exclusions = exclusions or {}  # sign -> probability the exclusion holds
        self.fail = fail
        self.requests = 0

    @property
    def available(self) -> bool:
        return True

    def _request(self, state, questions):
        self.requests += 1
        if self.fail:
            raise ConnectionError("Jev is unreachable")
        answers = {}
        for qid, question in questions.items():
            if question["type"] == "choice":
                mention = state["mentions"][int(qid[1:])]
                context, present = self.contexts.get(mention, ("present", 0.98))
                probabilities = dict.fromkeys(MENTION_CONTEXT_OPTIONS, 0.0)
                probabilities["present"] = present
                if context != "present":
                    probabilities[context] = 1.0 - present
                answers[qid] = {
                    "choice": context,
                    "probabilities": probabilities,
                    "confidence": max(probabilities.values()),
                }
            elif qid.endswith("_excluded"):
                answers[qid] = {"noul": self.exclusions.get(qid.removesuffix("_excluded"), 0.02)}
            else:
                answers[qid] = {"noul": self.signals.get(qid, 0.03)}
        return answers


# ----------------------------------------------------------------------
# Symptom extraction
# ----------------------------------------------------------------------

def test_preprocessor_sorts_mentions_by_jev_context():
    judge = FakeJudge(contexts={
        "fever": ("denied", 0.01),
        "stroke": ("other_person", 0.0),
        "pneumonia": ("past", 0.02),
        "cancer": ("hypothetical", 0.0),
    })
    result = TextPreprocessor(context_judge=judge).preprocess(
        "cough, no fever, my friend had a stroke, I had pneumonia once, could it be cancer?"
    )

    assert result["context_source"] == "jev"
    assert result["extracted_symptoms"] == ["cough"]
    assert result["negated_symptoms"] == ["fever"]
    assert result["family_history"] == ["stroke"]
    assert result["mention_contexts"] == {
        "cough": "present", "fever": "denied", "stroke": "other_person",
        "pneumonia": "past", "cancer": "hypothetical",
    }


def test_preprocessor_falls_back_to_rules_when_jev_fails():
    text = "no fever, my mother had a stroke last year, I have a headache"
    with_broken_jev = TextPreprocessor(context_judge=FakeJudge(fail=True)).preprocess(text)
    rules_only = TextPreprocessor().preprocess(text)

    assert with_broken_jev["context_source"] == "rules"
    for key in ("extracted_symptoms", "negated_symptoms", "family_history"):
        assert with_broken_jev[key] == rules_only[key]


def test_same_text_is_asked_once():
    judge = FakeJudge()
    preprocessor = TextPreprocessor(context_judge=judge)
    preprocessor.preprocess("cough and fever")
    preprocessor.preprocess("cough and fever")
    assert judge.requests == 1


def test_no_request_when_text_has_no_mentions():
    judge = FakeJudge()
    result = TextPreprocessor(context_judge=judge).preprocess("I feel a bit off today")
    assert judge.requests == 0
    assert result["extracted_symptoms"] == []


# ----------------------------------------------------------------------
# Failure handling
# ----------------------------------------------------------------------

def test_jev_is_skipped_after_a_failure(monkeypatch):
    """An outage must cost one failed call, not one per request."""
    judge = FakeJudge(fail=True)
    assert judge.emergency_signals("chest pain") is None
    assert judge.emergency_signals("a different text") is None
    assert judge.judge_mentions("cough", ["cough"]) is None
    assert judge.requests == 1

    # ...and it is tried again once the cooldown has passed
    monkeypatch.setattr(jev_module, "_FAILURE_COOLDOWN_SECONDS", 0.0)
    judge._skip_until = 0.0
    judge.fail = False
    assert judge.emergency_signals("chest pain") is not None
    assert judge.requests == 2


def test_unexpected_error_does_not_reach_the_caller():
    class ExplodingJudge(FakeJudge):
        def _request(self, state, questions):
            raise KeyError("unexpected response shape")

    flags = SafetyController(context_judge=ExplodingJudge()).patient_text_flags(
        "I am having a seizure", ""
    )
    assert flags.keywords == ["seizure"]
    assert flags.signals is None


def test_judge_without_api_key_is_unavailable():
    judge = JevJudge(api_key="", enabled=True)
    assert judge.available is False
    assert judge.judge_mentions("cough", ["cough"]) is None
    assert judge.emergency_signals("chest pain") is None


# ----------------------------------------------------------------------
# Red-flag gate
# ----------------------------------------------------------------------

def test_emergency_sign_is_a_red_flag_without_any_keyword():
    controller = SafetyController(context_judge=FakeJudge(signals={"stroke_signs": 0.95}))
    result = controller.evaluate(
        dict(APPROVABLE), symptoms_text="the left side of my face is drooping"
    )

    assert result["safety_status"] == "fallback"
    assert "Emergency warning signs detected: stroke signs" in result["risk_factors"]
    assert result["emergency_signals"]["stroke_signs"] == 0.95


def test_weak_sign_does_not_flag():
    controller = SafetyController(context_judge=FakeJudge(signals={"stroke_signs": 0.2}))
    result = controller.evaluate(dict(APPROVABLE), symptoms_text="my arm felt numb for a second")
    assert result["safety_status"] == "approved"


def test_keyword_read_as_history_is_dropped():
    judge = FakeJudge(contexts={"stroke": ("past", 0.0)})
    result = SafetyController(context_judge=judge).evaluate(
        dict(APPROVABLE),
        symptoms_text="I had a stroke five years ago and recovered. I have a runny nose",
    )

    assert result["safety_status"] == "approved"
    assert result["red_flags_cleared"] == ["stroke"]


@pytest.mark.parametrize("contexts, signals, clear_history_flags", [
    # Jev is not certain enough that it is history
    ({"stroke": ("past", 0.3)}, {}, True),
    # A question about a condition is never dropped
    ({"stroke": ("hypothetical", 0.0)}, {}, True),
    # An emergency sign fired: every keyword stands
    ({"stroke": ("past", 0.0)}, {"unresponsive": 0.9}, True),
    # Clearing switched off
    ({"stroke": ("past", 0.0)}, {}, False),
])
def test_keyword_flag_stands(contexts, signals, clear_history_flags):
    controller = SafetyController(
        context_judge=FakeJudge(contexts=contexts, signals=signals),
        clear_history_flags=clear_history_flags,
    )
    flags = controller.patient_text_flags("I had a stroke", "")
    assert "stroke" in flags.keywords
    assert flags.cleared == []


def test_generic_alarm_word_is_never_dropped():
    judge = FakeJudge(contexts={"severe": ("past", 0.0)})
    flags = SafetyController(context_judge=judge).patient_text_flags("severe sore throat", "")
    assert flags.keywords == ["severe"]


def test_rules_stand_when_jev_is_down():
    flags = SafetyController(context_judge=FakeJudge(fail=True)).patient_text_flags(
        "I had a stroke five years ago. I have a runny nose", ""
    )
    assert flags.keywords == ["stroke"]
    assert flags.emergency_signs == {}


def test_confident_exclusion_drops_the_stroke_sign():
    """Deficits the text calls long-standing and unchanged are not a new stroke."""
    chronic = FakeJudge(signals={"stroke_signs": 0.9}, exclusions={"stroke_signs": 0.95})
    unsure = FakeJudge(signals={"stroke_signs": 0.9}, exclusions={"stroke_signs": 0.6})

    assert chronic.emergency_signals("weak arm since my stroke, unchanged")["stroke_signs"] == 0.0
    assert unsure.emergency_signals("had a stroke, now cannot move my arm")["stroke_signs"] == 0.9


def test_every_sign_is_reported():
    signals = FakeJudge().emergency_signals("cough")
    assert set(signals) == set(EMERGENCY_SIGNS)


def test_vlm_approval_cannot_clear_an_emergency_sign():
    controller = SafetyController()
    risk = "Emergency warning signs detected: stroke signs"
    risk_factors, _, is_safe = controller._apply_vlm_review(
        {"decision": "approve"}, {}, [risk], [], False
    )
    assert risk_factors == [risk]
    assert is_safe is False
