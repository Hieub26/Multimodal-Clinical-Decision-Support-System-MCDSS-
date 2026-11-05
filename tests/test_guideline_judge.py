"""Guideline judge and validator, exercised with fake judges: no network, no API key."""

import pytest

import app.core.validation.guideline_judge as judge_module
from app.core.validation.guideline_judge import (
    GeminiJudge, GuidelineJudge, JudgeVerdict, OpenAIJudge, gemini_judge_prompt,
    judge_input, parse_prompted_verdict,
)
from app.core.validation.guideline_validator import (
    METHOD_JUDGE, METHOD_NO_GUIDELINES, METHOD_SKIPPED, METHOD_TOKENS,
    GuidelineValidator, default_judge,
)
from app.config import settings

DOCS = [
    {"text": "PNEUMONIA\nDefinition: infection that inflames the air sacs.", "distance": 0.35},
    {"text": "PNEUMOTHORAX\nDefinition: air in the pleural cavity.", "distance": 0.45},
]
TEXT_DIAGNOSIS = {"primary_diagnosis": "Pneumonia", "combined_confidence": 0.8, "modality": "nlp"}


class FakeJudge(GuidelineJudge):
    """Returns a canned verdict instead of calling a model."""

    provider = "fake"

    def __init__(self, score=0.9, negation=False, synonyms=(), fail=False, available=True):
        super().__init__("fake-judge-1")
        self.score = score
        self.negation = negation
        self.synonyms = list(synonyms)
        self.fail = fail
        self._available = available
        self.requests = []

    @property
    def available(self) -> bool:
        return self._available

    def _request(self, diagnosis, passages):
        self.requests.append((diagnosis, passages))
        if self.fail:
            raise ConnectionError("the judge is unreachable")
        verdict = JudgeVerdict(
            reasoning="Passage 1 is about this condition.",
            support_score=self.score,
            negation_detected=self.negation,
            synonym_matches=self.synonyms,
        )
        return verdict, {"input_tokens": 100, "output_tokens": 20}


class FakeStore:
    def __init__(self, docs=DOCS):
        self.docs = docs
        self.queries = []

    def search(self, query, k=5):
        self.queries.append((query, k))
        return self.docs[:k]


def validator_with(judge, docs=DOCS):
    return GuidelineValidator(vector_store=FakeStore(docs), judge=judge)


# ----------------------------------------------------------------------
# Validator
# ----------------------------------------------------------------------

def test_judge_verdict_decides_consistency():
    judge = FakeJudge(score=0.85)
    result = validator_with(judge).validate(dict(TEXT_DIAGNOSIS))

    assert result["is_guideline_consistent"] is True
    assert result["validation_status"] == "validated"
    assert result["guideline_support_score"] == 0.85
    assert result["validation_method"] == METHOD_JUDGE
    assert result["judge_model"] == "fake-judge-1"
    assert result["validation_notes"][0] == (
        "Guideline judge (fake-judge-1): Passage 1 is about this condition."
    )


def test_judge_sees_the_diagnosis_and_passages_only():
    judge = FakeJudge()
    diagnosis = dict(TEXT_DIAGNOSIS, explanation="Patient John reports fever and cough")
    validator_with(judge).validate(diagnosis)

    assert judge.requests == [("Pneumonia", [d["text"] for d in DOCS])]


@pytest.mark.parametrize("score, consistent", [
    (0.39, False),
    (0.40, True),
    (0.0, False),
    (1.0, True),
])
def test_consistency_is_derived_from_the_score(score, consistent):
    result = validator_with(FakeJudge(score=score)).validate(dict(TEXT_DIAGNOSIS))
    assert result["is_guideline_consistent"] is consistent
    assert result["validation_status"] == ("validated" if consistent else "needs_review")


def test_out_of_range_score_is_clamped():
    assert FakeJudge(score=1.7).judge("Pneumonia", ["text"])["support_score"] == 1.0
    assert FakeJudge(score=-0.2).judge("Pneumonia", ["text"])["support_score"] == 0.0


def test_negation_and_synonyms_are_reported():
    judge = FakeJudge(score=0.1, negation=True, synonyms=["MI = myocardial infarction"])
    notes = validator_with(judge).validate(dict(TEXT_DIAGNOSIS))["validation_notes"]

    assert any("Negation detected" in note for note in notes)
    assert "Synonym matches found: MI = myocardial infarction" in notes


def test_failed_judge_falls_back_to_token_matching():
    judge = FakeJudge(fail=True)
    result = validator_with(judge).validate(dict(TEXT_DIAGNOSIS))

    assert result["validation_method"] == METHOD_TOKENS
    assert result["judge_model"] is None
    assert result["validation_notes"][0].startswith("Validation method: token matching")
    # "pneumonia" is in the first passage
    assert result["guideline_support_score"] == 1.0
    assert result["is_guideline_consistent"] is True


# ----------------------------------------------------------------------
# Token matching (fallback)
# ----------------------------------------------------------------------

HEART_DOCS = [
    {"text": "HEART FAILURE\nTreatment: beta-blockers and diuretics.", "distance": 0.4},
    {"text": "HYPERTENSION (HIGH BLOOD PRESSURE)\nDefined as SBP >= 130 mmHg.", "distance": 0.5},
    {"text": "PULMONARY EDEMA\nFluid in the lungs. Tuberculosis (TB) is unrelated.", "distance": 0.6},
]


def match_tokens(diagnosis, docs=HEART_DOCS):
    consistent, support, notes = GuidelineValidator(
        vector_store=object(), judge=FakeJudge()
    )._validate_with_tokens(diagnosis, docs)
    return consistent, support, notes


@pytest.mark.parametrize("diagnosis", [
    "Heart Failure",
    "Acute heart failure exacerbation",   # generic qualifiers are not required
    "Hypertension",
    "High blood pressure",
    "TB",                                 # two-letter acronym, as a whole word
    "Pulmonary_Edema",                    # classifier label
    "Heart failure (possible HFrEF)",     # a parenthesis is a qualifier
])
def test_tokens_pass_a_diagnosis_one_passage_names(diagnosis):
    assert match_tokens(diagnosis)[0] is True


@pytest.mark.parametrize("diagnosis", [
    "Heart block",              # "block" is only inside "beta-blockers"
    "Pulmonary hypertension",   # one word in each of two passages
    "Migraine",
    "Analysis completed",
])
def test_tokens_reject_a_diagnosis_no_passage_names(diagnosis):
    consistent, support, _ = match_tokens(diagnosis)
    assert consistent is False
    assert support == 0.0


def test_tokens_do_not_pass_a_stated_measurement():
    consistent, _, notes = match_tokens("Hypertension (blood pressure 110/70 mmHg)")

    assert consistent is False
    assert any("measurement" in note for note in notes)


def test_unavailable_judge_is_not_called():
    judge = FakeJudge(available=False)
    result = validator_with(judge).validate(dict(TEXT_DIAGNOSIS))

    assert judge.requests == []
    assert result["validation_method"] == METHOD_TOKENS


def test_cv_only_diagnosis_skips_the_judge():
    judge = FakeJudge()
    result = validator_with(judge).validate(
        {"primary_diagnosis": "Effusion", "combined_confidence": 0.6, "modality": "cv"}
    )

    assert judge.requests == []
    assert result["validation_method"] == METHOD_SKIPPED
    assert result["is_guideline_consistent"] is True


def test_no_retrieved_guideline_means_not_consistent():
    judge = FakeJudge()
    result = validator_with(judge, docs=[]).validate(dict(TEXT_DIAGNOSIS))

    assert judge.requests == []
    assert result["validation_method"] == METHOD_NO_GUIDELINES
    assert result["is_guideline_consistent"] is False


def test_modality_conflict_overrides_a_supportive_judge():
    result = validator_with(FakeJudge(score=0.95)).validate(
        dict(TEXT_DIAGNOSIS, modality="multimodal", conflict_flag=True)
    )
    assert result["is_guideline_consistent"] is False
    assert result["guideline_support_score"] == 0.95


# ----------------------------------------------------------------------
# Judge: caching and failure handling
# ----------------------------------------------------------------------

def test_same_diagnosis_is_judged_once():
    judge = FakeJudge()
    first = judge.judge("Pneumonia", ["passage"])
    second = judge.judge("Pneumonia", ["passage"])

    assert first == second
    assert len(judge.requests) == 1
    judge.judge("Pneumonia", ["another passage"])
    assert len(judge.requests) == 2


def test_judge_is_skipped_after_a_failure(monkeypatch):
    """An outage must cost one failed call, not one per diagnosis."""
    judge = FakeJudge(fail=True)
    assert judge.judge("Pneumonia", ["passage"]) is None
    assert judge.judge("Asthma", ["passage"]) is None
    assert len(judge.requests) == 1

    # ...and it is tried again once the cooldown has passed
    monkeypatch.setattr(judge_module, "_FAILURE_COOLDOWN_SECONDS", 0.0)
    judge._skip_until = 0.0
    judge.fail = False
    assert judge.judge("Asthma", ["passage"])["support_score"] == 0.9
    assert len(judge.requests) == 2


def test_failed_call_is_not_cached():
    judge = FakeJudge(fail=True)
    assert judge.judge("Pneumonia", ["passage"]) is None
    judge.fail = False
    judge._skip_until = 0.0
    assert judge.judge("Pneumonia", ["passage"]) is not None


# ----------------------------------------------------------------------
# Choosing the judge
# ----------------------------------------------------------------------

def test_openai_key_selects_the_openai_judge(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", type(settings.openai_api_key)("sk-test"))
    judge = default_judge()

    assert isinstance(judge, OpenAIJudge)
    assert judge.model == settings.judge_model_name


def test_without_openai_key_the_generator_model_judges():
    judge = default_judge()

    assert isinstance(judge, GeminiJudge)
    assert judge.model == settings.llm_model_name
    assert judge.available is False  # the test environment has no Gemini key either


def test_judges_without_a_key_are_unavailable():
    assert OpenAIJudge(api_key="").available is False
    assert OpenAIJudge(api_key="").judge("Pneumonia", ["passage"]) is None
    assert GeminiJudge(api_key="").judge("Pneumonia", ["passage"]) is None


# ----------------------------------------------------------------------
# OpenAI judge request
# ----------------------------------------------------------------------

class FakeOpenAIResponses:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []

    def parse(self, **request):
        self.calls.append(request)

        class Usage:
            input_tokens, output_tokens = 1200, 90

        class Response:
            output_parsed = self.parsed
            usage = Usage()

        return Response()


class FakeOpenAIClient:
    def __init__(self, parsed):
        self.responses = FakeOpenAIResponses(parsed)


def openai_judge(parsed, **kwargs):
    judge = OpenAIJudge(api_key="sk-test", model="judge-model", **kwargs)
    judge._client = FakeOpenAIClient(parsed)
    return judge


def test_openai_request_asks_for_a_structured_verdict():
    verdict = JudgeVerdict(reasoning="Covered by passage 1.", support_score=0.9,
                           negation_detected=False, synonym_matches=[])
    judge = openai_judge(verdict, reasoning_effort="low")

    result = judge.judge("Pneumonia", ["first passage", "second passage"])

    assert result["support_score"] == 0.9
    request = judge._client.responses.calls[0]
    assert request["model"] == "judge-model"
    assert request["text_format"] is JudgeVerdict
    assert request["reasoning"] == {"effort": "low"}
    assert request["store"] is False
    assert request["input"][0]["role"] == "system"
    assert request["input"][1] == {
        "role": "user",
        "content": "Diagnosis: Pneumonia\n\nGuideline passages:\n[1] first passage\n\n[2] second passage",
    }


def test_openai_request_omits_reasoning_when_no_effort_is_set():
    verdict = JudgeVerdict(reasoning="ok", support_score=0.5,
                           negation_detected=False, synonym_matches=[])
    judge = openai_judge(verdict, reasoning_effort="")
    judge.judge("Pneumonia", ["passage"])

    assert "reasoning" not in judge._client.responses.calls[0]


def test_openai_refusal_counts_as_a_failure():
    judge = openai_judge(None)
    assert judge.judge("Pneumonia", ["passage"]) is None

    validator = validator_with(openai_judge(None))
    assert validator.validate(dict(TEXT_DIAGNOSIS))["validation_method"] == METHOD_TOKENS


# ----------------------------------------------------------------------
# Gemini judge prompt and response handling
# ----------------------------------------------------------------------

def test_gemini_prompt_carries_the_diagnosis_and_numbered_guidelines():
    prompt = gemini_judge_prompt("Pneumonia", ["first passage", "second passage"])

    assert "## Diagnosis to validate:\nPneumonia" in prompt
    assert "[Guideline 1] first passage\n\n[Guideline 2] second passage" in prompt
    assert '"support_score": 0.75' in prompt  # the JSON example survives formatting


@pytest.mark.parametrize("text", [
    '{"is_consistent": true, "support_score": 0.8, "reasoning": "ok", '
    '"negation_detected": false, "synonym_matches": ["MI = heart attack"]}',
    '```json\n{"support_score": 0.8, "reasoning": "ok", "synonym_matches": ["MI = heart attack"]}\n```',
    'Here is my answer: {"support_score": 0.8, "reasoning": "ok", '
    '"synonym_matches": ["MI = heart attack"]} Hope this helps.',
])
def test_prompted_verdict_is_read_from_plain_fenced_or_embedded_json(text):
    verdict = parse_prompted_verdict(text)

    assert verdict.support_score == 0.8
    assert verdict.reasoning == "ok"
    assert verdict.negation_detected is False
    assert verdict.synonym_matches == ["MI = heart attack"]


def test_unreadable_response_counts_as_uncertain():
    verdict = parse_prompted_verdict("I cannot answer that.")

    assert verdict.support_score == 0.3
    assert "Could not parse" in verdict.reasoning


def test_judge_input_numbers_the_passages():
    assert judge_input("Asthma", ["a", "b"]) == (
        "Diagnosis: Asthma\n\nGuideline passages:\n[1] a\n\n[2] b"
    )
