"""Patient text must not reach the logs."""

import logging

import pytest
from pydantic import BaseModel, ValidationError

from app.core.nlp.clinical_fallback import ClinicalFallbackEngine
from app.core.nlp.question_understanding import QuestionUnderstanding
from app.core.nlp.rag_engine import RAGEngine
from app.core.nlp.text_preprocessor import TextPreprocessor
from app.core.safety.safety_controller import SafetyController
from app.utils.logger import describe_error

# A name and symptoms that appear nowhere but in the patient's own text
SYMPTOMS = "My name is Zebediah Quillfeather. I have had a fever and a bad cough for 3 days"
QUESTION = "Could Zebediah Quillfeather have pneumonia in his left lung?"
PRIVATE = ("zebediah", "quillfeather", "fever", "cough", "lung")


@pytest.fixture
def logged():
    """Everything the application loggers emit during the test, lower-cased."""
    records: list[str] = []

    class Collector(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage().lower())

    handler = Collector(level=logging.DEBUG)
    loggers = [logging.getLogger(f"cdss.{name}") for name in ("api", "nlp", "safety", "fusion")]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    yield records
    for logger, level in zip(loggers, levels):
        logger.removeHandler(handler)
        logger.setLevel(level)


def assert_nothing_private(records):
    assert records, "the code under test logged nothing"
    for message in records:
        for word in PRIVATE:
            assert word not in message, message


def test_preprocessing_and_question_analysis_log_no_patient_text(logged):
    result = TextPreprocessor().preprocess(SYMPTOMS)
    QuestionUnderstanding().analyze(QUESTION)

    assert "fever" in result["extracted_symptoms"]  # the text was really processed
    assert_nothing_private(logged)


def test_fallback_diagnosis_logs_no_patient_text(logged):
    ClinicalFallbackEngine().diagnose(SYMPTOMS, [])
    # The diagnosis label is the system's own output and may be logged
    assert_nothing_private([m for m in logged if not m.startswith("fallback diagnosis:")])


def test_safety_gate_logs_no_patient_text(logged):
    SafetyController().evaluate(
        {"primary_diagnosis": "Common cold", "combined_confidence": 0.9, "modality": "nlp",
         "is_guideline_consistent": True},
        symptoms_text=SYMPTOMS, clinical_question=QUESTION,
    )
    assert_nothing_private(logged)


def test_unparseable_llm_response_is_not_logged(logged):
    engine = RAGEngine.__new__(RAGEngine)  # only the parser is exercised
    engine._parse_llm_response("Zebediah Quillfeather has a fever and a cough {broken")

    assert_nothing_private(logged)


def test_validation_errors_are_reduced_to_their_type():
    class Request(BaseModel):
        symptoms: int

    with pytest.raises(ValidationError) as caught:
        Request(symptoms=SYMPTOMS)

    assert "Zebediah" in str(caught.value)  # what a plain log line would have held
    assert describe_error(caught.value) == "ValidationError"
    assert describe_error(KeyError(SYMPTOMS)) == "KeyError"


def test_api_and_network_errors_keep_their_message():
    assert describe_error(TimeoutError("timed out")) == "TimeoutError: timed out"
    assert describe_error(ConnectionError("refused")) == "ConnectionError: refused"
