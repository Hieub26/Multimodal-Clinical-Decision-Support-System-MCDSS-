"""How the diagnosis pipelines order their independent steps."""

import threading

import pytest

import app.api.endpoints.cv_router as cv_module
import app.api.endpoints.nlp_router as nlp_module
from app.core.cv.cv_model import ModelWeightsNotFoundError
from app.core.safety.safety_controller import PatientTextFlags, SafetyController

FLAGS = PatientTextFlags(keywords=["stroke"])
WAIT = 5  # seconds a step waits for the other one before giving up


class FakeSafety:
    def __init__(self):
        self.flag_calls = []
        self.evaluated_with = None

    def patient_text_flags(self, symptoms_text, clinical_question):
        self.flag_calls.append((symptoms_text, clinical_question))
        return FLAGS

    def evaluate(self, validated, **kwargs):
        self.evaluated_with = kwargs
        return {**validated, "safety_status": "approved"}


class FakeRAG:
    def __init__(self, started=None, wait_for=None):
        self.calls = 0
        self._started, self._wait_for = started, wait_for

    def diagnose_from_text(self, symptoms_text, clinical_question):
        self.calls += 1
        if self._started:
            self._started.set()
            assert self._wait_for.wait(WAIT), "the image was not analysed alongside the text"
        return {"primary_diagnosis": "Pneumonia", "confidence": 0.8}


class FakeCV:
    def __init__(self, started=None, wait_for=None, weights=True):
        self._started, self._wait_for, self._weights = started, wait_for, weights

    def ensure_loaded(self):
        if not self._weights:
            raise ModelWeightsNotFoundError("no weights")

    def predict(self, image_path):
        if self._started:
            self._started.set()
            assert self._wait_for.wait(WAIT), "the text was not analysed alongside the image"
        return {"predicted_class": "Infiltration", "confidence": 0.6}


class FakeFusion:
    def fuse(self, nlp_diagnosis=None, cv_diagnosis=None):
        return {"nlp": nlp_diagnosis, "cv": cv_diagnosis}


class FakeValidator:
    def validate(self, fused):
        return fused


class FakeReports:
    def generate_clinical_report(self, *args):
        return "report.md"


@pytest.fixture
def stages(monkeypatch):
    """Both routers wired to fakes; returns the safety controller and a
    function that installs the RAG engine and the CV model of a test."""
    safety = FakeSafety()
    for module in (cv_module, nlp_module):
        monkeypatch.setattr(module, "get_safety_controller", lambda: safety)
        monkeypatch.setattr(module, "get_fusion_engine", FakeFusion)
        monkeypatch.setattr(module, "get_guideline_validator", FakeValidator)
        monkeypatch.setattr(module, "get_report_generator", FakeReports)

    def install(rag, cv=None):
        monkeypatch.setattr(cv_module, "get_rag_engine", lambda: rag)
        monkeypatch.setattr(nlp_module, "get_rag_engine", lambda: rag)
        monkeypatch.setattr(cv_module, "get_cv_model", lambda: cv)

    return safety, install


def test_text_pipeline_reads_the_red_flags_once_and_hands_them_to_the_safety_check(stages):
    safety, install = stages
    install(FakeRAG())

    safe_result, nlp_result, report_path = nlp_module._run_text_pipeline("had a stroke", None)

    assert safety.flag_calls == [("had a stroke", None)]
    assert safety.evaluated_with["patient_flags"] is FLAGS
    assert nlp_result["primary_diagnosis"] == "Pneumonia"
    assert safe_result["safety_status"] == "approved"
    assert report_path == "report.md"


def test_image_and_text_are_analysed_side_by_side(stages):
    """Each step waits for the other to have started: run one after the
    other, the first would time out."""
    safety, install = stages
    text_started, image_started = threading.Event(), threading.Event()
    install(
        FakeRAG(started=text_started, wait_for=image_started),
        FakeCV(started=image_started, wait_for=text_started),
    )

    safe_result, nlp_result, cv_result, modality, _ = cv_module._run_image_pipeline(
        "scan.png", "cough", None
    )

    assert modality == "multimodal"
    assert safe_result["nlp"] is nlp_result and safe_result["cv"] is cv_result
    assert safety.evaluated_with["patient_flags"] is FLAGS
    assert safety.evaluated_with["image_path"] == "scan.png"


def test_image_only_request_does_not_run_the_text_analysis(stages):
    _, install = stages
    rag = FakeRAG()
    install(rag, FakeCV())

    _, nlp_result, _, modality, _ = cv_module._run_image_pipeline("scan.png", None, None)

    assert (modality, nlp_result, rag.calls) == ("image", None, 0)


def test_missing_weights_stop_the_request_before_the_text_analysis_starts(stages):
    """Otherwise every image request would spend an LLM call on a case that
    cannot be completed."""
    safety, install = stages
    rag = FakeRAG()
    install(rag, FakeCV(weights=False))

    with pytest.raises(ModelWeightsNotFoundError):
        cv_module._run_image_pipeline("scan.png", "cough", None)

    assert rag.calls == 0
    assert safety.flag_calls == []


def test_safety_check_uses_the_flags_it_is_given(monkeypatch):
    controller = SafetyController()

    def not_again(*args):
        raise AssertionError("the patient's text was read a second time")

    monkeypatch.setattr(controller, "patient_text_flags", not_again)

    result = controller.evaluate(
        {"primary_diagnosis": "Common cold", "combined_confidence": 0.9, "modality": "nlp"},
        symptoms_text="mild cough",
        patient_flags=FLAGS,
    )

    assert result["is_approved"] is False
    assert any("stroke" in risk for risk in result["risk_factors"])
