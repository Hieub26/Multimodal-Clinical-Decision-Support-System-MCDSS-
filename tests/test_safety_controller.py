import pytest
from PIL import Image
from pydantic import SecretStr

import app.core.safety.safety_controller as safety_module
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


@pytest.fixture
def controller():
    return SafetyController()


@pytest.mark.parametrize("symptoms, explanation", [
    ("runny nose, no bleeding, nothing severe", ""),
    ("sore throat after 2000 keystrokes of typing", ""),
    ("runny nose", "Mild viral illness. Not life-threatening; return if symptoms become severe."),
    ("runny nose", "No signs of sepsis or meningitis."),
    # Family history the text states or dates is not the patient's emergency
    ("mild headache, my mother had a stroke last year", ""),
    ("mild headache, my mother's stroke was last year", ""),
    ("runny nose. family history of heart attack and stroke", ""),
    ("runny nose, my father died of a heart attack", ""),
    ("runny nose", "Family history of stroke slightly raises long-term risk."),
])
def test_no_false_red_flags(controller, symptoms, explanation):
    result = controller.evaluate({**APPROVABLE, "explanation": explanation}, symptoms_text=symptoms)
    assert result["risk_factors"] == []
    assert result["safety_status"] == "approved"


@pytest.mark.parametrize("kwargs, expected", [
    ({"symptoms_text": "severe chest pain"}, "severe"),
    ({"symptoms_text": "I had two seizures today"}, "seizure"),
    ({"clinical_question": "what are the risk factors for stroke?"}, "stroke"),
    # A symptom reported after a denied one is still the patient's
    ({"symptoms_text": "no fever, bleeding from the nose since morning"}, "bleeding"),
    # A relative's event that is undated, or happening now, may be an
    # emergency being reported: the gate keeps it
    ({"symptoms_text": "my mother had a stroke"}, "stroke"),
    ({"symptoms_text": "my mother had a stroke this morning"}, "stroke"),
    ({"symptoms_text": "my father is having a heart attack"}, "heart attack"),
    ({"symptoms_text": "my son had a seizure last year"}, "seizure"),
    ({"symptoms_text": "my mother had a stroke last year and she is having a seizure"}, "seizure"),
    ({"symptoms_text": "family history of stroke, my father is unconscious"}, "unconscious"),
])
def test_red_flags_in_patient_text(controller, kwargs, expected):
    result = controller.evaluate(dict(APPROVABLE), **kwargs)
    assert result["safety_status"] == "fallback"
    assert expected in result["risk_factors"][0]


# A red-flag condition is, or may be, happening to someone now. Whatever the
# wording around relatives and dates, the gate must keep these.
EMERGENCY_PHRASINGS = [
    "my mother had a stroke",
    "my mother had a stroke an hour ago",
    "my mother had a stroke and is not responding",
    "my mother is having a stroke",
    "my mother's stroke symptoms started today",
    "my mother's seizure will not stop",
    "my mothers stroke happened 10 minutes ago",
    "my mother suffered a stroke just now",
    "my dad had a seizure again",
    "my father had a heart attack last night",
    "mother had sepsis 3 days ago",
    "mother: stroke",
    "my husband had a heart attack last year",
    "i had a stroke last year",
    "my mother has been bleeding since last month",
    # History and a current event in the same text
    "my father had a heart attack last year and now he has a seizure",
    "my father had a heart attack last year, now he is unconscious",
    "last year my mother had a stroke and today she had a seizure",
    "my mother had a stroke last year. she is having another stroke",
    "my mother had a stroke last year, i think i am having a stroke",
    "my mother had a stroke last year and now a seizure",
    "my mother had a stroke last year and a seizure today",
    "my mother had a stroke last year and is bleeding",
    "my dad had a stroke 2 years ago and another stroke today",
    "my father died of a heart attack and i am having a seizure",
    "my father died of a heart attack last year and my mother is unconscious",
    "my grandfather had a stroke in 2015 and my father is unconscious",
    "my brother who had a stroke last year is bleeding heavily",
    "my sister has a history of seizures and she is having a seizure",
    "my mother has a history of stroke and is unconscious",
    "years ago my mother had sepsis and now my baby is unconscious",
    "family history of stroke. i think i am having a stroke",
    "family history of stroke, sudden severe headache",
    "family history of stroke, now unconscious",
    "family history of stroke and he is unconscious",
    "family history of stroke and now has slurred speech and a seizure",
    "family history significant for stroke. patient is having a seizure",
    "fhx of stroke. severe headache and bleeding",
]

# History the text states or dates: not the patient's emergency.
STATED_HISTORY_PHRASINGS = [
    "my mother had a stroke last year",
    "my mother had a stroke 5 years ago, i have a cough",
    "my mother's stroke was 3 years ago",
    "my grandmother had a stroke at the age of 80",
    "my mother has a history of stroke",
    "my mother, who had a stroke last year, is fine. i have a cough",
    "both my parents had heart attacks years ago",
    "my father died of a heart attack",
    "my father passed away from a heart attack in 2010",
    "my grandfather passed away from sepsis",
    "family history of heart attack and stroke",
    "family history significant for stroke and myocardial infarction",
    "fhx: stroke, heart attack",
    "stroke runs in my family",
]


@pytest.mark.parametrize("text", EMERGENCY_PHRASINGS)
def test_possible_emergency_is_always_flagged(text):
    assert SafetyController._find_red_flags(text, safety_module.RED_FLAG_KEYWORDS)


@pytest.mark.parametrize("text", STATED_HISTORY_PHRASINGS)
def test_stated_family_history_is_not_flagged(text):
    assert SafetyController._find_red_flags(text, safety_module.RED_FLAG_KEYWORDS) == []


def test_red_flag_condition_in_explanation(controller):
    result = controller.evaluate(
        {**APPROVABLE, "explanation": "Presentation is concerning for pulmonary embolism."}
    )
    assert result["safety_status"] == "fallback"
    assert "pulmonary embolism" in result["risk_factors"][0]


class _FailingVLM:
    class models:
        @staticmethod
        def generate_content(**kwargs):
            raise RuntimeError("429 quota exhausted")


def test_vlm_outage_does_not_clear_risks(controller, tmp_path, monkeypatch):
    """A failed VLM call must not turn a flagged case into an approved one."""
    monkeypatch.setattr(safety_module.settings, "gemini_api_key", SecretStr("dummy"))
    controller._client = _FailingVLM()
    image_path = tmp_path / "xray.png"
    Image.new("RGB", (32, 32)).save(image_path)

    result = controller.evaluate(
        {
            "primary_diagnosis": "Pneumonia",
            "combined_confidence": 0.8,
            "modality": "multimodal",
            "severity": "moderate",
            "urgency": "routine",
            "is_guideline_consistent": False,
            "fusion_strategy": "complementary",
            "agreement": "complementary",
            "explanation": "x",
            "nlp_diagnosis": {"primary_diagnosis": "Pneumonia"},
            "cv_diagnosis": {"detected_predictions": [{"class": "Infiltration"}]},
        },
        image_path=str(image_path),
        symptoms_text="cough",
    )

    assert result["vlm_safety_review"]["_vlm_fallback"] is True
    assert result["is_approved"] is False
    assert "Diagnosis not fully consistent with clinical guidelines" in result["risk_factors"]
