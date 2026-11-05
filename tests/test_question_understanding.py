import pytest

from app.core.nlp.question_understanding import QuestionUnderstanding


@pytest.fixture(scope="module")
def understanding():
    return QuestionUnderstanding()


@pytest.mark.parametrize("question, intent", [
    # "What is/are ..." opens most questions and decides nothing by itself
    ("What is the treatment for pneumonia?", "treatment"),
    ("What is the best therapy for COPD?", "treatment"),
    ("What is the prognosis for heart failure?", "prognosis"),
    ("What is the outlook for someone with CKD?", "prognosis"),
    ("What is the survival rate after a stroke?", "prognosis"),
    ("What are the side effects of metformin?", "drug_interaction"),
    ("What are the risk factors for COPD?", "risk_factors"),
    ("What is pneumonia?", "diagnosis"),
    ("What are the symptoms of tuberculosis?", "diagnosis"),
    ("What are the diagnostic criteria for diabetes?", "diagnosis"),
    # Treatment
    ("How do you treat community-acquired pneumonia?", "treatment"),
    ("What medications are used for hypertension?", "treatment"),
    ("How should asthma be managed in adults?", "treatment"),
    ("How long should I take antibiotics for a UTI?", "treatment"),
    # Prognosis
    ("How long does it take to recover from pneumonia?", "prognosis"),
    ("What are the complications of untreated hypertension?", "prognosis"),
    # Drug interaction
    ("Can I take ibuprofen with warfarin?", "drug_interaction"),
    ("Can metformin and insulin be used together?", "drug_interaction"),
    ("Are there any contraindications to thrombolysis?", "drug_interaction"),
    # Causes and risk
    ("What causes pneumothorax?", "risk_factors"),
    ("Why do I keep getting urinary tract infections?", "risk_factors"),
    ("Does smoking increase the risk of lung cancer?", "risk_factors"),
    ("Is diabetes hereditary?", "risk_factors"),
    # Prevention
    ("How can I prevent tuberculosis?", "prevention"),
    ("Is there a vaccine for pneumonia?", "prevention"),
    ("Who should be screened for diabetes?", "prevention"),
    # Asking which condition it is, also when phrased with "cause"
    ("Could this be asthma?", "diagnosis"),
    ("What could be causing my chest pain?", "diagnosis"),
    ("What condition causes cough and weight loss?", "diagnosis"),
    ("How is heart failure diagnosed?", "diagnosis"),
    # No pattern at all
    ("Tell me about pneumonia", "general"),
    ("I am worried because my cough is not going away", "general"),
])
def test_intent(understanding, question, intent):
    assert understanding.analyze(question)["intent"] == intent


def test_search_query_spells_the_intent_as_words(understanding):
    query = understanding.analyze("What are the side effects of metformin?")["search_query"]
    assert query.endswith("clinical drug interaction")
