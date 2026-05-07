"""
Text input component for symptoms and clinical questions.
"""

import streamlit as st

COMMON_SYMPTOMS = [
    "Fever", "Cough", "Headache", "Chest pain", "Shortness of breath",
    "Fatigue", "Nausea", "Dizziness", "Joint pain", "Rash",
    "Abdominal pain", "Back pain", "Sore throat", "Diarrhea", "Vomiting",
]


def _add_symptom(symptom: str):
    """Callback to add a symptom before the widget is re-instantiated."""
    current = st.session_state.get("symptoms_input", "")
    if current:
        st.session_state["symptoms_input"] = f"{current}, {symptom}"
    else:
        st.session_state["symptoms_input"] = symptom


def render_text_input() -> tuple[str, str]:
    """Render the text input section for symptoms and clinical questions."""

    st.markdown("""
    <div class="section-header">
        <h2>📝 Patient Information</h2>
    </div>
    """, unsafe_allow_html=True)

    # Symptoms text area
    symptoms = st.text_area(
        "Describe symptoms",
        placeholder="e.g., Patient presents with persistent dry cough for 2 weeks, "
                    "low-grade fever (38.1°C), fatigue, and mild shortness of breath "
                    "on exertion. No chest pain or hemoptysis.",
        height=120,
        key="symptoms_input",
    )

    # Quick symptom chips — use on_click callback so state is modified
    # BEFORE the widget is re-instantiated on the next rerun
    st.markdown("**Quick add symptoms:**")
    cols = st.columns(5)
    for i, symptom in enumerate(COMMON_SYMPTOMS):
        with cols[i % 5]:
            st.button(
                symptom,
                key=f"symptom_{i}",
                use_container_width=True,
                on_click=_add_symptom,
                args=(symptom.lower(),),
            )

    st.markdown("---")

    # Clinical question
    question = st.text_input(
        "Clinical question (optional)",
        placeholder="e.g., What could cause persistent cough with low-grade fever?",
        key="question_input",
    )

    return symptoms, question
