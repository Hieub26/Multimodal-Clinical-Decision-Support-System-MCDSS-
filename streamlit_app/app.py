"""
Streamlit Main Application — Multimodal Clinical Decision Support System.
Professional dark-themed medical interface with multimodal diagnosis capabilities.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import httpx
import json

# Page configuration
st.set_page_config(
    page_title="Clinical Decision Support System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Import components
from components.sidebar import render_sidebar
from components.text_input import render_text_input
from components.image_upload import render_image_upload
from components.results_display import render_results
from components.gradcam_viewer import render_gradcam
from components.history import render_history

# Constants — API_BASE_URL env var is set by Docker; fallback to localhost for local dev
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8008/api")


def main():
    """Main application entry point."""

    # Render sidebar — get navigation, mode, and settings
    page, mode, confidence_threshold = render_sidebar()

    # Route to appropriate page
    if "History" in page:
        render_history()
    elif "About" in page:
        render_about_page()
    else:
        render_diagnosis_page(mode, confidence_threshold)


def render_diagnosis_page(mode: str, confidence_threshold: float):
    """Render the main diagnosis page."""

    # Hero header
    st.markdown("""
    <div class="hero-header">
        <h1>🏥 Multimodal Clinical Decision Support System</h1>
        <p>AI-powered diagnostic analysis combining NLP symptom analysis and medical image classification</p>
    </div>
    """, unsafe_allow_html=True)

    # Input section
    symptoms = ""
    question = ""
    uploaded_file = None

    if mode in ("Multimodal", "Text Only"):
        symptoms, question = render_text_input()

    if mode in ("Multimodal", "Image Only"):
        uploaded_file = render_image_upload()

    st.markdown("---")

    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button(
            "🔬 Analyze",
            use_container_width=True,
            type="primary",
            key="analyze_btn",
        )

    # Process analysis
    if analyze_clicked:
        if not symptoms and not question and not uploaded_file:
            st.warning("Please provide symptoms text, a clinical question, or a medical image.")
            return

        with st.spinner("🔄 Running multimodal analysis pipeline..."):
            response = run_diagnosis(symptoms, question, uploaded_file, mode)

        if response:
            st.session_state["last_result"] = response
        else:
            st.error("Analysis failed. Please check that the API backend is running.")

    # Display results
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        safety = result.get("safety", {})
        cv = result.get("cv_details", {})
        gradcam_path = result.get("gradcam_path")

        # Only pass gradcam_path when approved and finding detected
        effective_gradcam = (
            gradcam_path
            if gradcam_path and safety.get("is_approved") and cv.get("finding_detected")
            else None
        )
        render_results(result, gradcam_path=effective_gradcam)


def run_diagnosis(symptoms: str, question: str, uploaded_file, mode: str) -> dict | None:
    """Send diagnosis request to the FastAPI backend."""
    try:
        if mode == "Text Only" or (not uploaded_file and (symptoms or question)):
            # Text-only diagnosis
            response = httpx.post(
                f"{API_BASE}/nlp/diagnose",
                json={
                    "symptoms_text": symptoms or None,
                    "clinical_question": question or None,
                },
                timeout=600,
            )
        elif uploaded_file:
            # Image or multimodal diagnosis
            files = {"image": (uploaded_file.name, uploaded_file.read(), "image/png")}
            data = {}
            if symptoms:
                data["symptoms_text"] = symptoms
            if question:
                data["clinical_question"] = question

            response = httpx.post(
                f"{API_BASE}/cv/diagnose",
                files=files,
                data=data,
                timeout=600,
            )
        else:
            return None

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None

    except httpx.ConnectError:
        st.error(
            "⚠️ Cannot connect to the API backend. "
            "Please make sure FastAPI is running: `python -m uvicorn app.main:app --port 8008`"
        )
        return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None


def render_about_page():
    """Render the About page."""
    st.markdown("""
    <div class="hero-header">
        <h1>ℹ️ About the System</h1>
        <p>Multimodal Clinical Decision Support System v1.0.0</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## Architecture Overview
    
    This system implements a comprehensive multimodal clinical decision support pipeline:
    
    ### 🧠 NLP Pipeline
    - **Text Preprocessing** — Normalizes symptoms, expands medical abbreviations
    - **Question Understanding** — Extracts clinical intent and medical entities
    - **Embedding Engine** — Converts text to vectors using sentence-transformers
    - **Vector Store (ChromaDB)** — Stores and retrieves clinical guidelines
    - **RAG Engine** — Combines retrieval with LLM reasoning for diagnosis
    
    ### 🩻 Computer Vision Pipeline
    - **Image Preprocessing** — Resizes and normalizes medical images
    - **CV Model (DenseNet-121)** — Classifies medical images
    - **Grad-CAM** — Provides visual explainability of model predictions
    
    ### 🔗 Fusion & Validation
    - **Fusion Engine** — Combines NLP and CV diagnoses (concordant/discordant/complementary)
    - **Guideline Validator** — Cross-references against clinical guidelines
    - **Safety Controller** — Confidence thresholds, red-flag detection, fallback mechanism
    
    ### 🏗️ Infrastructure
    - **FastAPI** — Backend REST API orchestrator
    - **Streamlit** — Interactive web interface
    - **SQLite** — Case history persistence
    - **ChromaDB** — Vector storage for clinical guidelines
    
    ---
    
    ### Technology Stack
    
    | Component | Technology |
    |-----------|-----------|
    | Backend | FastAPI + Uvicorn |
    | Frontend | Streamlit |
    | NLP | sentence-transformers + Gemini LLM |
    | CV | PyTorch (DenseNet-121) |
    | Vector DB | ChromaDB |
    | Database | PostgreSQL (asyncpg) |
    | Explainability | Grad-CAM |
    """)

    st.markdown("""
    <div class="disclaimer">
        ⚕️ <b>DISCLAIMER:</b> This system is for educational and research purposes only. 
        It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
