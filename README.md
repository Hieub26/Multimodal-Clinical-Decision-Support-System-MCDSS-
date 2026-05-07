# 🏥 Multimodal Clinical Decision Support System

An AI-powered clinical decision support system that combines **NLP-based symptom analysis** (RAG with ChromaDB) and **medical image classification** (PyTorch + Grad-CAM) for comprehensive diagnostic suggestions.

> ⚕️ **Disclaimer**: This system is for **educational and research purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment.

## 🏗️ Architecture

```
User Input (Symptoms Text / Clinical Question / Medical Image)
        │
        ├── NLP Pipeline: Text Preprocessing → Embedding → ChromaDB → RAG (LLM)
        │                                                              ↓
        │                                              NLP Diagnosis + Explanation
        │
        ├── CV Pipeline: Image Preprocessing → DenseNet-121 → Grad-CAM
        │                                        ↓
        │                              CV Diagnosis + Confidence
        │
        └──────────────── Fusion Engine ──────────────────┘
                              ↓
                   Combined Diagnosis + Confidence
                              ↓
                   Guideline Validation Layer
                              ↓
                   Safety & Risk Control
                      ↓              ↓
              ✅ Approved       ⚠️ Consult Doctor
                      ↓
        ┌─────────────┼─────────────┐
    JSON Output   Clinical Report   Web UI
                      ↓
              FastAPI Backend
                      ↓
        ┌─────────────┼─────────────┐
    PostgreSQL DB    Image Storage      Logs
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY (optional — system works without it using rule-based fallback)
```

### 3. Launch the System

```bash
python run.py
```

This starts both:
- **FastAPI Backend**: http://localhost:8008 (API docs at /docs)
- **Streamlit Frontend**: http://localhost:8501

### Alternative: Start Services Individually

```bash
# Terminal 1: Backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8008 --reload

# Terminal 2: Frontend
streamlit run streamlit_app/app.py --server.port 8501 --theme.base dark
```

## 📁 Project Structure

```
├── app/                          # FastAPI Backend
│   ├── main.py                   # Application entry point
│   ├── config.py                 # Configuration
│   ├── api/                      # API layer
│   │   ├── dependencies.py       # Auth, DB session, shared deps
│   │   ├── schemas.py            # Pydantic models
│   │   └── endpoints/            # Route handlers
│   │       ├── nlp_router.py     # Text diagnosis endpoints
│   │       ├── cv_router.py      # Image diagnosis endpoints
│   │       └── report_router.py  # Report & history endpoints
│   ├── core/                     # Business logic
│   │   ├── nlp/                  # NLP pipeline
│   │   ├── cv/                   # Computer Vision pipeline
│   │   ├── fusion/               # Diagnosis fusion
│   │   ├── validation/           # Guideline validation
│   │   └── safety/               # Safety & risk control
│   ├── db/                       # Database layer
│   └── utils/                    # Utilities
├── streamlit_app/                # Streamlit Frontend
├── data/                         # Guidelines & ChromaDB storage
├── storage/                      # Images, reports, logs
└── models/                       # PyTorch model weights
```

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend API | FastAPI + Uvicorn |
| Web UI | Streamlit |
| NLP / RAG | sentence-transformers + Google Gemini |
| Computer Vision | PyTorch (DenseNet-121) |
| Explainability | Grad-CAM |
| Vector Database | ChromaDB |
| Database | PostgreSQL (asyncpg) |

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/nlp/diagnose` | Text-based diagnosis |
| POST | `/api/cv/diagnose` | Image/multimodal diagnosis |
| GET | `/api/reports/history` | Case history |
| GET | `/api/reports/history/{id}` | Case details |
| GET | `/api/reports/health` | Health check |
| POST | `/api/nlp/ingest-guidelines` | Ingest guidelines |
