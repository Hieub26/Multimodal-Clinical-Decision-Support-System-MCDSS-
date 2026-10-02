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

### Prerequisites

- Python 3.11+ (developed and tested on 3.13)
- A running PostgreSQL instance (or use the Docker setup below)
- The fine-tuned CV weights at `models/best_densenet121_chestxray14.pth`. They are not in the repository; without them image diagnosis returns `503` instead of guessing.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env:
#   - DATABASE_URL / POSTGRES_PASSWORD (required)
#   - GEMINI_API_KEY (optional — system works without it using rule-based fallback)
```

The guideline index in `data/chroma_db` is built on first start and re-synced automatically whenever the files in `data/guidelines` change.

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

## 🐳 Docker Deployment

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/) v2+

### Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env — set GEMINI_API_KEY, POSTGRES_PASSWORD, etc.

# 2. Build and start all services
docker compose up --build -d

# 3. Check status
docker compose ps
docker compose logs -f
```

### Services

| Service | URL | Description |
|---------|-----|-------------|
| **backend** | http://localhost:8008 | FastAPI API (docs at `/docs`) |
| **frontend** | http://localhost:8501 | Streamlit web interface |
| **db** | localhost:5432 | PostgreSQL database |

### Useful Commands

```bash
# Stop all services
docker compose down

# Stop and remove data volumes
docker compose down -v

# Rebuild a specific service
docker compose build backend
docker compose up -d backend

# View logs for a specific service
docker compose logs -f backend
```

> **Note**: Model weights (`models/*.pth`) are mounted from the host via volume. Make sure the model file exists before starting.

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
├── tests/                        # Unit tests (pytest)
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

`ingest-guidelines` accepts inline `text`, or a `file_path` to a `.txt` file inside `data/guidelines` (relative to that directory). When `ADMIN_API_KEY` is set, the request must carry it in the `X-API-Key` header.

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite uses temporary storage and never calls the LLM or the database.

## 🌳 Git Branching Strategy

To maintain a professional development workflow, this project follows a structured branching strategy:

- **`main`**: Stable, production-ready code. No direct development here.
- **`develop`**: Integration branch for new features and bug fixes.
- **`feature/*`**: Dedicated branches for new features (e.g., `feature/ui-update`).
- **`bugfix/*`**: Dedicated branches for fixing issues (e.g., `bugfix/api-fix`).

All changes should be submitted via **Pull Requests** from `feature/*` or `bugfix/*` branches into `develop`. Once `develop` is stable, it is merged into `main` for release.

