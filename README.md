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
                   Guideline Validation Layer   (independent LLM judge)
                              ↓
                   Safety & Risk Control        (red-flag gate: rules + Jev)
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

## 🧠 Models and External Services

Every external model is optional. Each step has a fallback that runs when its key is missing or its call fails.

| Step | Model or service | Without its key, or when the call fails |
|------|------------------|------------------------------------------|
| Diagnosis from text (RAG) | Google Gemini (`LLM_MODEL_NAME`) | Rule-based fallback engine |
| What the text says about each symptom (present, denied, past, someone else's, asked about) | TypeSafe Jev (`JEV_MODEL`) | Rule-based NegEx-style logic |
| Red-flag gate on the patient's text | Keyword rules plus Jev emergency signs | Keyword rules only |
| Guideline validation (LLM-as-a-Judge) | OpenAI (`JUDGE_MODEL_NAME`) | Token matching. With no OpenAI key at all, Gemini judges its own output |
| Safety review of image cases | Gemini VLM (`VLM_SAFETY_MODEL`) | Skipped |
| Chest X-ray classification | DenseNet-121, local weights | Image diagnosis returns `503` |

- **The judge is not the generator.** Gemini writes the diagnosis and an OpenAI model grades whether the guideline library supports it, so the two do not share blind spots. If the OpenAI call fails, validation falls back to token matching, never quietly to Gemini.
- **What leaves the machine.** The patient's text goes to Gemini (diagnosis, image safety review) and to TypeSafe (context judgments). The OpenAI judge receives only the diagnosis label and guideline passages. Images go to Gemini only for the safety review.
- **Gemini free tier.** In October 2026 the free tier allowed 20 requests per day for `gemini-2.5-flash`. Once it is used up the generator falls back to the rule-based engine until the allowance resets. Judging on OpenAI leaves that allowance to diagnosis.
- **What the rule-based engine's answer is worth.** Its confidence follows how clearly the symptoms single out one disease, fitted on a public benchmark. A diagnosis is shown as approved only from a confidence of 0.70, where 15 of 16 were right on the benchmark's test split; a picture that fits several diseases ("cough, fever, phlegm") is named but held back for review.
- **What an image flag is worth.** The classifier's score is not the chance that a flag is right: on the model's validation set a pulmonary edema flag was a true finding 17% of the time. Every flag is reported with that measured precision, a flag below the validated threshold says so, and a flag the validation figures do not back is never rated critical or an emergency by itself.

How much each of these helps is measured in [evaluation/](evaluation/README.md).

## 🔐 Access and Privacy

- **Local by default.** The API and the UI listen on `127.0.0.1`, and Docker Compose publishes its ports on `127.0.0.1`. The case history is patient data and the UI has no login, so reaching them from another machine has to be a decision: set `FASTAPI_HOST` / `STREAMLIT_HOST` (or `BIND_ADDRESS` for Docker) to `0.0.0.0`, set `API_KEY`, and put an authenticating proxy in front of the UI.
- **`API_KEY`** (optional). When set, diagnosis, case history and report download require it in the `X-API-Key` header. `run.py` and Docker Compose hand it to the frontend.
- **`ADMIN_API_KEY`** (optional). When set, guideline ingestion requires it. It is accepted wherever `API_KEY` is.
- **Logs hold no patient text.** Symptom text, questions and uploaded file names are logged as lengths and counts; uploaded images are stored under generated names and are not served over HTTP.

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
#   - OPENAI_API_KEY (optional — independent judge for guideline validation)
#   - TYPESAFE_API_KEY (optional — Jev context judgments and emergency signs)
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
python -m uvicorn app.main:app --host 127.0.0.1 --port 8008 --reload

# Terminal 2: Frontend
streamlit run streamlit_app/app.py --server.address 127.0.0.1 --server.port 8501 --theme.base dark
```

The frontend does not read `.env`. If `API_KEY` is set there, export the same value in the frontend's terminal (`run.py` does this for you).

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

All three ports are published on `127.0.0.1` only (see [Access and Privacy](#-access-and-privacy)). The frontend image is built from `requirements-frontend.txt` and carries Streamlit, an HTTP client and Pillow, not the model stack.

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
├── evaluation/                   # Labeled datasets, evaluation scripts, results
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
| Clinical context & red-flag gate | Rule-based NegEx-style logic + TypeSafe Jev |
| Guideline judge | OpenAI (structured outputs) |
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

When `API_KEY` is set, the diagnosis, history and download endpoints require it in the `X-API-Key` header; the health check stays open. The `validation` object of a diagnosis response says how guideline support was judged (`method`, `judge_model`), and `cv_details.reliability` what an image flag was worth on the model's validation set.

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite uses temporary storage and never calls an external API or the database. It needs no API key, model weights or model download, and runs on every push and pull request ([.github/workflows/tests.yml](.github/workflows/tests.yml)), together with a check of the Compose file and a build of the frontend image.

## 📊 Evaluation

Labeled datasets, the scripts that score them and the saved results are in [evaluation/](evaluation/README.md). Each task has a development set, used to write the rules and the model instructions, and a held-out set written before any system was run on it. Headline results on the held-out sets:

| Task | Baseline | With the model |
|------|----------|----------------|
| Is a symptom mention a current finding of the patient? (133 mentions) | Rules: 68.4% accuracy | Jev: 99.2% |
| Should the red-flag gate escalate this text? (106 texts, first blind run) | Keyword rules: 27.3% sensitivity, 64.7% specificity | Rules + Jev: 98.2% sensitivity, 86.3% specificity |
| Do the retrieved guidelines support the diagnosis? (43 pairs) | Token matching as first measured: 72.1% accuracy | OpenAI judge: 100% (43 of 43) |
| Is a concept negated? (public NegEx set, 1,869 annotations) | Rules: 97.8% accuracy, F1 0.945 | Jev: 96.4%, F1 0.919 |

The rules are better than Jev on the public negation benchmark and worse on patient-style text with history, family members and questions, which is why Jev is added on top of the rules instead of replacing them.

The rule-based diagnosis engine, which runs when Gemini is unavailable, is measured on a public symptom-to-diagnosis benchmark (184 scored texts of its test split):

| | Engine as first measured | Current engine |
|---|---|---|
| First diagnosis is the labeled one, for conditions the engine covers | 27% to 33%, depending on the interpreter's hash seed | 55.7% |
| Diagnoses passing the confidence gate | none | 16, of which 15 correct (93.8%) |
| Conditions outside its list shown with a diagnosis | none | 1 of 96 |

The sets other than the two public ones are small and were labeled by one person; [evaluation/README.md](evaluation/README.md) lists what each number does and does not show.

## 🌳 Git Branching Strategy

To maintain a professional development workflow, this project follows a structured branching strategy:

- **`main`**: Stable, production-ready code. No direct development here.
- **`develop`**: Integration branch for new features and bug fixes.
- **`feature/*`**: Dedicated branches for new features (e.g., `feature/ui-update`).
- **`bugfix/*`**: Dedicated branches for fixing issues (e.g., `bugfix/api-fix`).

All changes should be submitted via **Pull Requests** from `feature/*` or `bugfix/*` branches into `develop`. Once `develop` is stable, it is merged into `main` for release.

