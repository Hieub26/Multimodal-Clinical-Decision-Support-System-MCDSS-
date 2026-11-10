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
| Safety review of image cases (can only add caution or escalate) | OpenAI vision model (`VLM_SAFETY_MODEL`) | Skipped |
| Chest X-ray classification | DenseNet-121, local weights | Image diagnosis returns `503` |
| Second image model, optional (`CV_CLEAR_ENABLED=true`) | CLEAR zero-shot, local weights (`requirements-clear.txt`, `scripts/fetch_clear.py`) | DenseNet alone, with its own thresholds |

- **The judge is not the generator.** Gemini writes the diagnosis and an OpenAI model grades whether the guideline library supports it, so the two do not share blind spots. If the OpenAI call fails, validation falls back to token matching, never quietly to Gemini.
- **The image safety review only escalates.** For cases with an image, an OpenAI vision model looks at the film and the pipeline's output and may add a caution or escalate the case for review. It cannot clear any risk, not even a soft one such as a text/image disagreement or a guideline judge's "not supported": it has not been evaluated, so it is never allowed to make a case look safer than the rules, the judge and the red-flag gate left it. It runs on a different provider from the Gemini generator, so in a text + image case the generator does not review its own answer.
- **What leaves the machine.** The patient's text goes to Gemini (diagnosis), to TypeSafe (context judgments) and, for image cases, to OpenAI with the image (safety review). The OpenAI guideline judge receives only the diagnosis label and guideline passages.
- **Gemini free tier.** In October 2026 the free tier allowed 20 requests per day for `gemini-2.5-flash`. Once it is used up the generator falls back to the rule-based engine until the allowance resets. Judging on OpenAI leaves that allowance to diagnosis.
- **What the rule-based engine's answer is worth.** Its confidence follows how clearly the symptoms single out one disease, fitted on a public benchmark. A diagnosis is shown as approved only from a confidence of 0.70, where 15 of 16 were right on the benchmark's test split; a picture that fits several diseases ("cough, fever, phlegm") is named but held back for review.
- **What an image flag is worth.** The classifier's score is not the chance that a flag is right: on the official patient-disjoint test split of ChestX-ray14, 23% of the pulmonary edema flags were true findings. Every flag is reported with the precision measured on that test split, a flag below the measured threshold says so, and a flag the test figures do not back (precision under 25%) is never rated critical or an emergency by itself.
- **The CLEAR ensemble is off by default.** When enabled, cardiomegaly, effusion, atelectasis, edema and pneumonia combine DenseNet with the CLEAR vision-language model, with thresholds and precision measured on Open-I instead of ChestX-ray14. It ranks findings better on three collections: mean AUROC over the combined classes +0.03 on held-out Open-I frontal films, +0.037 (95% CI +0.028 to +0.045) on VinDr-CXR (15,000 films read by three radiologists, which neither model was trained on: the external validation) and +0.010 (+0.002 to +0.019) on the ChestX-ray14 test split (the DenseNet's own collection, patient-disjoint from its training: an internal test), where atelectasis is worse (0.762 → 0.732). Its thresholds do not carry over: on ChestX-ray14 they flag 54% of the "No Finding" films against 46% for DenseNet, while on VinDr they are strict (cardiomegaly: 91% precision, 16% recall). Recalibrate on local films before relying on its flags; see [evaluation section 7](evaluation/README.md). It adds about 1 GB of memory, and CLEAR's training data (MIMIC-CXR, CheXpert-Plus) carry their own terms of use.

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

When `API_KEY` is set, the diagnosis, history and download endpoints require it in the `X-API-Key` header; the health check stays open. The `validation` object of a diagnosis response says how guideline support was judged (`method`, `judge_model`), and `cv_details.reliability` what an image flag was worth on the model's held-out test split.

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

### Chest X-ray classifier

DenseNet-121 (ImageNet initialisation) fine-tuned on NIH ChestX-ray14 with [training/train_densenet121_chestxray14.py](training/train_densenet121_chestxray14.py). The test set is the official `test_list.txt` (25,596 images, 2,797 patients), used once, after the checkpoint and the thresholds were fixed. Validation is 10% of the remaining patients (8,536 images), split by `Patient ID`; no patient is in two splits. Validation picked the epoch (6 of 10, early stopping) and the per-class thresholds (best F1).

**Mean AUROC on the test set: 0.801 (95% CI 0.795–0.807, patient-level bootstrap).**

| Finding | Test AUROC (95% CI) | Precision / recall at the threshold |
|---|---|---|
| Emphysema | 0.903 (0.889–0.915) | 35% / 62% |
| Hernia | 0.882 (0.824–0.932) | 75% / 24% |
| Cardiomegaly | 0.873 (0.854–0.889) | 45% / 29% |
| Pneumothorax | 0.852 (0.836–0.865) | 41% / 51% |
| Edema | 0.839 (0.822–0.855) | 23% / 17% |
| Effusion | 0.818 (0.807–0.828) | 48% / 55% |
| Mass | 0.814 (0.794–0.831) | 34% / 42% |
| Fibrosis | 0.809 (0.783–0.834) | 15% / 14% |
| Atelectasis | 0.770 (0.756–0.782) | 31% / 53% |
| Pleural thickening | 0.767 (0.746–0.786) | 15% / 37% |
| Nodule | 0.751 (0.730–0.771) | 28% / 29% |
| Consolidation | 0.729 (0.714–0.745) | 18% / 25% |
| Pneumonia | 0.719 (0.698–0.742) | 7% / 15% |
| Infiltration | 0.696 (0.685–0.706) | 42% / 45%* |

\* Infiltration is the most common and most noisily labelled finding (24% of test films), and its F1-optimal threshold flagged 60% of all films. Its threshold is instead the lowest that flags no more films than carry the finding ([training/recalibrate_thresholds.py](training/recalibrate_thresholds.py)). As the run kept no validation scores, it was chosen on half of the test patients and measured on the other half, which shares no patient with the first; the row shows that second half.

Per-class figures, the training history and the split are in [training/results/](training/results/) and `models/best_densenet121_chestxray14_meta.json`. An earlier version of the model was evaluated on a random per-image split, which lets one patient's films sit in both training and validation; its figures (mean AUROC 0.836) are not comparable and are no longer used.

## ⚠️ Limitations

- **No clinical validation.** Nothing here was tested on real patients or with clinicians; it is a research prototype.
- **Small, self-labeled text sets.** The context, red-flag and judge sets were labeled by one person (the author). The red-flag held-out set has 106 texts, of which 55 are emergencies.
- **Image labels are text-mined.** ChestX-ray14 labels were extracted from radiology reports and are noisy; the test figures inherit that noise. Low-prevalence findings (Pneumonia, Fibrosis) have low precision at any useful threshold.
- **No out-of-distribution check.** The image model accepts any picture; it scores a CT or MRI slice like a chest film (low scores in practice, but nothing rejects it).
- **The image safety review is not evaluated.** No labeled set measures how often the vision model's escalations are right, which is why it can only add caution.
- **Grad-CAM is coarse.** Maps come from a 7×7 feature grid, so they show the region the model used, not the outline of a lesion, and a bilateral finding can show up as one blob.

## 🌳 Git Branching Strategy

To maintain a professional development workflow, this project follows a structured branching strategy:

- **`main`**: Stable, production-ready code. No direct development here.
- **`develop`**: Integration branch for new features and bug fixes.
- **`feature/*`**: Dedicated branches for new features (e.g., `feature/ui-update`).
- **`bugfix/*`**: Dedicated branches for fixing issues (e.g., `bugfix/api-fix`).

All changes should be submitted via **Pull Requests** from `feature/*` or `bugfix/*` branches into `develop`. Once `develop` is stable, it is merged into `main` for release.

