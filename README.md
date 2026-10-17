# VitalSync — Metabolic Digital Twin Platform

**VitalSync** is a full-stack metabolic digital twin and early-warning clinical decision-support prototype for **2-hour post-prandial and stress-induced hyperglycaemic spike prediction** (`glucose >= 180 mg/dL` sustained with `>= +25 mg/dL` rise within 120 minutes).

It fuses five continuous wearable streams (**CGM interstitial glucose**, **optical heart rate**, **autonomic HRV RMSSD**, **accelerometer steps / MET expenditure**, and **pulse oximetry SpO2**) with **prior-night sleep architecture**, **meal carbohydrate logs**, and **longitudinal EHR context** onto a causal 5-minute grid, extracts **84 physiological features**, and scores each decision step with an isotonic-calibrated **Gradient Boosted Decision Tree (`XGBClassifier` / `HistGradientBoostingClassifier`)** paired with exact **TreeSHAP local attributions** and **risk-conditional quantile trajectory forecasts (`p10`, `p50`, `p90`)**.

> **Research & Engineering Prototype Notice:** All patient profiles, wearable streams, and EHR records in this repository are **100% synthetic**, generated from a compartmental Bergman-style glucose-insulin-meal-exercise-autonomic simulator grounded in published physiological priors (Dalla Man et al. 2007, Hovorka et al. 2004, Battelino et al. 2019, Task Force ESC/NASPE 1996). VitalSync is not a certified medical device and must not be used for real-world clinical diagnosis or dosing.

---

## Key Capabilities

1. **Interactive Clinical Analytics Dashboard (`frontend/`)**:
   - **Patient Cohort Selector**: Switch between 3 physiologically distinct synthetic patients (`DT-1047 Aarav Mehta`, `DT-1182 Sunita Rao`, `DT-2210 Imran Qureshi`), each with a 48-hour personal baseline window and a 24-hour high-resolution scenario day at 5-minute intervals.
   - **2-Hour Spike Risk & Conditional Forecast Fan**: Displays calibrated 2-hour hyperglycaemic spike probability (`0–100%`), operating threshold alerts (`0.44`), personal trend extrapolation, and empirical `p10 / p50 / p90` glucose trajectory bands.
   - **Digital Twin 5-Domain Assessment**: Real-time 0–100 physiological scoring across *Metabolic Stability*, *Glucose Stability*, *Cardiovascular State*, *Recovery*, and *Activity*, with interactive domain drill-down modals.
   - **Personal 48-Hour Baseline Comparison**: Individualized z-score deviations relative to each patient's own habitual physiology.
   - **Local TreeSHAP & Global Explainability**: Additive percentage-point risk attributions per feature and per signal family, held-out permutation importance, secondary logistic regression odds ratios, and an interactive 84-feature catalogue explorer.
   - **Physiological Replay & Time-Travel**: Play/pause auto-replay, step `-15m` / `+15m` / `+30m` / `+1h`, jump to clinical scenario clock presets (`09:30` to `14:30`), adjust replay speed (`1x` to `12x`), and inspect ground-truth observed outcomes when replaying historical windows.
   - **Clinical Narrative Layer (Groq LLM + Deterministic Fallback)**: Generates structured clinical interpretations (`Summary`, `Explain Risk`, `Action Plan`, `Meal Impact`) via Groq (`llama-3.3-70b-versatile`) when `GROQ_API_KEY` is configured, or instantaneously via a deterministic clinical template engine when offline.

2. **Validated Machine Learning Pipeline (`model/`, `datasets/`)**:
   - Trained on **72 synthetic patients** (41,472 5-minute windows), calibrated on **16 validation patients** (9,216 windows), and evaluated on **20 held-out test patients** (10,960 windows) with strict patient-wise splitting.
   - **Held-Out Test Metrics**:
     - **AUROC**: `0.920` (95% CI `0.913–0.926`) vs `0.785` for a glucose-only baseline rule (`+0.135` multi-sensor AUROC lift)
     - **AUPRC**: `0.625` (95% CI `0.600–0.650`, base event rate `12.6%`)
     - **Brier Score**: `0.0680` | **Calibration ECE**: `0.0156`
     - **Sensitivity / Specificity** (at threshold `0.44`): `62.2%` / `93.5%`

---

## Repository Structure

```text
digitaltwin/
├── main.py                         # Root ASGI entrypoint (`from backend.main import app`) for Vercel & Uvicorn
├── pyproject.toml                  # Python project metadata & Vercel FastAPI entrypoint config
├── vercel.json                     # Single-project Vercel FastAPI configuration
├── requirements.txt                # Production Python dependencies (uses xgboost-cpu on Linux)
├── .env.example                    # Environment variable template (no secrets committed)
├── backend/
│   ├── main.py                     # FastAPI application, CORS, static mount, & route registration
│   ├── settings.py                 # Environment-backed configuration & path resolution
│   ├── config/profile.json         # Researcher / platform author metadata
│   ├── data/
│   │   ├── patients.json           # 3 interactive digital twin patient profiles & scenario scripts
│   │   └── sensor_data/            # Reference CSV/JSON snapshots
│   ├── models/predictor.py         # RiskPredictor (model loading, TreeSHAP, occlusion fallback)
│   ├── routes/                     # FastAPI routers (/api/health, /patients, /simulation, /insights, /model)
│   └── services/                   # Twin, prediction, feature engineering, timeline, simulation, & Groq services
├── frontend/
│   ├── index.html                  # Clinical analytics dashboard shell (5 clinical views)
│   ├── favicon.svg                 # VitalSync vector icon
│   └── assets/
│       ├── css/styles.css          # Dark clinical interface stylesheet
│       ├── js/app.js               # Main dashboard state & simulation controller
│       ├── js/api.js               # Same-origin REST API client (/api/*)
│       ├── js/charts.js            # Chart.js 4.4.7 clinical chart wrappers
│       └── vendor/chart.umd.js     # Vendored Chart.js v4.4.7 UMD bundle (zero external CDN dependency)
├── model/
│   ├── feature_engineering.py      # 84 causal multi-sensor rolling features & event labeller
│   ├── train.py                    # Cohort training, isotonic calibration, & TreeSHAP artifact export
│   ├── evaluate.py                 # Standalone held-out verification & report generator
│   └── artifacts/                  # Pre-trained twin_model.joblib, metrics.json, calibration.json
├── datasets/synthetic/
│   ├── generate_cohort.py          # 120-patient physiological cohort generator
│   ├── cohort_patients.csv.gz      # Compressed synthetic cohort time-series
│   └── dataset_metadata.json       # Cohort provenance & split manifest
└── tests/
    └── test_pipeline.py            # 26 automated pytest regression & API/UI verification tests
```

---

## Local Setup & Running

### 1. Create a Virtual Environment & Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest>=8.0
```

### 2. Optional Environment Configuration

Copy `.env.example` to `.env` if you wish to customize settings or enable live Groq LLM narratives:

```bash
cp .env.example .env
```

- When `GROQ_API_KEY` is omitted or empty, VitalSync runs in **deterministic clinical interpretation mode** with zero external API calls.

### 3. Start the Application Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

*(Alternatively, `uvicorn backend.main:app --host 0.0.0.0 --port 8000` works identically.)*

Open `http://localhost:8000` in your browser:
- **Dashboard UI**: `http://localhost:8000/`
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/api/health`

### 4. Run Automated Tests

```bash
pytest -v
```

---

## Deploying to Vercel

### Why the Previous Deployment Returned `404: DEPLOYMENT_NOT_FOUND`

The error `404: DEPLOYMENT_NOT_FOUND` (`Code: DEPLOYMENT_NOT_FOUND`, `ID: bom1::6nq5g-1792169759547-b085c486d6b4`) occurs at Vercel's edge router when the requested `.vercel.app` hostname does not map to an active, successfully built production deployment. Specifically, the original repository had four deployment-blocking issues that have now been fixed:

1. **Missing Frontend Files**: `frontend/` only contained `assets/vendor/chart.umd.js` without `frontend/index.html`, CSS, or application JS.
2. **Broken Multi-Service `vercel.json`**: The previous `vercel.json` used an experimental `services` block splitting `frontend` (`root: "frontend"`) and `backend` (`root: "backend"`). Setting `root: "backend"` broke all `from backend...` absolute Python imports and excluded top-level `model/artifacts/` and `datasets/synthetic/` from the serverless bundle.
3. **Serverless Read-Only Filesystem Mutation & Entrypoint Discovery**:
   - `PatientRepository` previously wrote 6 CSV/JSON files into `backend/data/sensor_data/` on every startup (`rebuild_files=True`), which fails on Vercel's read-only `/var/task` filesystem. `persist_sensor_files` now defaults to `False`.
   - Added root `main.py` (`from backend.main import app`) and `pyproject.toml` (`[tool.vercel] entrypoint = "backend.main:app"`) alongside direct `fastapi>=0.115` listing in `requirements.txt` so `@vercel/frameworks` and `@vercel/python` detect the FastAPI entrypoint automatically.
4. **Linux Wheel Size Over Vercel's 500 MB Limit**: Unconditional `xgboost>=2.0` on Linux x86_64 pulled `nvidia-nccl-cu12` (`469 MB`) + CUDA `xgboost` (`228 MB`), exceeding Vercel's serverless bundle size limit. Switching to `xgboost-cpu>=2.0; sys_platform == "linux"` reduces `xgboost` from ~697 MB to **19 MB unpacked** while loading the exact same `twin_model.joblib` artifact.

### Step-by-Step Vercel Deployment Instructions

#### Option A: Via Vercel Dashboard (GitHub Integration)
1. Push this branch (`main` or your deployment branch) to GitHub (`aminotkartik/digitaltwin`).
2. Open the [Vercel Dashboard](https://vercel.com/dashboard) and select your project (or click **Add New... -> Project** and import `aminotkartik/digitaltwin`).
3. Verify **Project Settings -> General**:
   - **Framework Preset**: `FastAPI` (auto-detected from `vercel.json` / `pyproject.toml` / `requirements.txt`)
   - **Root Directory**: `.` (leave blank / repository root — **do not** set to `frontend` or `backend`)
   - **Build & Development Settings**: Leave Override switches **Off** so Vercel uses `@vercel/python`'s native `uv` builder and automatic public-package externalizer.
4. *(Optional)* In **Project Settings -> Environment Variables**, add `GROQ_API_KEY` if you want live Groq LLM clinical narratives (leave unset to use the built-in deterministic clinical engine).
5. In **Project Settings -> Domains**, confirm the production domain you are visiting is attached to this project.
6. Trigger a **Redeploy** from the **Deployments** tab and click **Visit** once the build status reaches **Ready**.

#### Option B: Via Vercel CLI
```bash
npm i -g vercel
vercel login
vercel link
vercel deploy --prod
```

---

## API Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health, loaded patient count, model artifact status, & LLM status |
| `GET` | `/api/config` | Public thresholds, risk bands, demo clocks, and headline metrics |
| `GET` | `/api/patients` | List all cohort patients with real-time snapshot summaries |
| `GET` | `/api/patients/{id}` | Full dashboard payload for patient `{id}` at `clock` or `index` |
| `GET` | `/api/patients/{id}/sensors` | Multi-channel time-series (`glucose`, `hr`, `hrv`, `steps`, `met`, `spo2`) |
| `GET` | `/api/patients/{id}/prediction` | 2-hour spike probability, trajectory fan, confidence, & data quality |
| `GET` | `/api/patients/{id}/prediction/explain` | Exact TreeSHAP local attributions & group contributions |
| `GET` | `/api/patients/{id}/twin` | 5-domain Digital Twin scores, radar payload, & physiological drivers |
| `GET` | `/api/patients/{id}/timeline` | Debounced clinical event feed & 24h pattern detection |
| `POST` | `/api/simulation/start` | Start or re-anchor a simulation replay session (`from_clock` or `from_index`) |
| `POST` | `/api/simulation/step` | Advance replay session by `minutes` and return updated dashboard |
| `POST` | `/api/insights` | Generate structured clinical interpretation (Groq LLM or deterministic fallback) |
| `GET` | `/api/model/metrics` | Held-out test/val/train AUROC, AUPRC, Brier, ECE, ROC curve, & subgroups |

---

## License

Released under the [MIT License](LICENSE).
