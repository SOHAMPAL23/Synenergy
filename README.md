# EnerVision AI 🔋

> **End-to-end Energy Forecasting, Anomaly Detection & Optimization Platform**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-NeonDB-blue?logo=postgresql)](https://neon.tech)
[![Python](https://img.shields.io/badge/Python-3.11-yellow?logo=python)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://docker.com)

---

## 🏗 Architecture

```
Synenergy/
├── backend/                    ← FastAPI backend
│   ├── api/
│   │   └── routes/
│   │       ├── auth.py         ← POST /auth/register, /login, /refresh, GET /me
│   │       ├── upload.py       ← POST /upload
│   │       ├── ml.py           ← POST /train, GET /forecast /anomalies /recommendations /explanations
│   │       └── dashboard.py    ← GET /dashboard, /health
│   ├── core/
│   │   ├── config.py           ← Pydantic settings (env vars)
│   │   └── security.py         ← JWT + bcrypt
│   ├── database/
│   │   └── session.py          ← Async SQLAlchemy + NeonDB
│   ├── models/
│   │   └── orm.py              ← 7 SQLAlchemy tables
│   ├── schemas/
│   │   └── schemas.py          ← Pydantic v2 request/response models
│   ├── repositories/
│   │   └── repositories.py     ← Generic async CRUD + domain repos
│   ├── services/
│   │   └── services.py         ← Business logic + ML integration
│   ├── middleware/
│   │   ├── rate_limit.py       ← Sliding window rate limiter
│   │   └── logging.py          ← Structured request logging
│   ├── tests/
│   │   └── test_api.py         ← Async httpx API tests
│   └── main.py                 ← FastAPI app factory
│
├── ml/                         ← ML Pipeline
│   ├── ingestion/              ← CSV loading + schema validation
│   ├── preprocessing/          ← Cleaning, dedup, outlier capping
│   ├── feature_engineering/    ← Time/lag/rolling features
│   ├── forecasting/            ← LinearRegression, RandomForest, XGBoost, ARIMA, SARIMA, SARIMAX
│   ├── anomaly_detection/      ← Z-Score, IQR, IsolationForest, LOF, One-Class SVM
│   ├── explainability/         ← SHAP values + plots
│   ├── recommendation_engine/  ← Rule-based optimization advice
│   ├── pipeline.py             ← Master orchestrator
│   └── tests/
│       ├── test_ingestion.py
│       ├── test_features.py
│       ├── test_forecasting.py
│       ├── test_anomaly.py
│       └── test_pipeline.py    ← Comprehensive integration tests (NEW)
│
├── alembic/                    ← Database migrations
│   ├── versions/
│   │   └── 001_initial.py      ← Creates all 7 tables
│   └── env.py                  ← Async Alembic environment
├── Dockerfile                  ← Multi-stage production image
├── docker-compose.yml
├── pyproject.toml              ← Pytest + coverage config
└── .env.example                ← Environment variable template
```

---

## 🗄 Database Tables (NeonDB / PostgreSQL)

| Table | Description |
|---|---|
| `users` | Auth accounts with RBAC roles (admin/analyst/viewer) |
| `sites` | Physical locations/facilities |
| `meters` | Energy meters per site |
| `energy_records` | Uploaded time-series energy readings |
| `forecasts` | 24h / 7d / 30d ML forecasts (JSONB) |
| `anomalies` | Detected anomalous consumption events |
| `recommendations` | Energy optimization recommendations |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# NeonDB is pre-configured in .env.example
```

### 3. Run database migrations

```bash
alembic upgrade head
```

### 4. Start the API

```bash
uvicorn backend.main:app --reload --port 8000
```

### 5. Open API docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 🐳 Docker

```bash
# Build and run
docker-compose up --build

# Health check
curl http://localhost:8000/api/v1/health
```

---

## 🔌 API Endpoints

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/auth/register` | Register new user |
| POST | `/api/v1/auth/login` | Login → get JWT tokens |
| POST | `/api/v1/auth/refresh` | Refresh access token |
| GET | `/api/v1/auth/me` | Get current user profile |

### Data & ML

| Method | Endpoint | Description | Auth |
|---|---|---|---|
| POST | `/api/v1/upload` | Upload energy CSV | ✅ |
| POST | `/api/v1/train` | Train all ML models | ✅ |
| GET | `/api/v1/forecast` | Get 24h/7d/30d forecasts | ✅ |
| GET | `/api/v1/anomalies` | Get anomaly detection results | ✅ |
| GET | `/api/v1/recommendations` | Get optimization recommendations | ✅ |
| GET | `/api/v1/explanations` | Get SHAP feature importance | ✅ |
| GET | `/api/v1/dashboard` | Complete dashboard data | ✅ |
| GET | `/api/v1/health` | Health check | ❌ |

---

## 🧪 Running Tests

### ML Pipeline Tests

```bash
# All tests
pytest ml/tests/ -v

# Specific test file
pytest ml/tests/test_pipeline.py -v

# With coverage
pytest ml/tests/ --cov=ml --cov-report=term-missing
```

### Backend API Tests

```bash
# All backend tests
pytest backend/tests/ -v

# With coverage
pytest backend/tests/ --cov=backend --cov-report=term-missing
```

### All Tests

```bash
pytest -v
```

---

## 🤖 ML Pipeline Stages

1. **Data Ingestion** → Load CSV, validate schema (timestamp index, target column)
2. **Preprocessing** → Deduplicate, fill missing values, cap IQR outliers
3. **Feature Engineering** → Time features + lag features (t-1, t-24, t-168) + rolling stats
4. **Model Selection** → Train 6 models, auto-select best by RMSE
5. **Forecasting** → Recursive 24h / 7d / 30d horizon forecasts
6. **Anomaly Detection** → Ensemble of 5 methods (Z-Score, IQR, IsolationForest, LOF, OC-SVM)
7. **Explainability** → SHAP values (TreeExplainer / LinearExplainer)
8. **Recommendations** → Rule-based energy optimization advice

---

## 🔐 Security Features

- **JWT Authentication** — Access token (30min) + Refresh token (7 days)
- **RBAC** — admin / analyst / viewer roles
- **Rate Limiting** — 100 req/60s per IP (sliding window)
- **bcrypt** password hashing
- **Input Validation** — Pydantic v2 schema validation on all inputs
- **CORS** — Configurable allowed origins
- **Request Logging** — Structured logs with correlation IDs

--

