<<<<<<< HEAD
# DengueAI — Dengue Prediction & Risk Assessment System

A full-stack academic MVP focused on Chattogram District, Bangladesh.

## Features
- React dashboard with trend charts, seasonality, data readiness and status notice.
- Scenario simulator (month, rainfall deviation, temperature, humidity).
- FastAPI backend with a Random Forest regression model and tree-spread estimate range.
- Historical data table and a safe rule-based assistant.
- Explicit synthetic-data warning throughout the app.

## Important scientific limitation
The bundled dataset is generated synthetic demonstration data. The model output is only for testing the software pipeline; it is **not** a real dengue forecast, not validated, and must not be used for public-health decisions. The tree percentile range is not a calibrated prediction interval. The prototype does not claim the demo data are DGHS records.

For a research-grade model, replace the CSV with verified monthly records at a consistent geographic level. Do not treat Chattogram Division totals as Chattogram District totals. Preserve source, publication date, geographic level, case definition, missingness and reporting notes. Evaluate using walk-forward time-series validation and report MAE/RMSE and interval coverage.

## Run locally
Requirements: Python 3.10+ and Node.js 18+.

### 1. Backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

### 2. Frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

Optional frontend API override: create `frontend/.env` with `VITE_API_URL=http://localhost:8000`.

## Replace demonstration data
Edit `backend/data/dengue_monthly.csv` after the first backend run. Required columns:
`year,month,location,cases,rainfall_mm,temperature_c,humidity_pct,data_type`

Use one row per month and region. Keep missing values missing rather than fabricating zeros. After replacing data, call `POST /api/train` to retrain. This starter training routine is intentionally simple; implement walk-forward validation before presenting model metrics.

## API
- `GET /api/health`
- `GET /api/overview`
- `GET /api/history`
- `POST /api/predict` — `{ "location":"Chattogram", "month":8, "rainfall_pct":20, "temperature":29, "humidity":80 }`
- `POST /api/chat` — `{ "message":"Explain the rainfall input" }`
- `POST /api/train`

## Project structure
```text
dengueai/
  backend/app/main.py
  backend/data/README.md
  backend/requirements.txt
  frontend/src/App.jsx
  frontend/src/style.css
  frontend/package.json
  README.md
```

## Production checklist
- Replace synthetic records with verified source data and document geographic granularity.
- Add walk-forward validation, baseline comparison, calibrated uncertainty and risk thresholds.
- Add authentication, rate limiting, structured logging, HTTPS and model versioning.
- Only then deploy as a research dashboard; retain clear uncertainty and public-health disclaimers.
=======
# Dengue-Probability-
FINALLYYYY
>>>>>>> 73b2bc7accce9c4389736be58281396fd22381c3
