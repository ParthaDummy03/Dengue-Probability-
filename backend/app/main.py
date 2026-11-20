from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import math
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dengue_monthly.csv"
MODEL_PATH = ROOT / "data" / "model.joblib"
app = FastAPI(title="DengueAI API", version="1.0.0", description="Dengue forecasting demonstration API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class PredictionRequest(BaseModel):
    location: str = "Chattogram"
    month: int = Field(ge=1, le=12)
    rainfall_pct: float = Field(default=0, ge=-80, le=300)
    temperature: float = Field(default=29, ge=10, le=45)
    humidity: float = Field(default=75, ge=0, le=100)

class ChatRequest(BaseModel):
    message: str


def create_demo_data():
    """Generate explicitly synthetic demonstration records; never present as observed surveillance data."""
    rng = np.random.default_rng(17)
    rows = []
    seasonal = [18, 14, 12, 10, 12, 24, 55, 95, 120, 90, 48, 25]
    for year in range(2016, 2026):
        year_factor = 0.75 + (year - 2016) * 0.055
        for month in range(1, 13):
            rain = max(0, [8, 18, 35, 75, 210, 420, 610, 520, 360, 180, 45, 12][month-1] + rng.normal(0, 18))
            temp = [20, 23, 27, 29, 29, 29, 28, 28, 28, 27, 24, 21][month-1] + rng.normal(0, .7)
            humidity = [62, 65, 68, 72, 78, 84, 88, 88, 86, 82, 73, 66][month-1] + rng.normal(0, 2)
            cases = max(0, int(seasonal[month-1] * year_factor + rain * .07 + (humidity-70)*.45 + rng.normal(0, 9)))
            rows.append({"year": year, "month": month, "location": "Chattogram", "cases": cases, "rainfall_mm": round(rain, 1), "temperature_c": round(temp, 1), "humidity_pct": round(float(np.clip(humidity, 35, 99)), 1), "data_type": "synthetic_demo"})
    pd.DataFrame(rows).to_csv(DATA, index=False)


def load_data():
    if not DATA.exists(): create_demo_data()
    return pd.read_csv(DATA)


def feature_frame(df):
    x = pd.DataFrame({"month": df.month, "rainfall_mm": df.rainfall_mm, "temperature_c": df.temperature_c, "humidity_pct": df.humidity_pct})
    x["month_sin"] = np.sin(2*np.pi*x.month/12)
    x["month_cos"] = np.cos(2*np.pi*x.month/12)
    return x


def train_model():
    from joblib import dump
    df = load_data().sort_values(["year", "month"])
    x, y = feature_frame(df), df.cases
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42)
    model.fit(x, y)
    dump(model, MODEL_PATH)
    return model


def get_model():
    from joblib import load
    if not MODEL_PATH.exists(): return train_model()
    return load(MODEL_PATH)


def risk_label(cases: float, history: pd.DataFrame, month: int):
    same_month = history[history.month == month].cases
    if len(same_month) < 3: same_month = history.cases
    p50, p85 = float(same_month.quantile(.50)), float(same_month.quantile(.85))
    if cases >= p85: return "High"
    if cases >= p50: return "Moderate"
    return "Low"

@app.get("/")
def root(): return {"name": "DengueAI API", "status": "running", "docs": "/docs"}

@app.get("/api/health")
def health(): return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.get("/api/overview")
def overview():
    df = load_data()
    latest = df.sort_values(["year", "month"]).iloc[-1]
    monthly = df.groupby("month", as_index=False).cases.mean().round(1)
    yearly = df.groupby("year", as_index=False).cases.sum()
    return {"location": "Chattogram", "total_records": int(len(df)), "latest_year": int(latest.year), "latest_month": int(latest.month), "latest_demo_cases": int(latest.cases), "average_monthly_cases": round(float(df.cases.mean()), 1), "monthly_average": monthly.to_dict(orient="records"), "yearly_total": yearly.to_dict(orient="records"), "data_status": "SYNTHETIC DEMONSTRATION DATA — replace with verified DGHS data before research or public-health use"}

@app.get("/api/history")
def history():
    df = load_data().sort_values(["year", "month"])
    return {"records": df.to_dict(orient="records"), "data_status": "synthetic_demo"}

@app.post("/api/predict")
def predict(req: PredictionRequest):
    if req.location.lower() not in ["chattogram", "chittagong"]:
        raise HTTPException(400, "MVP currently supports Chattogram only.")
    df = load_data()
    baseline_rain = float(df[df.month == req.month].rainfall_mm.median())
    scenario_rain = max(0, baseline_rain * (1 + req.rainfall_pct/100))
    features = pd.DataFrame([{"month": req.month, "rainfall_mm": scenario_rain, "temperature_c": req.temperature, "humidity_pct": req.humidity}])
    features["month_sin"] = np.sin(2*np.pi*features.month/12)
    features["month_cos"] = np.cos(2*np.pi*features.month/12)
    model = get_model()
    trees = np.array([tree.predict(features)[0] for tree in model.estimators_])
    estimate = max(0, float(model.predict(features)[0]))
    lower, upper = np.percentile(trees, [10, 90])
    # This interval is tree-to-tree spread, not a calibrated statistical prediction interval.
    return {"location": "Chattogram", "month": req.month, "month_name": datetime(2000, req.month, 1).strftime("%B"), "predicted_cases": round(estimate), "lower_estimate": round(max(0, float(lower))), "upper_estimate": round(max(estimate, float(upper))), "risk": risk_label(estimate, df, req.month), "scenario": {"rainfall_change_pct": req.rainfall_pct, "rainfall_mm_used": round(scenario_rain, 1), "temperature_c": req.temperature, "humidity_pct": req.humidity}, "model": "RandomForestRegressor trained on synthetic demonstration records", "data_status": "DEMO ONLY — not an epidemiological forecast", "interval_note": "Range is model-tree spread, not a calibrated prediction interval."}

@app.post("/api/chat")
def chat(req: ChatRequest):
    msg = req.message.strip()
    if not msg: raise HTTPException(400, "Please enter a question.")
    # Safe MVP assistant: deterministic, no LLM and no independent numerical forecasting.
    return {"answer": "I can help explain the dashboard and run a structured scenario. For a numerical estimate, use the Scenario Simulator so the ML API supplies the result. This MVP currently uses synthetic demonstration records, so its estimates are not real dengue forecasts. Rainfall, humidity, temperature, surveillance quality and local interventions can all affect reported cases. Do not use this prototype for clinical or public-health decisions.", "llm_enabled": False, "data_status": "synthetic_demo"}

@app.post("/api/train")
def retrain():
    model = train_model()
    return {"status": "trained", "model": type(model).__name__, "records": int(len(load_data())), "note": "Training uses the bundled synthetic demonstration dataset. Replace it before evaluating or deploying."}
