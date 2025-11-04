# predict_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(title="Crime Risk Prediction API")

# Add CORS Middleware (correct place)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load models and artifacts
MODEL_DIR = Path("models")
reg_model = joblib.load(MODEL_DIR / "xgboost_reg.joblib")
clf_model = joblib.load(MODEL_DIR / "xgboost_clf.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
le_top = joblib.load(MODEL_DIR / "le_top.joblib")
agg = pd.read_csv(MODEL_DIR / "grid_aggregated.csv")

class CrimeQuery(BaseModel):
    lat: float
    lon: float
    hour: int = 12
    top_crime_type: str = "Unknown"

@app.get("/")
def home():
    return {"message": "✅ Crime Risk Prediction API is running!"}

@app.post("/predict")
def predict_risk(q: CrimeQuery):
    nearest = agg.loc[((agg['lat_grid'] - q.lat).abs() + (agg['lon_grid'] - q.lon).abs()).idxmin()]
    if q.top_crime_type not in le_top.classes_:
        safe_type = le_top.classes_[0]
    else:
        safe_type = q.top_crime_type
    top_code = int(le_top.transform([safe_type])[0])

    features = np.array([[
        nearest["total_crimes"],
        nearest["unique_crime_types"],
        q.hour if 0 <= q.hour <= 23 else nearest.get("mean_hour", 12),
        nearest.get("std_hour", 0),
        nearest.get("night_prop", 0),
        top_code
    ]])

    features_scaled = scaler.transform(features[:, :-1])
    X_input = np.hstack([features_scaled, features[:, -1].reshape(-1, 1)])

    risk_score = float(reg_model.predict(X_input)[0])
    risk_type_code = int(clf_model.predict(X_input)[0])
    risk_type_label = ["low", "medium", "high"][risk_type_code] if risk_type_code < 3 else "unknown"

    return {
        "input": q.dict(),
        "nearest_grid": {
            "lat_grid": float(nearest["lat_grid"]),
            "lon_grid": float(nearest["lon_grid"]),
            "nm_pol": str(nearest.get("nm_pol", "Unknown")),
            "top_crime_type": str(nearest.get("top_crime_type", "Unknown")),
            "total_crimes": int(nearest.get("total_crimes", 0)),
            "risk_type": str(nearest.get("risk_type", "unknown")),
            "risk_score_data": float(nearest.get("risk_score", 0)),
        },
        "prediction": {
            "risk_score": round(risk_score, 4),
            "risk_type": risk_type_label,
        },
    }
