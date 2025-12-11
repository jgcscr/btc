import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import Booster, DMatrix


app = FastAPI(title="BTC Forecast API", version="1.0.0")

MODEL: Optional[Booster] = None
FEATURE_NAMES: List[str] = []
TARGET_NAME: str = "ret_1h"

DIR_MODEL: Optional[Booster] = None
DIR_FEATURE_NAMES: List[str] = []
DIR_THRESHOLD: float = 0.0


class PredictRequest(BaseModel):
    instances: List[Dict[str, float]]


class PredictResponse(BaseModel):
    predictions: List[float]


class PredictDirectionResponse(BaseModel):
    probabilities: List[float]  # P(up)
    labels: List[int]           # 0 or 1
    threshold: float            # probability threshold used for labels


def load_model(model_dir: Optional[str] = None) -> None:
    """Load the XGBoost model and metadata from the given directory.

    If ``model_dir`` is not absolute, it is resolved relative to this file's
    directory, so that ``src/api/model`` works in both local runs and Docker.
    """
    global MODEL, FEATURE_NAMES, TARGET_NAME

    base_dir = os.path.dirname(__file__)
    if model_dir is None:
        resolved_dir = os.path.join(base_dir, "model")
    elif os.path.isabs(model_dir):
        resolved_dir = model_dir
    else:
        resolved_dir = os.path.join(base_dir, model_dir)

    model_path = os.path.join(resolved_dir, "xgb_ret1h_model.json")
    meta_path = os.path.join(resolved_dir, "model_metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise RuntimeError(f"Model or metadata not found in {resolved_dir}")

    # Use low-level Booster interface to avoid sklearn wrapper issues
    model = Booster()
    model.load_model(model_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = meta.get("feature_names", [])
    target_name = meta.get("target", "ret_1h")

    if not isinstance(feature_names, list) or not feature_names:
        raise RuntimeError("Invalid or empty feature_names in model_metadata.json")

    MODEL = model
    FEATURE_NAMES.clear()
    FEATURE_NAMES.extend(feature_names)
    TARGET_NAME = target_name


def load_direction_model(model_dir: Optional[str] = None) -> None:
    """Load the direction XGBoost model and metadata from the given directory.

    Uses the low-level Booster interface to stay consistent with regression
    serving and avoid sklearn wrapper issues.
    """
    global DIR_MODEL, DIR_FEATURE_NAMES, DIR_THRESHOLD

    base_dir = os.path.dirname(__file__)
    if model_dir is None:
        resolved_dir = os.path.join(base_dir, "model")
    elif os.path.isabs(model_dir):
        resolved_dir = model_dir
    else:
        resolved_dir = os.path.join(base_dir, model_dir)

    model_path = os.path.join(resolved_dir, "xgb_dir1h_model.json")
    meta_path = os.path.join(resolved_dir, "model_metadata_direction.json")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise RuntimeError(f"Direction model or metadata not found in {resolved_dir}")

    model = Booster()
    model.load_model(model_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = meta.get("feature_names", [])
    threshold = float(meta.get("threshold", 0.0))

    if not isinstance(feature_names, list) or not feature_names:
        raise RuntimeError("Invalid or empty feature_names in model_metadata_direction.json")

    DIR_MODEL = model
    DIR_FEATURE_NAMES.clear()
    DIR_FEATURE_NAMES.extend(feature_names)
    DIR_THRESHOLD = threshold


@app.on_event("startup")
def startup_event() -> None:
    # Load both regression and direction models at startup.
    load_model()
    load_direction_model()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "num_features": len(FEATURE_NAMES),
        "target": TARGET_NAME,
        "direction_model_loaded": DIR_MODEL is not None,
        "direction_num_features": len(DIR_FEATURE_NAMES),
        "direction_threshold_label": 0.5,
        "direction_label_from_ret_threshold": DIR_THRESHOLD,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not FEATURE_NAMES:
        raise HTTPException(status_code=500, detail="Feature names missing")

    if not req.instances:
        raise HTTPException(status_code=400, detail="No instances provided")

    rows: List[List[float]] = []
    for inst in req.instances:
        # Build feature row in the exact order expected by the model
        row = [float(inst.get(name, 0.0)) for name in FEATURE_NAMES]
        rows.append(row)

    X = np.asarray(rows, dtype=float)
    dmatrix = DMatrix(X, feature_names=FEATURE_NAMES)
    preds = MODEL.predict(dmatrix)
    return PredictResponse(predictions=preds.tolist())


@app.post("/predict_direction", response_model=PredictDirectionResponse)
def predict_direction(req: PredictRequest) -> PredictDirectionResponse:
    if DIR_MODEL is None:
        raise HTTPException(status_code=500, detail="Direction model not loaded")

    if not DIR_FEATURE_NAMES:
        raise HTTPException(status_code=500, detail="Direction feature names missing")

    if not req.instances:
        raise HTTPException(status_code=400, detail="No instances provided")

    rows: List[List[float]] = []
    for inst in req.instances:
        row = [float(inst.get(name, 0.0)) for name in DIR_FEATURE_NAMES]
        rows.append(row)

    X = np.asarray(rows, dtype=float)
    dmatrix = DMatrix(X, feature_names=DIR_FEATURE_NAMES)
    # Objective is binary:logistic, so predictions are probabilities for class 1 (up)
    proba_up = DIR_MODEL.predict(dmatrix).tolist()
    threshold = 0.5
    labels = [1 if p >= threshold else 0 for p in proba_up]

    return PredictDirectionResponse(
        probabilities=proba_up,
        labels=labels,
        threshold=threshold,
    )
