# -*- coding: utf-8 -*-
"""
Model loader and predictor helpers.
Loads a scikit-learn model from disk and exposes a predict() helper.
"""
import os
import joblib
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Set MODEL_PATH env var or place model.pkl alongside app.")
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_batch(rows, feature_order=None):
    """
    rows: list[dict] or list[list/tuple]
    feature_order: explicit list of keys to extract from dicts (e.g., ["Age","EstimatedSalary"])
    """
    if len(rows) == 0:
        return []

    if isinstance(rows[0], dict):
        if feature_order is None:
            # default order (commonly used in the course)
            feature_order = ["Age", "EstimatedSalary"]
        X = np.array([[float(r[k]) for k in feature_order] for r in rows], dtype=float)
    else:
        X = np.array(rows, dtype=float)

    model = get_model()

    # Try proba; fallback to decision_function; fallback to predict
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, -1].tolist()
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X)
        # map decision function to 0-1 via logistic as a naive fallback
        import math
        proba = [1/(1+math.exp(-z)) for z in np.ravel(df)]
    preds = model.predict(X).tolist()

    return [{"prediction": int(p), "proba": (float(proba[i]) if proba is not None else None)} for i, p in enumerate(preds)]
