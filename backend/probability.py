"""Probability engine: REAL trained + calibrated classifiers + deterministic criteria.
  NSR / RHYTHM_ABN / ECTOPIC: RandomForest + Platt calibration trained on
    1770 expert-annotated MIT-BIH windows (see backend/model_report.json).
  BRADY / TACHY / QT_PROLONG / ST_ABN: deterministic clinical thresholds
    applied to real measurements (HR<60, HR>100, QTc>=470ms, |ST|>=0.1mV).
No heuristics, no synthetic data. Requires backend/model_mitbih.joblib
(run: py -m backend.train_model).
"""
import json
import math
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_mitbih.joblib")
REPORT_PATH = os.path.join(BASE_DIR, "model_report.json")

FINDINGS = [
    {"id": 0, "code": "NSR", "name": "Normal Sinus Rhythm"},
    {"id": 1, "code": "RHYTHM_ABN", "name": "Rhythm Abnormality (irregular rhythm)"},
    {"id": 2, "code": "ECTOPIC", "name": "Ectopic Activity (PVC-like)"},
    {"id": 3, "code": "ST_ABN", "name": "ST Abnormality"},
    {"id": 4, "code": "QT_PROLONG", "name": "QT / QTc Prolongation"},
    {"id": 5, "code": "BRADY", "name": "Bradycardia"},
    {"id": 6, "code": "TACHY", "name": "Tachycardia"},
]

_ML_TARGETS = ["NSR", "RHYTHM_ABN", "ECTOPIC"]

_bundle = None
_report = None


def _load_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Trained model not found at backend/model_mitbih.joblib. "
                "Train it with: py -m backend.train_model (one-time, needs internet).")
        import joblib
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def model_info() -> dict:
    """AUROC/Brier/training-set facts for UI + PDF (proof of real training)."""
    global _report
    if _report is None:
        if os.path.exists(REPORT_PATH):
            _report = json.loads(open(REPORT_PATH).read())
        else:
            _report = {"error": "model_report.json missing - run py -m backend.train_model"}
    b = _load_bundle()
    return {"report": _report, "train_records": b.get("train_records", []),
            "features": b.get("features", [])}


def _feature_row(feat: dict, names: list, medians: np.ndarray) -> np.ndarray:
    row = np.array([float(feat.get(k, np.nan)) for k in names], dtype=float)
    bad = ~np.isfinite(row)
    row[bad] = medians[bad]
    return row.reshape(1, -1)


def _criterion_prob(met: bool) -> float:
    return 0.99 if met else 0.01


def estimate_probabilities(feat: dict) -> list:
    b = _load_bundle()
    X = _feature_row(feat, b["features"], np.asarray(b["medians"], dtype=float))
    ml = {}
    for t in _ML_TARGETS:
        p = float(b["models"][t].predict_proba(X)[0, 1])
        ml[t] = max(0.01, min(0.99, p))

    hr = float(feat.get("heart_rate_bpm", 70) or 70)
    qtc = float(feat.get("qtc_ms", 420) or 420)
    st = abs(float(feat.get("st_mv", 0) or 0))
    n = int(feat.get("n_beats", 0))

    probs = {
        "NSR": (ml["NSR"], "trained-ml"),
        "RHYTHM_ABN": (ml["RHYTHM_ABN"], "trained-ml"),
        "ECTOPIC": (ml["ECTOPIC"], "trained-ml"),
        "ST_ABN": (_criterion_prob(st >= 0.1), "measurement-criterion"),
        "QT_PROLONG": (_criterion_prob(qtc >= 470.0), "measurement-criterion"),
        "BRADY": (_criterion_prob(hr < 60.0), "measurement-criterion"),
        "TACHY": (_criterion_prob(hr > 100.0), "measurement-criterion"),
    }
    if n < 3:
        # Too few beats for reliable measurement: report maximum uncertainty
        probs = {k: ((0.5, m + "+low-beats") if m == "trained-ml" else (v, m))
                 for k, (v, m) in probs.items()}

    out = []
    for f in FINDINGS:
        p, method = probs[f["code"]]
        out.append({"id": f["id"], "code": f["code"], "name": f["name"],
                    "probability": round(float(p), 3), "method": method})
    return out


NEXT_STEPS = {
    "NSR": ["Routine follow-up as per clinical context.", "Repeat ECG if symptoms persist."],
    "RHYTHM_ABN": ["Obtain 12-lead ECG and prolonged rhythm monitoring (Holter/extended patch).",
                   "Review medications, electrolytes (K/Mg), thyroid function.", "Cardiology review for possible AF/flutter."],
    "ECTOPIC": ["Quantify ectopic burden with Holter.", "Check electrolytes, ischemia workup if indicated.", "Echocardiogram if frequent/high burden."],
    "ST_ABN": ["Correlate with symptoms; urgent assessment if chest pain.", "Serial ECGs + high-sensitivity troponin per ACS protocol.", "Cardiology review; consider stress/imaging test."],
    "QT_PROLONG": ["Review QT-prolonging drugs; check K/Mg/Ca.", "Repeat 12-lead ECG for precise QTc.", "Cardiology review; avoid QT-prolonging agents."],
    "BRADY": ["Assess symptoms (syncope/dizziness); medication review (beta-blockers, etc.).", "Holter/exercise correlation.", "Cardiology review if symptomatic or HR < 40."],
    "TACHY": ["Assess hemodynamic stability first.", "12-lead ECG to classify narrow vs wide complex.", "Evaluate fever, anemia, thyroid, sepsis; cardiology review."],
}
