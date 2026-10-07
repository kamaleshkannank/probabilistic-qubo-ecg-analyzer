"""FastAPI backend for Probabilistic Randomized QUBO ECG Analyzer."""
import os
import tempfile
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, Query, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .ecg_processing import apply_filter_pipeline, signal_quality_index, DEFAULT_FS
from .features import extract_features
from .probability import estimate_probabilities, NEXT_STEPS
from .stochastic import stability_analysis
from .qubo import build_qubo, solve_qubo
from .mitbih_loader import load_mitbih, AVAILABLE_RECORDS, is_cached
from .probability import estimate_probabilities, NEXT_STEPS, model_info
from .gemini_client import interpret as gemini_interpret, detect_urgent, summarize_findings
from .explain import VALUE_GUIDE, FINDING_GUIDE, SCORE_GUIDE, flag_measurement
from .timing import stage, log_pipeline
from .report import waveform_png, build_pdf, waveform_clinical

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = FastAPI(title="Probabilistic Randomized QUBO ECG Analyzer",
              description="Research prototype - physician assistance only. Trained-model results with methods labeled.",
              version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _warmup():
    """Preload model + spin up sklearn thread pools so request #1 is fast."""
    try:
        from .probability import _load_bundle
        b = _load_bundle()
        X = np.asarray([b["medians"]], dtype=float)
        for t in ("NSR", "RHYTHM_ABN", "ECTOPIC"):
            b["models"][t].predict_proba(X)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warmup()
    yield


app.router.lifespan_context = lifespan


def _np_clean(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _np_clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_np_clean(v) for v in o]
    return o


def run_pipeline(signal: np.ndarray, fs: float, meta_extra=None, n_states=100, seed=42):
    timings = {}
    with stage(timings, "asarray"):
        signal = np.asarray(signal, dtype=float)
    with stage(timings, "preprocess"):
        filtered = apply_filter_pipeline(signal, fs)
    with stage(timings, "sqi"):
        sqi = signal_quality_index(signal, filtered)
    with stage(timings, "features"):
        feat = extract_features(filtered, fs)
    with stage(timings, "inference"):
        probs = estimate_probabilities(feat)
    with stage(timings, "bootstrap"):
        _, stab = stability_analysis(feat, probs, n_states=n_states, seed=seed)
    stab_map = {s["code"]: s for s in stab}
    with stage(timings, "qubo"):
        qubo = build_qubo(probs)
        sol = solve_qubo(qubo["a"], qubo["B"], seed=seed)
    enriched = []
    for p, x in zip(probs, sol["solution"]):
        s = stab_map.get(p["code"], {})
        enriched.append({**p, "stability": s.get("stability"),
                         "support": s.get("support"), "qubo_selected": x,
                         "next_steps": NEXT_STEPS.get(p["code"], []),
                         "explanation": FINDING_GUIDE.get(p["code"], "")})
    # Per-value explanations for doctor report / dashboard
    explanations = {}
    for k, v in feat.items():
        if k in ("r_peaks", "fs"):
            continue
        g = VALUE_GUIDE.get(k)
        if g:
            title, normal, interp = g
            explanations[k] = {"title": title, "value": v,
                               "normal": normal, "meaning": interp,
                               "flag": flag_measurement(k, v)}
    # Downsample for frontend plotting
    max_pts = 3000
    step = max(1, len(signal) // max_pts)
    info = model_info()
    _hide = {"r_peaks", "rr_intervals_s", "qrs_per_beat_ms",
             "qt_per_beat_ms", "st_per_beat_mv"}
    out = {
        "meta": {**(meta_extra or {}), "fs": fs, "n_samples": len(signal),
                 "sqi": sqi, "model": info["report"],
                 "notice":
                 "Trained ML probabilities (MIT-BIH) + deterministic criteria. Research prototype - physician assistance only, not a clinical diagnosis."},
        "features": {k: v for k, v in feat.items() if k not in _hide},
        "r_peaks": feat["r_peaks"],
        "probabilities": enriched,
        "explanations": explanations,
        "score_guide": {k: {"title": v[0], "what": v[1], "how": v[2]}
                        for k, v in SCORE_GUIDE.items()},
        "qubo": {"codes": qubo["codes"], "a": qubo["a"], "B": qubo["B"],
                 "solution": sol["solution"], "energy": sol["energy"], "method": sol["method"],
                 "formula": "E(x) = sum a_i x_i + sum b_ij x_i x_j"},
        "stability": stab,
        "waveform": {"raw": signal[::step].tolist(), "filtered": filtered[::step].tolist(),
                     "step": step, "fs": fs},
        "timings": timings,
    }
    log_pipeline(timings, f"n={len(signal)} n_states={n_states}")
    return _np_clean(out), (signal, filtered, feat, enriched)


@app.get("/api/health")
def health():
    cached = sum(1 for r in AVAILABLE_RECORDS if is_cached(r))
    try:
        info = model_info()
        model_ok = True
    except Exception:
        model_ok = False
        info = {}
    return {"status": "ok", "model_trained": model_ok,
            "mitbih_cached_records": cached,
            "model": info.get("report", {}) if isinstance(info, dict) else {}}


@app.get("/api/records")
def records():
    return {"records": AVAILABLE_RECORDS, "source": "MIT-BIH Arrhythmia Database (PhysioNet mitdb)",
            "cached": [r for r in AVAILABLE_RECORDS if is_cached(r)]}


@app.get("/api/analyze_mitbih")
def analyze_mitbih(record: str = "100", seconds: float = Query(10.0, ge=4, le=60),
                   n_states: int = Query(100, ge=20, le=500), seed: int = 42):
    data = load_mitbih(record, seconds)
    result, _ = run_pipeline(data["signal"], data["fs"], meta_extra={
        "record": data["record"], "source": data["source"],
        "dataset": "MIT-BIH Arrhythmia Database", "from_cache": data["cached"]},
        n_states=n_states, seed=seed)
    return JSONResponse(result)


class SignalIn(BaseModel):
    signal: list
    fs: float = 360.0
    label: str = "custom"
    n_states: int = 100
    seed: int = 42
    clinic_name: str = ""
    patient_id: str = ""
    patient_name: str = ""
    age: str = ""
    sex: str = ""


def _patient(record_id: str, clinic_name="", patient_id="", patient_name="",
             age="", sex="") -> dict:
    return {"clinic_name": clinic_name, "patient_id": patient_id,
            "patient_name": patient_name, "age": age, "sex": sex,
            "record_id": record_id}


def _with_ai(feat: dict, enriched: list, qubo_codes: list, use_ai: bool):
    """Gemini interpretation AFTER QUBO. Never raises; unavailable -> marked."""
    urgent = detect_urgent({k: v for k, v in feat.items()}, enriched)
    summary = summarize_findings(enriched)
    ai = gemini_interpret(
        {k: v for k, v in feat.items()}, enriched, qubo_codes, urgent) if use_ai else {
        "available": False, "reason": "AI interpretation not requested."}
    return urgent, summary, ai


@app.post("/api/analyze_signal")
def analyze_signal(body: SignalIn):
    sig = np.array(body.signal, dtype=float)
    result, _ = run_pipeline(sig, body.fs, meta_extra={"record": body.label, "source": "user signal"},
                             n_states=body.n_states, seed=body.seed)
    return JSONResponse(result)


@app.post("/api/upload_csv")
async def upload_csv(file: UploadFile = File(...), fs: float = 360.0):
    raw = await file.read()
    import numpy as np
    try:
        txt = raw.decode("utf-8", errors="ignore")
        arr = np.loadtxt(txt.splitlines(), delimiter=",", ndmin=1)
    except Exception:
        arr = np.frombuffer(raw, dtype=np.float64)
    arr = arr[~np.isnan(arr)].astype(float)
    result, _ = run_pipeline(arr, fs, meta_extra={"record": file.filename, "source": "CSV upload"})
    return JSONResponse(result)


@app.get("/api/report_mitbih")
def report_mitbih(record: str = "100", seconds: float = 10.0, n_states: int = 100, seed: int = 42,
                 clinic_name: str = "", patient_id: str = "", patient_name: str = "",
                 age: str = "", sex: str = "", ai: bool = True):
    data = load_mitbih(record, seconds)
    result, (sig, filt, feat, enriched) = run_pipeline(
        data["signal"], data["fs"],
        meta_extra={"record": data["record"], "source": data["source"],
                    "dataset": "MIT-BIH Arrhythmia Database",
                    "filter": "HP 0.5Hz + notch 50Hz Q30 + LP 45Hz (filtfilt)",
                    "qubo_solver": "classical", "stochastic": f"bootstrap N={n_states}, seed={seed}"},
        n_states=n_states, seed=seed)
    qubo_codes = [c for c, x in zip(result["qubo"]["codes"], result["qubo"]["solution"]) if x == 1]
    timings = dict(result.get("timings", {}))
    with stage(timings, "gemini"):
        urgent, summary, ai_out = _with_ai(feat, enriched, qubo_codes, ai)
    tmp = tempfile.mkdtemp()
    try:
        png = os.path.join(tmp, "wave.png")
        with stage(timings, "plot"):
            waveform_clinical(filt, data["fs"], feat["r_peaks"],
                              "mV" if data.get("units") == "mV" else "signal units", png)
        with stage(timings, "pdf"):
            pdf = build_pdf({"meta": {**result["meta"], "units": data.get("units", "")},
                             "features": result["features"],
                             "probabilities": enriched, "waveform_png": png,
                             "r_peaks": feat["r_peaks"],
                             "model": result["meta"].get("model", {}),
                             "patient": _patient(data["record"], clinic_name, patient_id,
                                                 patient_name, age, sex),
                             "urgent": urgent, "summary": summary, "ai": ai_out})
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    log_pipeline(timings, f"report_mitbih record={record}")
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=ECG_report_{record}.pdf"})


@app.post("/api/report_signal")
def report_signal(body: SignalIn, ai: bool = True):
    sig = np.array(body.signal, dtype=float)
    result, (s, f, feat, enriched) = run_pipeline(
        sig, body.fs, meta_extra={"record": body.label, "source": "user signal",
                                  "filter": "HP 0.5Hz + notch 50Hz Q30 + LP 45Hz"},
        n_states=body.n_states, seed=body.seed)
    qubo_codes = [c for c, x in zip(result["qubo"]["codes"], result["qubo"]["solution"]) if x == 1]
    timings = dict(result.get("timings", {}))
    with stage(timings, "gemini"):
        urgent, summary, ai_out = _with_ai(feat, enriched, qubo_codes, ai)
    tmp = tempfile.mkdtemp()
    try:
        png = os.path.join(tmp, "wave.png")
        with stage(timings, "plot"):
            waveform_clinical(f, body.fs, feat["r_peaks"], "signal units", png)
        with stage(timings, "pdf"):
            pdf = build_pdf({"meta": {**result["meta"], "units": "signal units"},
                             "features": result["features"],
                             "probabilities": enriched, "waveform_png": png,
                             "r_peaks": feat["r_peaks"],
                             "model": result["meta"].get("model", {}),
                             "patient": _patient(body.label, body.clinic_name, body.patient_id,
                                                 body.patient_name, body.age, body.sex),
                             "urgent": urgent, "summary": summary, "ai": ai_out})
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    log_pipeline(timings, "report_signal")
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=ECG_report.pdf"})


@app.get("/api/ai_insights")
def ai_insights(record: str = "100", seconds: float = Query(10.0, ge=4, le=60),
                n_states: int = Query(100, ge=20, le=500), seed: int = 42):
    """Pipeline + Gemini insights as JSON (for dashboard preview). Key stays server-side."""
    data = load_mitbih(record, seconds)
    result, (_, _, feat, enriched) = run_pipeline(
        data["signal"], data["fs"], meta_extra={"record": data["record"]},
        n_states=n_states, seed=seed)
    qubo_codes = [c for c, x in zip(result["qubo"]["codes"], result["qubo"]["solution"]) if x == 1]
    urgent, summary, ai_out = _with_ai(feat, enriched, qubo_codes, True)
    ai_out = dict(ai_out)
    ai_out.pop("reason", None) if ai_out.get("available") else None
    return JSONResponse(_np_clean({"urgent": urgent, "summary": summary, "ai": ai_out,
                                   "qubo_selected": qubo_codes}))


# Serve React/static frontend at / (must be last)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
