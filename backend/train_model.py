"""Train calibrated finding classifiers on REAL MIT-BIH annotations.
Labels come from expert .atr annotation files - nothing simulated:
  NSR        = 1 if every beat in window is 'N' and no abnormal rhythm active
  RHYTHM_ABN = 1 if an abnormal rhythm ((AFIB (AFL (SVTA (VT (IVR (SBR (B (T (WPW active
  ECTOPIC    = 1 if any beat in {V r R e E j J A a S F f} in window
BRADY / TACHY / QT / ST are NOT trained: they are deterministic clinical
thresholds applied to real measurements (HR<60, HR>100, QTc>=470, |ST|>=0.1mV).
Run: py -m backend.train_model   (needs internet once; ~3-6 min)
Output: backend/model_mitbih.joblib + backend/model_report.json
"""
import json
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_mitbih.joblib")
REPORT_PATH = os.path.join(BASE_DIR, "model_report.json")

TRAIN_RECORDS = ["100", "101", "102", "103", "104", "105", "106", "107",
                 "200", "201", "202", "203", "205", "208", "209", "210",
                 "212", "213", "214", "215", "217", "219", "222", "223",
                 "228", "230", "231", "232", "233", "234"]
TRAIN_MINUTES = 5
WIN_S, STEP_S = 10.0, 5.0

ECTOPIC_SYMS = set(list("VrReEjJAaSFf"))
ABN_RHYTHMS = {"(AFIB", "(AFL", "(SVTA", "(VT", "(IVR", "(SBR", "(B", "(T", "(WPW"}

FEATURES = ["heart_rate_bpm", "rr_mean_s", "rr_std_s", "rmssd_s",
            "regularity_cv", "qrs_ms", "qt_ms", "qtc_ms", "st_mv",
            "p_present_frac", "n_beats"]


def record_windows(record: str):
    import wfdb
    from .ecg_processing import apply_filter_pipeline
    from .features import extract_features
    dur = TRAIN_MINUTES * 60
    rec = wfdb.rdrecord(record, pn_dir="mitdb", physical=True, sampto=int(dur * 360))
    sig = np.array(rec.p_signal[:, 0], dtype=float)
    fs = float(rec.fs)
    ann = wfdb.rdann(record, "atr", pn_dir="mitdb", sampto=int(dur * 360))
    syms = list(ann.symbol)
    samps = list(ann.sample)
    aux = [(a or "").replace("\x00", "") for a in ann.aux_note]
    # rhythm state timeline: list of (start_sample, rhythm_or_None)
    rhythm_at = []
    cur = None
    for s, a in zip(samps, aux):
        if a.startswith("("):
            cur = a
            rhythm_at.append((s, cur))
        elif a.endswith(")") and cur is not None:
            rhythm_at.append((s, None))
            cur = None
    X, y = [], []
    n = len(sig)
    step = int(STEP_S * fs)
    win = int(WIN_S * fs)
    for start in range(0, n - win + 1, step):
        end = start + win
        idx = [i for i, s in enumerate(samps) if start <= s < end]
        bs = [syms[i] for i in idx]
        beats_only = [b for b in bs if b not in ["+", "~", '"', "|"]]
        if len(beats_only) < 3:
            continue
        seg = sig[start:end]
        filt = apply_filter_pipeline(seg, fs)
        feat = extract_features(filt, fs)
        # rhythm active in window: last change at/before end
        active = None
        for rs, r in rhythm_at:
            if rs < end:
                active = r
            else:
                break
        ect = int(any(b in ECTOPIC_SYMS for b in beats_only))
        rhy = int(active in ABN_RHYTHMS)
        sinus = (active is None) or (active == "(N")
        nsr = int(all(b == "N" for b in beats_only) and sinus and ect == 0)
        X.append([feat[k] for k in FEATURES])
        y.append([nsr, rhy, ect])
    return np.array(X), np.array(y)


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score, brier_score_loss
    import joblib

    Xs, ys = [], []
    used = []
    for r in TRAIN_RECORDS:
        try:
            X, y = record_windows(r)
        except Exception as e:
            print(f"[{r}] skip: {e}")
            continue
        print(f"[{r}] windows={len(X)} nsr={y[:,0].sum()} rhythm={y[:,1].sum()} ectopic={y[:,2].sum()}")
        if len(X) == 0:
            continue
        Xs.append(X)
        ys.append(y)
        used.append(r)
    X = np.vstack(Xs)
    Y = np.vstack(ys)
    # Guard: replace NaN/inf with column medians (same transform saved for inference)
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(X)
    X[bad] = np.take(med, np.where(bad)[1])
    print(f"TOTAL windows={len(X)}")
    targets = ["NSR", "RHYTHM_ABN", "ECTOPIC"]
    models, metrics = {}, {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for j, t in enumerate(targets):
        y = Y[:, j]
        print(f"--- {t}: pos={y.sum()}/{len(y)}")
        base = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                      class_weight="balanced_subsample",
                                      random_state=42, n_jobs=-1)
        # Metric on uncalibrated base (CV); shipped model adds Platt calibration
        from sklearn.base import clone
        proba = cross_val_predict(clone(base), X, y, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, proba))
        brier = float(brier_score_loss(y, proba))
        print(f"   5-fold AUROC={auc:.3f} Brier={brier:.4f}")
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        clf.fit(X, y)
        models[t] = clf
        metrics[t] = {"n": int(len(y)), "positives": int(y.sum()),
                      "auroc_5fold": round(auc, 4), "brier_5fold": round(brier, 5)}
    joblib.dump({"models": models, "features": FEATURES, "medians": med,
                 "train_records": used,
                 "brady_hr": 60.0, "tachy_hr": 100.0,
                 "qtc_thresh": 470.0, "st_thresh_mv": 0.1}, MODEL_PATH)
    report = {"model": "RandomForest(300, balanced_subsample) + sigmoid calibration (Platt), 5-fold CV",
              "features": FEATURES, "targets": targets,
              "windows": int(len(X)), "records": used,
              "metrics": metrics,
              "deterministic_criteria": {
                  "BRADY": "measured HR < 60 bpm (AHA definition)",
                  "TACHY": "measured HR > 100 bpm (AHA definition)",
                  "QT_PROLONG": "measured QTc (Bazett) >= 470 ms",
                  "ST_ABN": "measured |ST deviation| >= 0.1 mV"},
              "note": "All labels from MIT-BIH expert annotations; probabilities are calibrated classifier outputs, not heuristics."}
    open(REPORT_PATH, "w").write(json.dumps(report, indent=2))
    print(f"\nSaved {MODEL_PATH}\nSaved {REPORT_PATH}")


if __name__ == "__main__":
    main()
