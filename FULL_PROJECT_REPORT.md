# FULL PROJECT REPORT — Probabilistic Randomized QUBO-Based ECG Analyzer
*Everything about this project, written so you can learn it from zero.*

---

## 1. What this project is (one paragraph)
A **web portal** where a doctor (or student) picks a heart recording (ECG), and the system cleans the signal, measures it, predicts which heart findings are present, checks how robust each prediction is, resolves conflicts between findings with math called **QUBO**, and produces an **explainable doctor-style PDF report**. It is a **research prototype for physician assistance — never an autonomous diagnosis device**.

## 2. The big picture: data flow
```
MIT-BIH heart records (real, from PhysioNet)
        |
        v
[STEP 1] Cleaning (filters) ---> Signal Quality check
        |
        v
[STEP 2] Measurements (features: heart rate, QRS, QT, ST, ...)
        |
        v
[STEP 3] Fast gate (obvious cases pass, hard cases go deeper)
        |
        v
[STEP 4] Prediction P(F|E): trained ML model (3 findings)
        + fixed medical rules (4 findings)
        |
        v
[STEP 5] Robustness check: bootstrap resampling ---> Stability score S
        |
        v
[STEP 6] Conflict resolution: QUBO optimization (picks final set)
        |
        v
[STEP 7] Explanation + dashboard + PDF report
```

## 3. Architecture: what runs where
- **Backend (Python, FastAPI)** in `backend/`: all the science. Exposes a REST API (`/api/...`) and also serves the website itself.
- **Frontend (React)** in `frontend/index.html`: the dashboard you see in the browser. It only displays; it calls the backend API for every result.
- **Data**: MIT-BIH records cached on disk (`data/mitbih_cache` + D:/E: overflow); trained model file (`backend/model_mitbih.joblib`, ~20 MB).
- **How they connect**: browser -> HTTP (`http://127.0.0.1:8000`) -> FastAPI -> Python pipeline -> JSON back -> React renders it. PDF is generated server-side and downloaded.

## 4. Every backend file, explained
| File | Job | Key technique (real tech background) |
|---|---|---|
| `ecg_processing.py` | Removes noise, keeps heartbeat shape | **Butterworth filters** (maximally flat frequency response, standard in biomedical DSP) + **50 Hz notch** (kills power-line hum) + **filtfilt** (runs filter forward+backward = zero phase shift, so R-peaks don't move) |
| `features.py` | Finds beats, measures 12 values | **R-peak detection** (derivative -> square -> moving average -> adaptive threshold, Pan-Tompkins family); HR from RR intervals; **RMSSD** (standard HRV metric); QRS/QT/ST morphology estimates; **Bazett formula** QTc = QT/sqrt(RR) |
| `probability.py` | P(F\|E): how likely each finding is | **RandomForest (300 trees) + Platt sigmoid calibration**, trained on 1770 expert-annotated windows (NSR AUROC 0.988, Rhythm 0.972, Ectopic 0.971). BRADY/TACHY/QT/ST = fixed clinical cutoffs (HR<60, HR>100, QTc>=470ms, \|ST\|>=0.1mV) |
| `stochastic.py` | "Would the result survive small changes?" | **Nonparametric bootstrap**: resample the patient's own beats with replacement x100 states, recompute, recount. No invented noise numbers |
| `qubo.py` | Picks the final consistent finding set | **QUBO**: minimize E(x)=sum a_i x_i + sum b_ij x_i x_j. Evidence in linear terms, conflicts (e.g. Normal-vs-abnormal, brady-vs-tachy) in quadratic penalties. Solved exactly (exhaustive, n<=20) or by **simulated annealing** |
| `mitbih_loader.py` | Gets real heart data | **WFDB** library + PhysioNet; disk cache first, download second; raises error if unavailable (no fake data anywhere) |
| `train_model.py` | Trains the ML model (run once) | Reads MIT-BIH `.atr` expert annotations, builds labeled 10 s windows, trains + 5-fold evaluates, saves model + `model_report.json` |
| `explain.py` | Doctor-level explanations | Normal ranges + meaning + auto-flags for every value |
| `report.py` | PDF report | ReportLab: training-metrics table, explained measurements, waveform plot (matplotlib), findings, next steps, method + disclaimer |
| `main.py` | API + wiring | FastAPI routes: `/api/health`, `/api/records`, `/api/analyze_mitbih`, `/api/analyze_signal`, `/api/upload_csv`, `/api/report_mitbih`, `/api/report_signal`; serves frontend |
| `prefetch.py` | One-time downloader | Caches records to disk so the portal works offline |

## 5. The 7 findings (what each means)
1. **NSR** — normal rhythm (all beats normal, regular). 2. **RHYTHM_ABN** — irregular rhythm (possible AF etc.; needs Holter to classify). 3. **ECTOPIC** — extra/wide beats (PVC-like; quantify burden). 4. **ST_ABN** — ST shifts (possible ischemia; urgent if chest pain). 5. **QT_PROLONG** — long QTc (arrhythmia risk; check drugs/electrolytes). 6. **BRADY** — heart rate <60. 7. **TACHY** — heart rate >100.

## 6. The 3 scores (don't confuse them)
- **P (probability)**: how strongly evidence supports the finding (trained model or fixed rule).
- **S (stability)**: fraction of 100 resampled states still supporting it = robustness, NOT certainty.
- **QUBO x (0/1)**: final keep/drop after conflict resolution. Strongest signal = high P + high S + selected.

## 7. Honest limits (important for learning)
- The ML model learned from MIT-BIH annotations (real), but it is still a **research model**: single-lead, one dataset family, no prospective clinical trial, no regulatory approval. Hence "physician assistance only."
- QRS/QT/ST are heuristic estimates (good for screening, not 12-lead precision).
- The QUBO conflict weights are engineering choices, not learned.

## 8. How to run / host it
```powershell
cd C:\Users\ADMIN\Desktop\PROBABIITY
py -m pip install -r backend\requirements.txt
py -m backend.prefetch        # one-time: cache records (needs internet)
py -m backend.train_model     # one-time: train model (~5 min) - already done
py -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```
Internet hosting: push folder to GitHub -> Render Web Service from `render.yaml`/`Dockerfile` -> public URL. Local network: `--host 0.0.0.0`, open `http://YOUR-PC-IP:8000` from other devices.

## 9. Key vocabulary
**ECG** (heart's electrical trace) • **QRS** (ventricular spike) • **RR/HR** (beat spacing/rate) • **HRV/RMSSD** (beat-to-beat variation) • **QTc** (rate-corrected recovery time) • **ST** (injury/ischemia marker) • **AUROC** (ranking quality 0-1) • **Brier** (probability error, lower=better) • **Calibration/Platt** (making outputs true probabilities) • **Bootstrap** (resampling data to measure stability) • **QUBO** (binary optimization) • **WFDB/PhysioNet** (real open ECG database) • **filtfilt** (zero-phase filtering).

## 10. Suggested learning path
1. Open the dashboard, analyze records 100 (normal-ish), 208 (abnormal), 233 — compare P/S/QUBO. 2. Read `ecg_processing.py` + `docs/FILTER_VALIDATION.md` (signal cleaning). 3. Read `features.py` (measurements). 4. Read `model_report.json` + `train_model.py` (how the ML learned). 5. Read `qubo.py` + `stochastic.py` (decisions + robustness). 6. Re-train the model yourself (`py -m backend.train_model`). 7. Next level: validate on PTB-XL, try 12-lead data, replace RandomForest with a neural net.
