# WORK LOG — Probabilistic Randomized QUBO-Based ECG Analyzer
**Project folder:** `C:\Users\ADMIN\Desktop\PROBABIITY` — **Date:** 2026-10-06

## 1. Opened the PPT, created the plan
- Read `ECG_Probabilistic_Randomized_QUBO_Project_Presentation-1.pptx` (19 slides) with python-pptx.
- Delivered a phased build plan (MVP pipeline, tech stack, Phases 1-8, validation, hardware future, safety).

## 2. Built the complete web portal (first version)
**Backend (`backend/`, FastAPI + SciPy + WFDB + classical QUBO):**
- `ecg_processing.py` — Butterworth HP 0.5 Hz (ord 4) + 50 Hz notch (Q=30) + LP 45 Hz (ord 4), zero-phase `filtfilt` + SQI (SNR/saturation/flat-line).
- `features.py` — R-peak detection, HR, RR mean/std, RMSSD, regularity CV, QRS, QT/QTc (Bazett), ST, PR, P-presence.
- `probability.py` (v1) — rule-based sigmoid P(F|E) for 7 findings, labeled SIMULATED.
- `stochastic.py` (v1) — Gaussian perturbation sampling, N states, S(F)=N_F/N.
- `qubo.py` — E(x)=Σaᵢxᵢ+Σbᵢⱼxᵢxⱼ, exhaustive (n≤20) / simulated annealing.
- `mitbih_loader.py` (v1) — PhysioNet download + synthetic offline fallback.
- `report.py` — ReportLab PDF (waveform PNG + tables + disclaimers).
- `main.py` — REST API + serves frontend.
**Frontend (`frontend/index.html`, React 18 via CDN):** record/seconds/N/seed controls, CSV upload, waveform canvas, KPI cards, P/S/QUBO table, next steps, PDF download.
**Docs/hosting:** `docs/TECHNICAL_DOCUMENTATION.md` (patent), `docs/FILTER_VALIDATION.md`, `Dockerfile`, `docker-compose.yml`, `README.md`.
- Calibrated estimators on records 100/101/200/208; verified pipeline + PDF end-to-end.

## 3. Ran the portal, fixed startup problems
- Server must be started by the user (my background jobs don't persist): `py -m uvicorn backend.main:app --host 127.0.0.1 --port 8000` → `http://127.0.0.1:8000`.
- Usage guide delivered (workflow, how to read each section).

## 4. Doctor report upgrade
- New `backend/explain.py` — plain-language guide for all 12 measurements, 7 findings, 4 scores + auto-flags.
- PDF now explains every value (value + flag + normal ref + meaning); API returns `explanations` + `score_guide`.
- Fixed unreadable PDF glyphs (unicode → ASCII); verified 3-page PDF extracts cleanly, 0 bad characters.

## 5. Architecture review + honest REAL-vs-SIMULATED validation
- Verified all 5 routes via test client (all HTTP 200).
- Documented: filters/features/QUBO/stability-machinery/app = REAL; v1 rule-based P(F|E) = SIMULATED heuristic.

## 6. ZERO synthetic / ZERO simulated conversion (current phase)
- `mitbih_loader.py` (v2) — local disk cache (`data/mitbih_cache/`) first, PhysioNet download second; synthetic fallback **deleted**; missing data raises explicit error.
- `backend/prefetch.py` (new) — one-time cache downloader. Status: **8/24 records cached** (100,101,102,103,200,201,202,203); rest auto-download on first use.
- `backend/train_model.py` (new) — trains on REAL expert `.atr` annotations: 1770 windows / 30 records. **Results: NSR AUROC 0.988, RHYTHM_ABN 0.972, ECTOPIC 0.971** → `backend/model_mitbih.joblib` (19.7 MB) + `backend/model_report.json`.
- `probability.py` (v2) — trained calibrated RandomForest+Platt for NSR/Rhythm/Ectopic; BRADY/TACHY/QT/ST = deterministic clinical thresholds. Missing model file raises error (no silent fallback).
- `features.py` — added per-beat distributions (`rr_intervals_s`, `qrs/qt/st_per_beat`) for bootstrap.
- `stochastic.py` (v2) — nonparametric bootstrap over the record's own beats (no assumed sigmas).
- `main.py` — serves model metrics in every response (`meta.model`); health endpoint reports model + cache status.
- `report.py` / `explain.py` / `frontend/index.html` / `docs/*` / `README.md` — all relabeled from SIMULATED to trained/bootstrap language; physician-safety disclaimer kept.
- Requirements added: `scikit-learn`, `joblib`.

## 7. Final verification (all passed)
## 8. Hosting-ready + reload verification
## 10. Performance optimization pass (measured, all tests pass)
## 13. Professional 5-page clinical report (validated)
- New backend/clinical.py: dynamic conclusion (NSR never abnormal), controlled condition/next-step mappings, insight blocks. Fixed summarize_findings NSR mislabel.
- Rewrote backend/report.py: P1 summary+conclusion, P2 full-page B&W waveform (numbered R-peaks only), P3 measurements table, P4 analytical results + workflow + model performance, P5 automated insight + optional Gemini + review blanks + exact disclaimer + technical notes. Header/footer + Page X of Y on every page.
- main.py: clinical B&W waveform + units threading in report routes. No algorithm file touched (predictions/probabilities/bootstrap/QUBO identical).
- Validation 34/34 + regression 45/45 pass. Output: ECG_Analyzer_Professional_Report.pdf (5 pages, record 101).
## 12. Key-free AI alternatives (43/43 checks pass)
- `AI_PROVIDER` env: gemini (default) | ollama (local LLM, no key/quota) | template (deterministic writer, always works, labeled "no LLM").
- Template restates QUBO findings + explanations + urgent flags; never invents associations. Ollama calls localhost:11434 with JSON mode + timeout, graceful hint if not running.
- Suite now 43 checks (template/ollama-down/fallback tests). Docs updated (.env.example, README).
## 11. Gemini 404 root cause + fix (verified against live API)
- Root cause: retired model IDs. Google API verbatim: gemini-2.0-flash "no longer available", gemini-2.5-flash/pro "no longer available to new users" -> use gemini-3.8-flash.
- Default model now gemini-3.8-flash; automatic fallback gemini-3.7-flash,gemini-3.6-flash on server/quota/timeout (answering model recorded in output).
- Errors classified separately (missing_key/auth/not_found/quota/server/timeout/network/request/bad_response), sanitized, key never exposed.
- New backend/gemini_doctor.py (key length only, model list, live test). Test suite 39/39 pass incl. 503-fallback simulation.
- Baseline (rec 208/10 s/N=100): TOTAL 70.7 s - bootstrap scoring 68.7 s, single inference 3.2 s, qubo 2.1 ms, plot 560 ms, pdf 309 ms.
- `probability.py`: batched inference (3 model calls per N states, was 3N) - identical outputs; fixed unclosed file handle.
- `stochastic.py`: same RNG sampling loop + batched scoring (bootstrap 68.7 s -> 0.6 s, ~110x).
- `qubo.py`: removed dead buggy `energy()`; vectorized exhaustive search, same order/tie-break (tested vs brute force incl. ties).
- `mitbih_loader.py`: in-memory LRU record cache (repeat loads 11 ms -> 0.3 ms).
- `report.py`: waveform dpi 150->110 + PNG optimize (plot 560->374 ms, pdf 309->146 ms); temp dirs cleaned up.
- `main.py`: per-stage `timings` in every response + server log; startup model warmup (lifespan); plot/pdf/gemini timed on report routes.
- `gemini_client.py`: official `google-genai` SDK, single call/report, `GEMINI_TIMEOUT_MS` (30 s default), graceful fallback; PDF heading now "AI-Assisted Interpretation - Requires Clinician Review".
- New `backend/timing.py`, `backend/test_optimize.py` (31 checks pass). Deps added: `google-genai`.
- After: steady-state TOTAL ~1.2 s (57x). Remaining bottleneck: RF predict itself (~1.2 s, needs retrain to reduce - not done, outputs preserved).
## 9. Gemini AI interpretation + doctor-style PDF (all passed)
- New `backend/gemini_client.py`: key from `GEMINI_API_KEY` env only; grounded prompt (measured data only + 6 guardrails); deterministic urgent flags + normal/abnormal/uncertain split; never raises.
- `report.py`: official header (provided data or "Not provided", never invented), urgent banner, training summary, explained measurements, findings, deterministic overall summary, Gemini section (or "unavailable" notice), blank doctor-review fields, exact disclaimer, timestamps + page numbers, UNREVIEWED footer.
- `main.py`: patient fields + `ai` flag on report endpoints, Gemini called AFTER QUBO, new `GET /api/ai_insights` (key stays server-side).
- Frontend: facility/patient inputs, AI toggle, "Preview AI insights" card. `.env.example`, `.gitignore` covers `.env`, README documents setup.
- Tests: prompt grounding, no-key/invalid-key graceful paths, PDF content (4 pages, no key material), routes - ALL PASSED.
- `Dockerfile` now honors `$PORT` (Render/Railway compatible); added `render.yaml` (one-click Render deploy) and `.dockerignore` (slim image; MIT-BIH cache downloads on demand at runtime).
- Gave user two routes: free Render internet hosting (GitHub + Render steps) vs local-network access (`--host 0.0.0.0`).
- Reload check passed: health ok (model True, 24 cached), analyze 200, PDF 200.
- MIT-BIH cache: 24/24 records (local `data/mitbih_cache` + D:/ overflow since C: is full).
- Full route test: `/`, `/api/health` (model_trained=True), `/api/records`, `/api/analyze_mitbih` (records 100/208/233), `/api/report_mitbih` (PDF ~164KB) - all HTTP 200.
- Record 208 result: ECTOPIC P=0.806 [trained-ml] S=0.98 QUBO=1; QT_PROLONG P=0.99 [measurement-criterion] - real model outputs.
- Grep sweep: no synthetic data paths remain (only history notes + the legitimate "simulated annealing" optimizer name).
