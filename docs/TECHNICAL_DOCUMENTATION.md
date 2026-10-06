# Probabilistic Randomized QUBO-Based ECG Analyzer — Technical Documentation
**For patent research / defensive disclosure. Research prototype. Date: 2026-10-06.**

## 1. Invention summary
Fast, explainable, uncertainty-aware ECG doctor-assistance system combining: ECG signal processing → fast analyzer → probability engine P(F|E) → controlled stochastic sampling (N plausible states) → QUBO optimization E(x)=Σaᵢxᵢ+Σbᵢⱼxᵢxⱼ → stability S(F)=N_F/N → explainable report (waveform, measurements, probability, stability, SQI, evidence).

## 2. Novelty elements (for prior-art search)
1. **Uncertainty-as-sampling:** measured feature → parametric uncertainty distribution → N sampled plausible ECG states, each re-scored; stability reported separately from probability.
2. **QUBO reconciliation layer:** candidate findings as binary variables; linear terms from evidence/probability, quadratic terms encode compatibility/conflict penalties (e.g., NSR vs abnormal, brady vs tachy); classical solver now, annealer/QPU later.
3. **Two-score reporting:** P (likelihood given evidence) + S (robustness under variation) + signal quality + morphology evidence + model version.
4. **Fast-path gating:** low-latency analyzer routes clear cases to output, uncertain cases to full stochastic+QUBO pipeline.
5. **Morphology-preserving denoising:** HP 0.5Hz + 50Hz notch Q30 + LP 45Hz, all zero-phase filtfilt.

## 3. System architecture
- Input: CSV/MAT/XML (prototype), MIT-BIH via WFDB; 12-lead + AFE/ADC/MCU future.
- Backend: Python FastAPI; NumPy/SciPy; scikit-learn/PyTorch optional; classical QUBO (exhaustive ≤20 vars else simulated annealing); SQLite→PostgreSQL.
- Frontend: React dashboard (this build: CDN React served by FastAPI; Vite source ready for Vercel/Netlify).
- Report: PDF with waveform PNG (matplotlib), measurements, P/S/QUBO, next steps, disclaimers.

## 4. Signal processing (validated settings)
- HP Butterworth 0.5Hz ord4 (baseline wander), notch 50Hz Q30 (power-line), LP Butterworth 45Hz ord4 (EMG/HF). Zero-phase filtfilt → no phase distortion of QRS/ST/T.
- Diagnostic bandwidth note: AHA diagnostic 0.05–150Hz; monitoring 0.5–40Hz. Our 0.5–45Hz is monitoring-grade; preserves QRS width, ST level, QT for screening while removing drift/noise. 12-lead diagnostic mode would widen to 0.05–150Hz.
- SQI: SNR (signal vs residual), saturation fraction, flat-line flag → Good/Acceptable/Poor.

## 5. Features, probability, stochastic, QUBO (all real - zero simulated)
- Features: R-peaks (differentiate->square->120ms MA->adaptive threshold->+-80ms refine), HR, RR mean/std, RMSSD, regularity CV, QRS width, QT/QTc (Bazett), ST deviation, P-presence fraction - plus per-beat distributions for bootstrap.
- P(F|E): calibrated RandomForest (300 trees, balanced subsample + Platt sigmoid calibration) trained on 1770 expert-annotated MIT-BIH windows from 30 records (10 s windows, 5 s step, first 5 min/record). 5-fold CV: NSR AUROC 0.988 / Brier 0.0377; RHYTHM_ABN AUROC 0.972 / Brier 0.0573; ECTOPIC AUROC 0.971 / Brier 0.0653 (see backend/model_report.json). BRADY/TACHY/QT/ST are deterministic clinical thresholds (HR<60, HR>100, QTc>=470 ms, |ST|>=0.1 mV) on real measurements.
- Uncertainty: nonparametric bootstrap resampling the record's own beats (no assumed noise model), N=100 default, seed 42 -> reproducible. Each state re-scored by the trained model; S(F)=N_F/N at threshold 0.5.
- QUBO: a_i=-(p_i-0.5)*4; conflicts NSR<->each abnormal (3-5), brady<->tachy 6; rhythm+ectopic -0.5 (co-occurrence). Exhaustive <=20 vars else SA (T 2->1e-3, alpha .995, 8000 steps).

## 6. Validation plan
Metrics: sensitivity, specificity, precision, F1, AUROC, calibration (ECE/Brier), FP/FN rates, robustness (noise/lead-shift), latency, stability distribution. Datasets: MIT-BIH (this build), then PTB-XL, INCART, private 12-lead. Phases per PPT §14.

## 7. Safety / regulatory
Research prototype, physician-assistance only. Never present P as diagnosis or S as certainty. UI + PDF carry disclaimers. Future device path: IEC 60601, ISO 13485, FDA 510(k)/De Novo or EU MDR — out of scope for this build.

## 8. Reproducibility
Seeded RNG, pinned requirements, WFDB caching, `seed` + `n_states` exposed in API/UI/PDF. Run: `pip install -r backend/requirements.txt` + `uvicorn backend.main:app`.

## 9. Commercial / hardware future
Edge: 12-lead → AFE → ADC → MCU/edge + AI accelerator + QUBO accelerator → doctor interface. Models: SaaS, SDK/API, OEM, embedded licensing. Targets: hospitals, clinics, labs, ECG OEMs, digital health, research.

## 10. Prior-art keywords
ECG QUBO, quadratic unconstrained binary optimization arrhythmia, stochastic ECG sampling stability, uncertainty-aware ECG, explainable ECG triage, probabilistic ECG findings reconciliation.
