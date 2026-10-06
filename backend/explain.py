"""Doctor-report explanations for every value in the ECG report.
Each entry: normal range, what it means, how to read abnormal values.
Informational only - not medical advice."""

VALUE_GUIDE = {
    "heart_rate_bpm": ("Heart rate (bpm)",
        "Normal resting: 60-100. Beats per minute from mean RR interval.",
        "Below 60 suggests bradycardia; above 100 suggests tachycardia. Correlate with symptoms, medication (beta-blockers), fever, anemia, thyroid."),
    "rr_mean_s": ("Mean RR interval (s)",
        "Normal: ~0.6-1.0 s (HR 60-100). Time between consecutive R-peaks.",
        "Short RR = fast rhythm; long RR = slow rhythm. Used to derive heart rate."),
    "rr_std_s": ("RR variability, SD (s)",
        "Normal: small (<0.06 s at rest). Spread of RR intervals.",
        "Large values mean irregular rhythm - possible atrial fibrillation, frequent ectopics, or sinus arrhythmia."),
    "rmssd_s": ("RMSSD (s)",
        "Normal resting: ~0.02-0.08 s. Beat-to-beat variability.",
        "High RMSSD with irregular RR supports irregular rhythm. Sensitive to R-detection jitter - read with regularity CV."),
    "regularity_cv": ("Regularity, CV (unitless)",
        "Normal: <0.10. Coefficient of variation of RR (std/mean).",
        "Above ~0.12 suggests irregular rhythm. Main driver of the Rhythm Abnormality probability in this system."),
    "qrs_ms": ("QRS width (ms)",
        "Normal: 70-110. Ventricular depolarization duration (estimate).",
        "Above ~120 suggests wide-QRS / ventricular conduction delay or ectopic (PVC-like) beats. Main driver of Ectopic probability."),
    "qt_ms": ("QT interval (ms)",
        "Normal: ~350-450 (rate-dependent). R-peak to T-end estimate.",
        "Prolonged QT predisposes to arrhythmia. Always correct for rate (see QTc)."),
    "qtc_ms": ("QTc, Bazett (ms)",
        "Normal: <450 (men) / <470 (women). QT corrected for heart rate.",
        "Above threshold suggests QT prolongation - review QT-prolonging drugs, K/Mg/Ca, repeat 12-lead. Main driver of QT probability."),
    "st_mv": ("ST deviation (mV)",
        "Normal: -0.05...+0.05. ST level vs baseline (estimate).",
        "Elevation or depression beyond +/-0.1 mV may indicate ischemia/injury - urgent correlation with symptoms, serial ECGs, troponin. Main driver of ST probability."),
    "pr_ms": ("PR interval (ms)",
        "Normal: 120-200. Atrial-to-ventricular conduction time (estimate; P-wave lead-dependent).",
        "Prolonged PR suggests AV block; short PR with delta wave suggests pre-excitation. Confirm on 12-lead."),
    "p_present_frac": ("P-wave presence (fraction)",
        "Normal sinus rhythm: near 1.0 (P before every QRS).",
        "Low values with irregular rhythm support atrial fibrillation/flutter. Lead- and noise-dependent."),
    "n_beats": ("Detected beats (count)",
        "Depends on window length and rate (e.g. ~7-12 in 10 s at 60-100 bpm).",
        "Very low counts mean bradycardia, missed detection, or poor signal - check waveform and signal quality."),
}

FINDING_GUIDE = {
    "NSR": "Normal Sinus Rhythm: regular rhythm, P before each QRS, normal rate and conduction. High P here means the tracing looks unremarkable to the model.",
    "RHYTHM_ABN": "Rhythm Abnormality: irregular RR intervals. Consider atrial fibrillation/flutter, sinus arrhythmia, or frequent ectopics. Needs prolonged monitoring (Holter) to classify.",
    "ECTOPIC": "Ectopic Activity: wide/premature beats suggesting PVC-like activity. Quantify burden on Holter; check electrolytes and ischemia if indicated.",
    "ST_ABN": "ST Abnormality: ST elevation/depression. If chest pain present, follow acute coronary syndrome protocol (serial ECG + troponin) urgently.",
    "QT_PROLONG": "QT/QTc Prolongation: repolarization abnormality with torsades risk. Review drugs and electrolytes; repeat precise 12-lead QTc.",
    "BRADY": "Bradycardia: heart rate below ~60. Assess symptoms (syncope, dizziness) and rate-slowing drugs.",
    "TACHY": "Tachycardia: heart rate above ~100. Assess stability first, then classify narrow vs wide complex on 12-lead.",
}

SCORE_GUIDE = {
    "probability": ("Model probability P(F|E) (0-1, TRAINED)",
        "NSR/Rhythm/Ectopic: calibrated classifier outputs trained on 1770 expert-annotated MIT-BIH windows. "
        "BRADY/TACHY/QT/ST: deterministic clinical thresholds on real measurements (0.99 met / 0.01 not met).",
        ">=0.5 = supported; 0.3-0.5 = borderline; <0.3 = not supported. Research prototype - needs independent validation for clinical use."),
    "stability": ("Stability S = N_F/N (0-1)",
        "Fraction of N bootstrap states (resampled from this record's own beats) that still support the finding. Measures robustness, NOT clinical certainty.",
        ">=0.8 = robust; 0.5-0.8 = moderate; <0.5 = fragile - small measurement variation flips the result."),
    "qubo": ("QUBO selection (0/1)",
        "Final reconciled set after conflict resolution: E(x)=Sa?x?+Sb??x?x?. 1 = kept, 0 = rejected (e.g. NSR vs abnormal, brady vs tachy cannot coexist).",
        "Read together with P and S: high P + high S + selected = strongest model signal."),
    "sqi": ("Signal quality (Good/Acceptable/Poor + SNR dB)",
        "Quality of the tracing after denoising. Poor means noise, saturation, or flat-line - measurements and probabilities are unreliable.",
        "If Poor: reacquire the ECG (skin prep, lead contact, patient stillness) before interpreting."),
}


def flag_measurement(key: str, value: float) -> str:
    try:
        v = float(value)
    except Exception:
        return ""
    if key == "heart_rate_bpm":
        if v < 60: return "LOW - bradycardia range"
        if v > 100: return "HIGH - tachycardia range"
        return "Normal"
    if key == "qrs_ms":
        return "WIDE - abnormal" if v > 120 else ("Borderline" if v > 110 else "Normal")
    if key == "qtc_ms":
        return "PROLONGED - abnormal" if v >= 470 else ("Borderline" if v >= 450 else "Normal")
    if key == "st_mv":
        return "ABNORMAL - possible ischemia" if abs(v) >= 0.1 else ("Borderline" if abs(v) >= 0.05 else "Normal")
    if key == "regularity_cv":
        return "IRREGULAR - abnormal" if v >= 0.12 else ("Borderline" if v >= 0.10 else "Regular")
    if key == "pr_ms":
        if v > 200: return "PROLONGED - possible AV block"
        if v < 120: return "SHORT - verify"
        return "Normal"
    return ""
