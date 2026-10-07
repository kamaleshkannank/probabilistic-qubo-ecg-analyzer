"""Controlled clinical interpretation layer (presentation only).
READS analysis outputs; never modifies predictions, probabilities,
bootstrap stability, or QUBO selections.
- NSR is a NORMAL finding: QUBO-selected NSR renders as SUPPORTED normal,
  never as abnormal (fixes the old summarize_findings mislabel).
- Possible conditions + next steps come from fixed cautious mappings.
- Dynamic: only supported / borderline / review-needed findings are shown.
"""
from .explain import FINDING_GUIDE

# Code -> (display name, is_normal_rhythm, possible condition, next step)
MAPPING = {
    "NSR": ("Normal sinus rhythm",
            "No specific abnormal rhythm is supported by this analysis window.",
            "Routine clinical follow-up according to clinical context; "
            "repeat ECG if symptoms persist."),
    "RHYTHM_ABN": ("Rhythm abnormality",
            "Possible atrial fibrillation/flutter or another rhythm disturbance. "
            "Requires confirmation; pattern alone does not diagnose.",
            "Clinical review with a standard 12-lead ECG and, when clinically "
            "indicated, prolonged ambulatory ECG/Holter monitoring."),
    "ECTOPIC": ("Ectopic activity",
            "Possible premature ventricular/ectopic beats; cannot be confirmed "
            "from this short analysis window alone.",
            "12-lead ECG and, when clinically indicated, ambulatory/Holter "
            "monitoring to quantify ectopic burden."),
    "ST_ABN": ("ST abnormality",
            "Possible myocardial ischemia/injury pattern. Requires clinical "
            "correlation; never a confirmed infarction from this system alone.",
            "Prompt clinical assessment with 12-lead/serial ECG and cardiac "
            "biomarker evaluation when clinically indicated."),
    "QT_PROLONG": ("QT/QTc prolongation",
            "Possible prolonged-QT pattern with increased ventricular "
            "arrhythmia risk. Not a confirmed long-QT syndrome.",
            "Confirm QT/QTc using an appropriate ECG and review relevant "
            "medications and electrolytes."),
    "BRADY": ("Bradycardia",
            "Heart rate below the bradycardia threshold.",
            "12-lead ECG and clinical/symptom/medication review; ambulatory "
            "monitoring when clinically indicated."),
    "TACHY": ("Tachycardia",
            "Heart rate above the tachycardia threshold.",
            "12-lead ECG to classify the rhythm and further "
            "monitoring/evaluation when clinically indicated."),
}

ORDER = ["NSR", "RHYTHM_ABN", "ECTOPIC", "ST_ABN", "QT_PROLONG", "BRADY", "TACHY"]
ABNORMAL_CODES = [c for c in ORDER if c != "NSR"]

THRESHOLD = 0.5
BORDERLINE_P = 0.3


def _num(p, key):
    try:
        return float(p.get(key, 0))
    except Exception:
        return 0.0


def classify(probabilities: list) -> dict:
    """Dynamic triage from actual values. NSR selected => normal_supported."""
    supported_normal, supported_abnormal, borderline, not_supported = [], [], [], []
    for p in probabilities:
        code = p.get("code", "")
        pr, st, sel = _num(p, "probability"), _num(p, "stability"), int(p.get("qubo_selected", 0))
        if code == "NSR":
            if sel == 1 and pr >= THRESHOLD:
                supported_normal.append(code)
            elif pr >= BORDERLINE_P:
                borderline.append(code)
            else:
                not_supported.append(code)
        elif sel == 1 and pr >= THRESHOLD:
            supported_abnormal.append(code)
        elif pr >= BORDERLINE_P or (0.3 <= st < 0.5 and pr >= 0.2):
            borderline.append(code)
        else:
            not_supported.append(code)
    return {"supported_normal": supported_normal,
            "supported_abnormal": supported_abnormal,
            "borderline": borderline, "not_supported": not_supported}


def conclusion(probabilities: list) -> dict:
    """Primary automated conclusion (dynamic, nothing hardcoded)."""
    c = classify(probabilities)
    by_code = {p.get("code"): p for p in probabilities}
    if c["supported_normal"] and not c["supported_abnormal"]:
        primary = {"code": "NSR", "verdict": "NORMAL SINUS RHYTHM - SUPPORTED",
                   "detail": by_code.get("NSR", {})}
    elif c["supported_abnormal"]:
        code = c["supported_abnormal"][0]
        primary = {"code": code,
                   "verdict": f"{MAPPING[code][0].upper()} - SUPPORTED (requires clinician confirmation)",
                   "detail": by_code.get(code, {})}
    elif c["borderline"]:
        code = c["borderline"][0]
        primary = {"code": code, "verdict": "BORDERLINE - CLINICIAN REVIEW REQUIRED",
                   "detail": by_code.get(code, {})}
    else:
        primary = {"code": "NSR", "verdict": "NO FINDING SUPPORTED - CLINICIAN REVIEW REQUIRED",
                   "detail": by_code.get("NSR", {})}
    return {"primary": primary, "borderline": [by_code.get(b, {"code": b}) for b in c["borderline"]
                                               if b != primary["code"]],
            "not_supported": c["not_supported"], "supported_abnormal": c["supported_abnormal"]}


RELEVANT_MEAS = {
    "NSR": (("heart_rate_bpm", "bpm"), ("regularity_cv", ""), ("n_beats", "beats")),
    "RHYTHM_ABN": (("heart_rate_bpm", "bpm"), ("rr_mean_s", "s"), ("rr_std_s", "s"),
                   ("rmssd_s", "s"), ("regularity_cv", "")),
    "ECTOPIC": (("qrs_ms", "ms"), ("heart_rate_bpm", "bpm"), ("regularity_cv", "")),
    "ST_ABN": (("st_mv", "mV"),),
    "QT_PROLONG": (("qt_ms", "ms"), ("qtc_ms", "ms")),
    "BRADY": (("heart_rate_bpm", "bpm"), ("rr_mean_s", "s")),
    "TACHY": (("heart_rate_bpm", "bpm"), ("rr_mean_s", "s")),
}


def _stability_text(st):
    try:
        st = float(st)
    except Exception:
        return "stability not calculated."
    if st >= 0.8:
        return (f"stability {st:.2f}: robust - supported across nearly all "
                f"bootstrap resamples.")
    if st >= 0.5:
        return (f"stability {st:.2f}: moderate - supported by most resamples, "
                f"sensitive to beat variation.")
    if st > 0.0:
        return (f"stability {st:.2f}: fragile - disappears under small "
                f"measurement variation.")
    return ("stability 0.00: unsupported across all bootstrap resamples - "
            "the point estimate alone should not drive decisions.")


def _qubo_text(sel):
    if int(sel or 0) == 1:
        return ("QUBO Selected: kept in the final consistent set after "
                "conflict resolution (selection is consistency, not proof).")
    return ("QUBO Not selected: rejected or superseded during consistency "
            "resolution (e.g. conflicts with a stronger finding).")


def detailed_insight_blocks(probabilities: list, features: dict) -> list:
    """Full-detail blocks: evidence, measurement context, stability and QUBO
    reading, condition, next step, limitations. All values from analysis."""
    blocks = insight_blocks(probabilities, features)
    for b in blocks:
        code = b["code"]
        p = next((x for x in probabilities if x.get("code") == code), {})
        meas = "; ".join(f"{k}={features.get(k, 'n/a')}{u}".rstrip()
                         for k, u in RELEVANT_MEAS.get(code, ()))
        b["evidence_full"] = (
            f"{b['finding']} Model probability={_num(p, 'probability'):.3f} "
            f"(method: {p.get('method', 'n/a')}). "
            f"{_stability_text(p.get('stability'))} {_qubo_text(p.get('qubo_selected'))}")
        b["measurements"] = (f"Supporting measurements: {meas}."
                             if meas else "Supporting measurements: not available.")
    return blocks


def synthesis(probabilities: list, features: dict) -> str:
    """Overall picture paragraph from the actual triage (dynamic)."""
    c = classify(probabilities)
    hr = features.get("heart_rate_bpm", "n/a")
    n = features.get("n_beats", "n/a")
    parts = []
    if c["supported_normal"]:
        parts.append("the tracing shows a supported normal sinus rhythm pattern")
    if c["supported_abnormal"]:
        parts.append("with supported abnormal findings: " + ", ".join(c["supported_abnormal"]))
    if c["borderline"]:
        parts.append("borderline considerations requiring review: " + ", ".join(c["borderline"]))
    if not c["supported_normal"] and not c["supported_abnormal"]:
        parts.append("no finding is firmly supported")
    s = (f"This {features.get('n_beats', 'n/a')}-beat window (heart rate {hr} bpm) suggests "
         + ("; ".join(parts)) + ". "
         f"{len(c['not_supported'])} of 7 candidate findings are not supported. "
         "All statements below restate algorithm outputs; clinical correlation is required.")
    return s


def insight_blocks(probabilities: list, features: dict) -> list:
    """FINDING / CONDITION / EVIDENCE / NEXT STEP / LIMITATIONS per relevant finding."""
    by_code = {p.get("code"): p for p in probabilities}
    c = classify(probabilities)
    shown = []
    for code in ORDER:
        if code in c["supported_normal"] or code in c["supported_abnormal"] or code in c["borderline"]:
            shown.append(code)
    if not shown:
        shown = ["NSR"]
    blocks = []
    for code in shown:
        p = by_code.get(code, {})
        name, cond, step = MAPPING[code]
        status = ("SUPPORTED" if code in c["supported_normal"] + c["supported_abnormal"]
                  else "BORDERLINE - requires review")
        ev = (f"{name}: probability={_num(p, 'probability'):.3f}, "
              f"bootstrap stability={p.get('stability', 'n/a')}, "
              f"QUBO={'Selected' if int(p.get('qubo_selected', 0)) == 1 else 'Not selected'}.")
        if code == "NSR":
            ev += f" Heart rate={features.get('heart_rate_bpm', 'n/a')} bpm."
            finding = "Normal sinus rhythm pattern supported. Not a disease."
        else:
            finding = f"{name} pattern {'supported' if status == 'SUPPORTED' else 'borderline'} by the algorithm."
        blocks.append({"code": code, "status": status, "finding": finding,
                       "condition": cond, "evidence": ev, "next_step": step,
                       "limitations": "Single-lead screening estimate; confirm with standard 12-lead ECG and clinical correlation."})
    return blocks
