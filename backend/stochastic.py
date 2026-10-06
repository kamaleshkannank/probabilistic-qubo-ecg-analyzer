"""Uncertainty processor: NONPARAMETRIC BOOTSTRAP over real measured beats.
No assumed sigmas: each of the N states resamples the record's own beats
(RR intervals, per-beat QRS/QT/ST) with replacement and recomputes the
aggregate features. Stability S(F) = N_F / N is therefore grounded in the
patient's own beat-to-beat variability.
"""
import numpy as np

from .probability import estimate_probabilities


def _summarize_rr(rr: np.ndarray) -> dict:
    rr = np.asarray(rr, dtype=float)
    m = float(np.mean(rr)) if len(rr) else 0.0
    out = {"rr_mean_s": m,
           "rr_std_s": float(np.std(rr)) if len(rr) else 0.0,
           "rmssd_s": float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else 0.0,
           "regularity_cv": float(np.std(rr) / (m + 1e-9)) if len(rr) else 0.0,
           "heart_rate_bpm": round(60.0 / m, 1) if m > 0 else 0.0}
    return out


def sample_states(base_feat: dict, n_states: int = 100, seed: int = 42):
    """Bootstrap states from the record's own per-beat distributions."""
    rng = np.random.default_rng(seed)
    rr = np.asarray(base_feat.get("rr_intervals_s", []), dtype=float)
    qrs = np.asarray(base_feat.get("qrs_per_beat_ms", []), dtype=float)
    qt = np.asarray(base_feat.get("qt_per_beat_ms", []), dtype=float)
    st = np.asarray(base_feat.get("st_per_beat_mv", []), dtype=float)
    n_beats = int(base_feat.get("n_beats", 0))
    k = max(3, n_beats)
    states = []
    for _ in range(n_states):
        f = dict(base_feat)
        if len(rr):
            rrs = rng.choice(rr, size=len(rr), replace=True)
            f.update(_summarize_rr(rrs))
        if len(qrs):
            f["qrs_ms"] = float(np.median(rng.choice(qrs, size=len(qrs), replace=True)))
        if len(qt):
            q = float(np.median(rng.choice(qt, size=len(qt), replace=True)))
            f["qt_ms"] = q
            rrm = f.get("rr_mean_s", 0) or 0
            f["qtc_ms"] = float(q / ((rrm ** 0.5) + 1e-9)) if rrm > 0 else q
        if len(st):
            f["st_mv"] = float(np.median(rng.choice(st, size=len(st), replace=True)))
        states.append(f)
    return states


def stability_analysis(base_feat: dict, base_probs: list, n_states=100, seed=42, threshold=0.5):
    states = sample_states(base_feat, n_states, seed)
    counts = {p["code"]: 0 for p in base_probs}
    for s in states:
        for p in estimate_probabilities(s):
            if p["probability"] >= threshold:
                counts[p["code"]] += 1
    stability = [{"code": c, "support": counts[c], "total": n_states,
                  "stability": round(counts[c] / n_states, 3)} for c in counts]
    return states, stability
