"""
ECG signal processing: Butterworth high-pass, 50 Hz notch, low-pass.
Settings validated to preserve ECG morphology (zero-phase filtfilt).

Pipeline (default fs=360 Hz, MIT-BIH native):
  1. High-pass Butterworth 0.5 Hz, order 4  -> baseline wander removal
  2. Notch 50 Hz, Q=30                      -> power-line interference
  3. Low-pass Butterworth 45 Hz, order 4    -> EMG/high-freq noise
All applied with scipy.signal.filtfilt (zero phase) to preserve
QRS morphology, ST-segment and T-wave shape.

References:
- Clifford et al. ECG signal processing; AHA recommendations
  (diagnostic bandwidth 0.05-150 Hz; monitoring 0.5-40 Hz).
- Our defaults (0.5-45 Hz + 50 Hz notch) are a monitoring-grade
  compromise that preserves QRS/ST for analysis while removing noise.
"""
import numpy as np
from scipy import signal


DEFAULT_FS = 360.0
HP_CUTOFF = 0.5
LP_CUTOFF = 45.0
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0


def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal = cutoff / nyq
    b, a = signal.butter(order, normal, btype="high", analog=False)
    return b, a


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal = cutoff / nyq
    b, a = signal.butter(order, normal, btype="low", analog=False)
    return b, a


def apply_filter_pipeline(raw: np.ndarray, fs: float = DEFAULT_FS,
                          hp_cut=HP_CUTOFF, lp_cut=LP_CUTOFF,
                          notch_freq=NOTCH_FREQ, notch_q=NOTCH_Q,
                          notch_on: bool = True):
    raw = np.asarray(raw, dtype=float)
    out = raw.copy()

    # 1. High-pass (baseline wander)
    b_hp, a_hp = butter_highpass(hp_cut, fs, order=4)
    out = signal.filtfilt(b_hp, a_hp, out)

    # 2. Notch (power-line). Skip if fs too low or notch near Nyquist.
    if notch_on and notch_freq < (fs / 2 - 1):
        b_n, a_n = signal.iirnotch(notch_freq, notch_q, fs)
        out = signal.filtfilt(b_n, a_n, out)

    # 3. Low-pass (muscle/EMG noise)
    b_lp, a_lp = butter_lowpass(lp_cut, fs, order=4)
    out = signal.filtfilt(b_lp, a_lp, out)

    return out


def signal_quality_index(raw: np.ndarray, filtered: np.ndarray) -> dict:
    """Simple SQI: high-freq SNR estimate, saturation, flat-line checks.
    Note: baseline-wander removal inflates (raw-filtered) residual, so SNR
    thresholds are lenient; Poor flags true noise/saturation/flat-line."""
    raw = np.asarray(raw, dtype=float)
    filtered = np.asarray(filtered, dtype=float)
    noise = raw - filtered
    sig_pow = float(np.mean(filtered ** 2) + 1e-12)
    noise_pow = float(np.mean(noise ** 2) + 1e-12)
    snr_db = 10 * np.log10(sig_pow / noise_pow)
    # Saturation: fraction of samples at extreme
    sat = float(np.mean(np.abs(raw) > (np.mean(np.abs(raw)) * 5 + 1e-9)))
    # Flat-line
    flat = float(np.std(raw) < 1e-6)
    if snr_db > 5 and sat < 0.01 and not flat:
        quality = "Good"
    elif snr_db > -8 and sat < 0.05 and not flat:
        quality = "Acceptable"
    else:
        quality = "Poor - flag for reacquisition/review"
    return {"snr_db": round(snr_db, 2), "saturation_frac": round(sat, 4),
            "is_flat": bool(flat), "quality": quality}
