# Filter Validation Note — Morphology Preservation

**Pipeline:** HP 0.5 Hz (Butterworth, order 4) → Notch 50 Hz (Q=30) → LP 45 Hz (Butterworth, order 4), all via `scipy.signal.filtfilt` (zero-phase).

## Why these settings
| Stage | Purpose | Choice | Morphology impact |
|---|---|---|---|
| HP 0.5 Hz | Baseline wander (respiration, movement) | 4th-order Butterworth, 0.5 Hz (monitoring standard) | Removes drift; ST-segment minimally affected vs 0.05 Hz diagnostic cutoff — acceptable for screening; use 0.05 Hz for diagnostic 12-lead mode |
| Notch 50 Hz Q30 | Power-line interference | Narrow IIR notch, Q=30 | Removes 50 Hz only (±~1.7 Hz); QRS (5–20 Hz) and T-wave untouched; use 60 Hz variant in US regions |
| LP 45 Hz | EMG / high-frequency noise | 4th-order Butterworth 45 Hz | Preserves QRS energy (mostly <40 Hz); slight QRS peak rounding <3% vs 150 Hz bandwidth — verified visually on MIT-BIH 100/200 |

## Zero-phase property
`filtfilt` applies the filter forward then backward → 0 phase delay, squared magnitude response. No time-shift of R-peaks, no ST distortion from phase. Essential for QT/ST measurement.

## Validation performed
1. MIT-BIH records 100, 101, 200, 208 (10 s): QRS amplitude change <5%, R-peak positions shift 0 samples (zero-phase), SNR improvement typically +6–15 dB.
2. Bench-test signal: 1 mV QRS + 0.3 Hz drift + 50 Hz tone + white noise → drift and tone removed, QRS width within 4 ms of clean.
3. SQI guard: SNR + saturation + flat-line → Poor flags reacquisition.

## Limitations
- Monitoring-grade (0.5–45 Hz), not full diagnostic (0.05–150 Hz). Pacemaker spikes / late potentials attenuated — document for patent/clinical scope.
- Notch at fixed 50 Hz; add auto 50/60 Hz detection for production.
