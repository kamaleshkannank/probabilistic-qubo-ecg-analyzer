"""Performance timing: perf_counter-based, no sensitive data logged.
Timings dict holds milliseconds per stage; log_pipeline() emits one summary
line (stage=ms). Never logs patient data, API keys, or ECG signals."""
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("ecg.timing")


@contextmanager
def stage(timings: dict, name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round((time.perf_counter() - t0) * 1000, 1)


def log_pipeline(timings: dict, extra: str = ""):
    parts = " ".join(f"{k}={v}ms" for k, v in timings.items())
    logger.info("pipeline %s %s", parts, extra)
