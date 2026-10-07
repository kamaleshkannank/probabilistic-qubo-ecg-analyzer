"""MIT-BIH Arrhythmia Database loader: local disk cache first, PhysioNet download second.
NO synthetic fallback: if data is unavailable an explicit error is raised."""
import os
from functools import lru_cache
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_CACHE = os.path.join(BASE_DIR, "data", "mitbih_cache")


def _cache_dirs():
    """Prefer repository local cache first; support env override and Windows drive fallbacks."""
    dirs = []
    env = os.environ.get("MITBIH_CACHE_DIR")
    if env:
        dirs.append(env)
    dirs.append(LOCAL_CACHE)
    if os.name == "nt":
        for d in ["D:/ecg_cache/mitbih", "E:/ecg_cache/mitbih"]:
            if d not in dirs:
                dirs.append(d)
    return dirs


def _cache_path(record: str) -> str:
    for d in _cache_dirs():
        p = os.path.join(d, f"mitdb_{record}.npz")
        if os.path.exists(p):
            return p
    return os.path.join(_cache_dirs()[0], f"mitdb_{record}.npz")


def is_cached(record: str) -> bool:
    return os.path.exists(_cache_path(str(record)))


@lru_cache(maxsize=8)
def _read_cached(path: str):
    """In-memory cache of full records: repeat analyzes skip disk (~50ms).
    Returned array is never mutated (callers slice+copy via astype)."""
    z = np.load(path)
    return z["signal"], float(z["fs"])


def download_record(record: str) -> dict:
    """Download full record from PhysioNet; cache on disk (best effort)."""
    import wfdb
    record = str(record)
    rec = wfdb.rdrecord(record, pn_dir="mitdb", physical=True)
    sig = np.array(rec.p_signal[:, 0], dtype=np.float32)
    fs = float(rec.fs)
    units = rec.units[0] if rec.units else "mV"
    try:
        os.makedirs(os.path.dirname(_cache_path(record)), exist_ok=True)
        np.savez_compressed(_cache_path(record), signal=sig, fs=fs, units=units)
    except OSError:
        pass  # disk full etc: serve from memory, retry cache next time
    return {"signal": sig, "fs": fs, "units": units}

AVAILABLE_RECORDS = ["100", "101", "102", "103", "104", "105", "106", "107",
                     "109", "111", "112", "113", "114", "115", "116", "117",
                     "119", "121", "122", "123", "124", "200", "201", "202",
                     "203", "205", "208", "209", "210", "212", "213", "214",
                     "215", "217", "219", "220", "221", "222", "223", "228",
                     "230", "231", "232", "233", "234"]

PREFETCH_DEFAULT = ["100", "101", "102", "103", "200", "201", "202", "203",
                    "208", "209", "210", "212", "213", "214", "215", "217",
                    "219", "222", "223", "228", "230", "231", "232", "233"]


def load_mitbih(record: str = "100", seconds: float = 10.0, channel: int = 0):
    """Real MIT-BIH data only. Cache-first; downloads once then serves from disk."""
    record = str(record)
    if record not in AVAILABLE_RECORDS:
        raise ValueError(f"Unknown MIT-BIH record '{record}'.")
    if channel != 0:
        raise ValueError("Only channel 0 is cached in this build.")
    if is_cached(record):
        sig, fs = _read_cached(_cache_path(record))
        source = "MIT-BIH Arrhythmia Database — local disk cache (real data)"
    else:
        try:
            d = download_record(record)
        except Exception as e:
            raise RuntimeError(
                f"MIT-BIH record {record} is not cached and download failed: {e}. "
                "Run: py -m backend.prefetch (needs internet once).") from e
        sig, fs = d["signal"], d["fs"]
        source = "MIT-BIH Arrhythmia Database (PhysioNet mitdb download, now cached)"
    n = int(seconds * fs)
    if n > len(sig):
        raise ValueError(f"Requested {seconds}s but record {record} holds {len(sig)/fs:.0f}s.")
    return {"signal": sig[:n].astype(float), "fs": fs, "record": record,
            "units": "mV", "source": source,
            "cached": is_cached(record)}
