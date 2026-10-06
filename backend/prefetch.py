"""Prefetch MIT-BIH records into the local disk cache (one-time download).
Run:  py -m backend.prefetch            (default record set)
      py -m backend.prefetch 100 101    (specific records)
After this, the portal serves 100% real data from disk - no internet needed,
no synthetic fallback exists anywhere in the codebase.
"""
import sys

from .mitbih_loader import download_record, is_cached, PREFETCH_DEFAULT


def main(records=None):
    records = records or PREFETCH_DEFAULT
    ok, failed = [], []
    for r in records:
        if is_cached(r):
            print(f"[{r}] already cached - skip")
            ok.append(r)
            continue
        try:
            d = download_record(r)
            print(f"[{r}] downloaded {len(d['signal'])/d['fs']:.0f}s @ {d['fs']:.0f}Hz - cached")
            ok.append(r)
        except Exception as e:
            print(f"[{r}] FAILED: {e}")
            failed.append(r)
    print(f"\nDone: {len(ok)} cached, {len(failed)} failed {failed}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
