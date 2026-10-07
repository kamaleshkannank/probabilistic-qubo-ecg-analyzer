"""
Classical QUBO solver.
E(x) = sum a_i x_i + sum_{i<j} b_ij x_i x_j,  x_i in {0,1}
Linear terms encode (negative) evidence: a_i = -(p_i - 0.5) * w.
Quadratic terms encode conflict/compatibility penalties.
Solved by exhaustive search (n<=20) else simulated annealing.
"""
import numpy as np


def build_qubo(prob_list: list, weight: float = 4.0):
    codes = [p["code"] for p in prob_list]
    probs = [p["probability"] for p in prob_list]
    n = len(probs)
    a = np.array([-(p - 0.5) * weight for p in probs], dtype=float)
    B = np.zeros((n, n), dtype=float)
    idx = {c: i for i, c in enumerate(codes)}

    def penalize(c1, c2, w):
        if c1 in idx and c2 in idx:
            i, j = idx[c1], idx[c2]
            B[i, j] += w; B[j, i] += w

    # Clinical-logic conflicts (documented, tunable)
    penalize("NSR", "RHYTHM_ABN", 5.0)
    penalize("NSR", "ECTOPIC", 4.0)
    penalize("NSR", "ST_ABN", 4.0)
    penalize("NSR", "QT_PROLONG", 3.0)
    penalize("NSR", "BRADY", 4.0)
    penalize("NSR", "TACHY", 4.0)
    penalize("BRADY", "TACHY", 6.0)
    # Mild co-occurrence encouragement (negative = reward)
    penalize("RHYTHM_ABN", "ECTOPIC", -0.5)
    return {"codes": codes, "a": a.tolist(), "B": B.tolist()}


def energy_correct(x, a, B):
    x = np.asarray(x); a = np.asarray(a); B = np.asarray(B)
    e = float(np.dot(a, x))
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            e += float(B[i, j] * x[i] * x[j])
    return e


def _all_bit_vectors(n: int) -> np.ndarray:
    """Same order as itertools.product([0,1], repeat=n): leftmost slowest."""
    k = np.arange(1 << n, dtype=np.int64)[:, None]
    shifts = np.arange(n - 1, -1, -1, dtype=np.int64)[None, :]
    return ((k >> shifts) & 1).astype(np.int64)


def solve_qubo(a, B, seed=42):
    a = np.asarray(a, dtype=float); B = np.asarray(B, dtype=float)
    n = len(a)
    if n <= 20:
        # Vectorized exhaustive search. B is symmetric with zero diagonal, so
        # sum_{i<j} B[i,j] x_i x_j = (x^T B x)/2. argmin keeps the original
        # first-minimum tie-break (same candidate order as before).
        X = _all_bit_vectors(n).astype(float)
        energies = X @ a + (X @ B * X).sum(axis=1) / 2.0
        best_idx = int(np.argmin(energies))
        best_e = float(energies[best_idx])
        return {"solution": list(map(int, X[best_idx])), "energy": round(best_e, 4),
                "method": "exhaustive-classical"}
    # Simulated annealing fallback
    rng = np.random.default_rng(seed)
    x = (rng.random(n) < 0.3).astype(int)
    best = x.copy(); best_e = energy_correct(x, a, B)
    cur_e = best_e
    T, Tmin, alpha, steps = 2.0, 1e-3, 0.995, 8000
    for _ in range(steps):
        i = int(rng.integers(0, n))
        x[i] ^= 1
        e = energy_correct(x, a, B)
        if e < cur_e or rng.random() < np.exp(-(e - cur_e) / max(T, 1e-9)):
            cur_e = e
            if e < best_e:
                best_e, best = e, x.copy()
        else:
            x[i] ^= 1
        T = max(Tmin, T * alpha)
    return {"solution": list(map(int, best)), "energy": round(float(best_e), 4), "method": "simulated-annealing-classical"}
