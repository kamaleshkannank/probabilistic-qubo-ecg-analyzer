"""Optimization regression tests: equivalence, safety, PDF validity.
Run: py -m backend.test_optimize   (exit 0 = all pass)
Covers: batch==loop inference, QUBO==brute force, timings present,
Gemini missing/invalid key + timeout, PDF valid without AI.
"""
import sys

PASS = []


def check(name, cond):
    assert cond, f"FAILED: {name}"
    PASS.append(name)
    print(f"  ok: {name}")


def test_inference_equivalence(feat):
    from backend.probability import estimate_probabilities, estimate_probabilities_batch
    from backend.stochastic import sample_states
    states = sample_states(feat, 12, 7)
    loop = [estimate_probabilities(s) for s in states]
    batch = estimate_probabilities_batch(states)
    check("batch==loop probabilities", loop == batch)


def test_qubo_equivalence():
    import numpy as np
    from backend.qubo import build_qubo, solve_qubo, energy_correct
    rng = np.random.default_rng(0)
    cases = []
    probs = [{"code": c, "probability": p} for c, p in
             [("NSR", 0.9), ("RHYTHM_ABN", 0.1), ("ECTOPIC", 0.2), ("ST_ABN", 0.05),
              ("QT_PROLONG", 0.4), ("BRADY", 0.02), ("TACHY", 0.03)]]
    cases.append(build_qubo(probs))
    for _ in range(5):  # random cases incl. near-ties
        r = rng.random(7).round(2)
        cases.append(build_qubo([{"code": c, "probability": float(p)}
                                 for c, p in zip(probs and [q["code"] for q in probs], r)]))
    tie = build_qubo([{"code": f"X{i}", "probability": 0.5} for i in range(5)])
    cases.append(tie)
    for q in cases:
        sol = solve_qubo(q["a"], q["B"])
        # independent brute-force reference
        import itertools
        best, be = None, 1e18
        for bits in itertools.product([0, 1], repeat=len(q["a"])):
            e = energy_correct(bits, q["a"], q["B"])
            if e < be:
                be, best = e, bits
        check(f"qubo==bruteforce n={len(q['a'])} e={sol['energy']}",
              sol["solution"] == list(map(int, best)) and sol["energy"] == round(be, 4))
    big = solve_qubo([0.1] * 25, [[0.0] * 25 for _ in range(25)], seed=1)
    check("qubo SA path format", len(big["solution"]) == 25 and big["method"] == "simulated-annealing-classical")


def test_gemini_safety():
    import os
    from unittest.mock import patch
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ["AI_PROVIDER"] = "gemini"
    from backend import gemini_client as g
    check("missing key unavailable", g.interpret({}, [], [], [])["available"] is False)
    pr = g.build_prompt({"heart_rate_bpm": 70}, [], [], [])
    check("prompt grounded", "NEVER invent findings" in pr and "70" in pr)
    os.environ["GEMINI_API_KEY"] = "dummy-invalid-key"
    try:
        r = g.interpret({"heart_rate_bpm": 70}, [], [], [])
        check("invalid key graceful", r["available"] is False and "dummy" not in str(r))
        check("invalid key has kind", r.get("error_kind") in (
            "auth", "not_found", "quota", "request", "server", "network", "timeout"))
    finally:
        os.environ.pop("GEMINI_API_KEY", None)
    os.environ["GEMINI_API_KEY"] = "dummy"
    try:
        with patch("google.genai.Client") as c:
            c.return_value.models.generate_content.side_effect = TimeoutError("t/o")
            r = g.interpret({"heart_rate_bpm": 70}, [], [], [])
            check("timeout graceful", r["available"] is False and r.get("error_kind") == "timeout")
        from google.genai import errors as _err
        for code, kind in ((404, "not_found"), (401, "auth"), (429, "quota"), (500, "server")):
            check(f"classify HTTP {code} -> {kind}",
                  g._classify(_err.APIError(code, {})).get("error_kind") == kind)
        import json as _json

        class _Resp:
            text = _json.dumps({"summary": "s", "finding_explanations": {},
                                "normal_vs_abnormal": "n", "possible_associations": "a",
                                "recommendations": "r", "warnings": "w"})

        class _Models:
            def __init__(self):
                self.calls = []

            def generate_content(self, model, contents, config=None):
                self.calls.append(model)
                if len(self.calls) == 1:
                    raise _err.APIError(503, {})
                return _Resp()

        with patch("google.genai.Client") as c:
            c.return_value.models = _Models()
            r = g.interpret({"heart_rate_bpm": 70}, [], [], [])
            check("503 falls back + succeeds", r["available"] is True)
            check("answering model recorded", r.get("model") == "gemini-3.5-flash")
            check("single call per model", c.return_value.models.calls ==
                  ["gemini-3.5-flash-lite", "gemini-3.5-flash"])
    finally:
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("AI_PROVIDER", None)
    os.environ["AI_PROVIDER"] = "template"
    try:
        probs = [{"code": "ECTOPIC", "name": "Ectopic", "probability": 0.8,
                  "stability": 0.9, "qubo_selected": 1, "method": "trained-ml"},
                 {"code": "NSR", "name": "NSR", "probability": 0.05,
                  "stability": 0.0, "qubo_selected": 0, "method": "trained-ml"}]
        r = g.interpret({"heart_rate_bpm": 96}, probs, ["ECTOPIC"], [])
        check("template always available", r["available"] is True)
        check("template names selection", "ECTOPIC" in r["summary"])
        check("template labeled honestly", r.get("provider") == "template"
              and "no LLM" in r.get("model", ""))
    finally:
        os.environ.pop("AI_PROVIDER", None)
    os.environ["AI_PROVIDER"] = "ollama"
    os.environ["OLLAMA_HOST"] = "http://localhost:9"  # nothing listens: fast refuse
    try:
        r = g.interpret({"heart_rate_bpm": 70}, [], [], [])
        check("ollama-down graceful", r["available"] is False and "ollama" in r["reason"])
    finally:
        os.environ.pop("AI_PROVIDER", None)
        os.environ.pop("OLLAMA_HOST", None)


def test_pdf_and_routes():
    import os
    os.environ.pop("GEMINI_API_KEY", None)
    import numpy as np
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.report import build_pdf, DISCLAIMER
    c = TestClient(app)
    r = c.get("/api/analyze_mitbih", params={"record": "100", "seconds": 8, "n_states": 20, "seed": 42})
    check("analyze 200", r.status_code == 200)
    j = r.json()
    for k in ("features", "probabilities", "qubo", "stability", "waveform", "timings",
              "explanations", "r_peaks"):
        check(f"contract key {k}", k in j)
    check("timings has bootstrap+qubo", "bootstrap" in j["timings"] and "qubo" in j["timings"])
    r = c.get("/api/report_mitbih", params={"record": "100", "seconds": 8, "n_states": 20,
                                            "seed": 42, "clinic_name": "C", "patient_id": "P"})
    check("pdf route 200", r.status_code == 200 and len(r.content) > 50000)
    from pypdf import PdfReader
    import io as _io
    t = "\n".join((p.extract_text() or "") for p in PdfReader(_io.BytesIO(r.content)).pages)
    for needle in ["ECG ANALYSIS REPORT", "AI-Assisted Interpretation - Requires Clinician Review",
                   "AI-assisted narrative unavailable", "Clinician Review", DISCLAIMER[:40], "Page 1",
                   "Clinician Reviewer", "Signature"]:
        check(f"pdf has '{needle[:30]}'", needle in t)
    check("no key material in pdf", "AIza" not in t and "GEMINI_API_KEY" not in t)


def main():
    import numpy as np
    sys.path.insert(0, ".")
    from backend.mitbih_loader import load_mitbih
    from backend.ecg_processing import apply_filter_pipeline
    from backend.features import extract_features
    d = load_mitbih("208", 10.0)
    feat = extract_features(apply_filter_pipeline(np.asarray(d["signal"]), d["fs"]), d["fs"])
    print("inference equivalence:"); test_inference_equivalence(feat)
    print("qubo equivalence:"); test_qubo_equivalence()
    print("gemini safety:"); test_gemini_safety()
    print("pdf + routes:"); test_pdf_and_routes()
    print(f"\nALL {len(PASS)} CHECKS PASSED")


if __name__ == "__main__":
    main()
