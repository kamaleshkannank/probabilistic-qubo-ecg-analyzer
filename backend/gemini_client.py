"""AI interpretation: called AFTER QUBO, grounded ONLY in measured outputs.
Providers (env AI_PROVIDER, default "gemini"):
  gemini   - Google Gemini via official google-genai SDK (needs GEMINI_API_KEY).
  ollama   - local Ollama server, no key/quota (needs Ollama + pulled model).
  template - built-in deterministic writer, always works, no LLM.
Any failure -> {"available": False} so the PDF renders without AI section.
Security: keys only from env, never in frontend/PDF/logs.
"""
import json
import os

MODEL_DEFAULT = "gemini-3.5-flash-lite"  # reliable high-capacity model
# Fallbacks tried in order if primary hits server/quota/timeout capacity limits.
FALLBACKS_DEFAULT = "gemini-3.5-flash,gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash"
TIMEOUT_MS = 30000  # configurable request timeout (SDK http_options, milliseconds)


def is_configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def _safe(v, ndigits=3):
    try:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return "missing"
        return round(f, ndigits)
    except Exception:
        return "missing"


def build_prompt(features: dict, probabilities: list, qubo_codes: list,
                 urgent: list) -> str:
    """Deterministic prompt built only from pipeline outputs (testable offline)."""
    meas_lines = []
    for k in ("heart_rate_bpm", "rr_mean_s", "rr_std_s", "rmssd_s",
              "regularity_cv", "qrs_ms", "qt_ms", "qtc_ms", "st_mv",
              "pr_ms", "p_present_frac", "n_beats"):
        meas_lines.append(f"- {k}: {_safe(features.get(k))}")
    prob_lines = []
    for p in probabilities:
        prob_lines.append(
            f"- {p['code']} ({p['name']}): P={p.get('probability')}, "
            f"stability={p.get('stability')}, QUBO_selected={p.get('qubo_selected')}, "
            f"source={p.get('method')}")
    prompt = f"""You are assisting with a RESEARCH ECG report. Interpret ONLY the data below.
STRICT RULES:
1. NEVER invent findings not present in the data. NEVER claim diagnostic certainty.
2. NEVER treat a 'missing' measurement as normal - say it is unavailable.
3. Probabilities marked measurement-criterion are deterministic rules (0.99/0.01), not ML confidence.
4. Stability is bootstrap robustness, not clinical certainty.
5. Flag potentially urgent patterns for prompt clinician review.
6. This is decision support, not a diagnosis. A doctor must review everything.

MEASURED ECG VALUES:
{chr(10).join(meas_lines)}

MODEL FINDINGS (post-QUBO final set: {qubo_codes}):
{chr(10).join(prob_lines)}

DETERMINISTIC URGENT FLAGS (from fixed cutoffs): {urgent if urgent else 'none'}

Return JSON ONLY with exactly these keys:
{{"summary": "2-3 sentence overall picture",
  "finding_explanations": {{"CODE": "one-line plain explanation", ...}},
  "normal_vs_abnormal": "which listed findings look normal, abnormal, or uncertain and why",
  "possible_associations": "possible clinical associations (differential context, not diagnosis)",
  "recommendations": "suggested further medical evaluation steps",
  "warnings": "warnings/limitations, including anything urgent"}}"""
    return prompt


def _fail(kind: str, reason: str) -> dict:
    """Sanitized failure: kind for code, short reason for users. Never any secret."""
    return {"available": False, "error_kind": kind, "reason": reason}


def _classify(exc: Exception) -> dict:
    """Map SDK/network exceptions to separate, sanitized error kinds.
    Only the exception TYPE name and numeric code are used - never the
    message text (it can echo request details)."""
    name = type(exc).__name__
    code = getattr(exc, "code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if name in ("TimeoutError", "TimeoutException", "ConnectTimeout", "ReadTimeout",
                "PoolTimeout", "WriteTimeout") or "Timeout" in name:
        return _fail("timeout", "AI unavailable: request timed out.")
    if code == 404 or "NotFound" in name:
        return _fail("not_found", "AI unavailable: model not found for this API key.")
    if code in (401, 403) or "API_KEY_INVALID" in name or "Unauthenticated" in name:
        return _fail("auth", "AI unavailable: invalid API key or access denied.")
    if code == 429 or "Exhausted" in name or "RateLimit" in name:
        return _fail("quota", "AI unavailable: quota or rate limit exceeded.")
    if isinstance(code, int) and 500 <= code <= 599:
        return _fail("server", "AI unavailable: provider server error.")
    if isinstance(code, int) and 400 <= code <= 499:
        return _fail("request", "AI unavailable: rejected request.")
    return _fail("network", f"AI unavailable: {name}.")


def interpret(features: dict, probabilities: list, qubo_codes: list,
              urgent: list) -> dict:
    """Single call per report (never per-finding). Never raises.
    Provider from AI_PROVIDER env: gemini (default) | ollama | template."""
    provider = os.environ.get("AI_PROVIDER", "gemini").strip().lower()
    if provider == "template":
        return _template_insights(features, probabilities, qubo_codes, urgent)
    if provider == "ollama":
        return _ollama_insights(features, probabilities, qubo_codes, urgent)
    return _gemini_insights(features, probabilities, qubo_codes, urgent)


def _template_insights(features: dict, probabilities: list, qubo_codes: list,
                       urgent: list) -> dict:
    """Deterministic, always-available writer. Not an LLM: it only restates
    pipeline outputs in report language. Clearly labeled as template."""
    try:
        from .explain import FINDING_GUIDE
    except ImportError:
        from explain import FINDING_GUIDE
    sel = [p for p in probabilities if int(p.get("qubo_selected", 0)) == 1]
    border = [p for p in probabilities
              if int(p.get("qubo_selected", 0)) == 0
              and float(p.get("probability", 0)) >= 0.3]
    if sel:
        summary = ("QUBO-selected findings: " +
                   ", ".join(f"{p['code']} (P={p.get('probability')}, "
                             f"S={p.get('stability')})" for p in sel) + ".")
    else:
        summary = ("No finding met the QUBO selection threshold; "
                   "tracing is unremarkable to the model.")
    if border:
        summary += " Borderline (not selected): " + ", ".join(p["code"] for p in border) + "."
    hr = features.get("heart_rate_bpm", "missing")
    return {"available": True, "provider": "template",
            "model": "template (deterministic, no LLM)",
            "summary": summary,
            "finding_explanations": {p["code"]: FINDING_GUIDE.get(p["code"], "")
                                     for p in probabilities},
            "normal_vs_abnormal":
                f"Abnormal: {[p['code'] for p in sel] or 'none'}. "
                f"Uncertain: {[p['code'] for p in border] or 'none'}. "
                f"Heart rate {hr} bpm.",
            "possible_associations":
                "Template restatement only - no clinical associations inferred. "
                "Use Gemini/Ollama provider for AI-generated context.",
            "recommendations":
                "Physician review of the QUBO-selected findings above; "
                "correlate with symptoms, 12-lead ECG, labs and imaging.",
            "warnings":
                ("Template output, not AI reasoning. " +
                 ("Urgent flags present: " + "; ".join(urgent) if urgent else
                  "No deterministic urgent flags."))}


def _ollama_insights(features: dict, probabilities: list, qubo_codes: list,
                     urgent: list) -> dict:
    """Local Ollama LLM: no key, no quota. Needs `ollama serve` + pulled model."""
    import urllib.request
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    timeout_s = int(os.environ.get("OLLAMA_TIMEOUT_S", "120"))
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps({"model": model, "prompt": build_prompt(
            features, probabilities, qubo_codes, urgent),
            "stream": False, "temperature": 0.2,
            "format": "json"}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            outer = json.loads(r.read().decode())
        data = json.loads(outer.get("response", ""))
    except Exception as e:
        name = type(e).__name__
        hint = (" Is Ollama running? Start it, then run: ollama pull " + model
                if "URLError" in name or "Connection" in name or "refused" in str(e).lower()
                else "")
        return _fail("network", f"AI unavailable (ollama): {name}." + hint)
    data["available"] = True
    data["provider"] = "ollama"
    data["model"] = "ollama/" + model
    return data


def _gemini_insights(features: dict, probabilities: list, qubo_codes: list,
                     urgent: list) -> dict:
    """Google Gemini via the official google-genai SDK; never raises.
    Single call per report (never per-finding). Key from env only."""
    if not is_configured():
        return _fail("missing_key",
                     "AI interpretation unavailable: server AI key not configured.")
    try:
        from google import genai
    except Exception:
        return _fail("network", "AI unavailable: SDK missing.")
    model = os.environ.get("GEMINI_MODEL", MODEL_DEFAULT)
    timeout_ms = int(os.environ.get("GEMINI_TIMEOUT_MS", str(TIMEOUT_MS)))
    fallbacks = [m.strip() for m in
                 os.environ.get("GEMINI_FALLBACKS", FALLBACKS_DEFAULT).split(",")
                 if m.strip() and m.strip() != model]
    last_err = None
    for attempt, mid in enumerate([model] + fallbacks):
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                                  http_options={"timeout": timeout_ms})
            resp = client.models.generate_content(
                model=mid,
                contents=build_prompt(features, probabilities, qubo_codes, urgent),
                config={"temperature": 0.2, "response_mime_type": "application/json"})
            text = (resp.text or "").strip()
            break
        except Exception as e:
            err = _classify(e)  # never includes key material or raw traces
            if err.get("error_kind") in ("missing_key", "auth"):
                return err  # retrying cannot help: wrong/missing credentials
            last_err = err
    else:
        return last_err or _fail("network", "AI unavailable.")
    if not text:
        return _fail("bad_response", "AI unavailable: empty response.")
    try:
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(text)
    except Exception:
        return _fail("bad_response", "AI unavailable: unreadable response.")
    data["available"] = True
    data["model"] = mid  # the model that actually answered (may be a fallback)
    return data


def detect_urgent(features: dict, probabilities: list) -> list:
    """Deterministic urgent flags from fixed cutoffs (independent of Gemini)."""
    flags = []
    try:
        hr = float(features.get("heart_rate_bpm", 0))
        if hr >= 150:
            flags.append(f"Severe tachycardia (HR {hr} bpm) - prompt review")
        if 0 < hr <= 35:
            flags.append(f"Severe bradycardia (HR {hr} bpm) - prompt review")
        if abs(float(features.get("st_mv", 0))) >= 0.2:
            flags.append("Marked ST deviation (>=0.2 mV) - possible acute ischemia, urgent")
        if float(features.get("qtc_ms", 0)) >= 500:
            flags.append("Severely prolonged QTc (>=500 ms) - arrhythmia risk, prompt review")
    except Exception:
        pass
    return flags


def summarize_findings(probabilities: list) -> dict:
    """Deterministic normal/abnormal/uncertain split (code-computed, not LLM).
    NSR is normal rhythm: a selected NSR is reported under supported_normal,
    never under abnormal."""
    normal, abnormal, uncertain, supported_normal = [], [], [], []
    for p in probabilities:
        pr = float(p.get("probability", 0))
        st = p.get("stability")
        try:
            st = float(st)
        except Exception:
            st = None
        sel = int(p.get("qubo_selected", 0))
        if p.get("code") == "NSR":
            if sel == 1 and pr >= 0.5:
                supported_normal.append(p["code"])
            elif pr >= 0.3:
                uncertain.append(p["code"])
            else:
                normal.append(p["code"])
        elif sel == 1 and pr >= 0.5 and (st is None or st >= 0.5):
            abnormal.append(p["code"])
        elif pr >= 0.3 or (st is not None and 0.3 <= st < 0.5):
            uncertain.append(p["code"])
        else:
            normal.append(p["code"])
    return {"abnormal": abnormal, "uncertain": uncertain, "normal": normal,
            "supported_normal": supported_normal}
