"""Gemini connectivity doctor. Run locally where your key is configured:
    py -m backend.gemini_doctor
It NEVER prints your key (only length + status). Paste the output back
to get the correct model configured.
"""
import os

print("== 1. SDK ==")
try:
    import google.genai as genai
    from importlib.metadata import version
    print("import OK, google-genai version:", version("google-genai"))
except Exception as e:
    print("SDK IMPORT FAILED:", type(e).__name__, str(e)[:200])
    raise SystemExit(2)

print("== 2. Key presence (value never shown) ==")
key = os.environ.get("GEMINI_API_KEY", "")
print("GEMINI_API_KEY set:", bool(key), "| length:", len(key) if key else 0)
print("GEMINI_MODEL env:", os.environ.get("GEMINI_MODEL", "(unset -> default)"))
if not key:
    print("RESULT: no key configured. Set it, then re-run.")
    raise SystemExit(3)

print("== 3. Client init ==")
try:
    client = genai.Client(api_key=key, http_options={"timeout": 30000})
    print("Client init OK")
except Exception as e:
    print("CLIENT INIT FAILED:", type(e).__name__, str(e)[:200])
    raise SystemExit(4)

print("== 4. List models available to this key ==")
try:
    models = list(client.models.list())
    for m in models:
        acts = getattr(m, "supported_actions", None) or []
        print(f"  - {m.name}  actions={list(acts)[:4]}")
except Exception as e:
    print("LIST FAILED:", type(e).__name__, str(e)[:300])

print("== 5. Test generation (full messages, key never shown) ==")
from backend.gemini_client import MODEL_DEFAULT
want = os.environ.get("GEMINI_MODEL", MODEL_DEFAULT)
cands = [want]
try:
    names = [m.name.replace("models/", "") for m in client.models.list()]
    print("available:", names)
    for n in names:
        if n not in cands and ("flash" in n.lower() or "pro" in n.lower()):
            cands.append(n)
except Exception as e:
    print("LIST for candidates FAILED:", type(e).__name__)
    cands += ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
for mid in cands[:8]:
    try:
        r = client.models.generate_content(model=mid, contents="Reply with: OK")
        print(f"  {mid}: SUCCESS -> {(r.text or '').strip()[:60]}")
        print("  >>> USE THIS MODEL: " + mid)
        break
    except Exception as e:
        msg = str(e).replace(os.environ.get("GEMINI_API_KEY", ""), "***")
        print(f"  {mid}: {type(e).__name__}: {msg[:500]}")
print("DONE.")
