"""
Fake News Detector API 🚀
========================
A lightweight Flask micro‑service that exposes a single **POST /detect** endpoint.
It accepts a news **claim** (string JSON field), fetches related articles with **Tavily**,
then asks **Gemini 2.0 Flash** to judge whether the claim is *Likely True*, *Likely Fake*,
or *Unverifiable*. CORS is enabled so browsers can call the API from any origin.

---
## Environment (`.env`)
```env
GEMINI_API_KEY=your_google_key
TAVILY_API_KEY=your_tavily_key
PORT=8000            # optional
TIMEOUT=8            # optional request timeout (sec)
```

## Install & Run
```bash
pip install flask flask-cors google-generativeai tavily python-dotenv
python fake_news_detector_tavily.py   # default port 8000
```

## Example Request
```bash
curl -X POST http://localhost:8000/detect \
     -H "Content-Type: application/json" \
     -d '{"claim":"NASA confirms aliens exist"}'
```
---
"""
from __future__ import annotations

import json
import os
import traceback
from typing import List, Tuple

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import google.generativeai as genai
from tavily import TavilyClient

# ---------------------------------------------------------------------------
# 🔧  Setup
# ---------------------------------------------------------------------------

load_dotenv()

# ----  Validate environment -------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not GEMINI_API_KEY or not TAVILY_API_KEY:
    raise RuntimeError("GEMINI_API_KEY and TAVILY_API_KEY must be set in the environment")

PORT = int(os.getenv("PORT", 8000))
TIMEOUT = int(os.getenv("TIMEOUT", 8))  # seconds

# ----  Configure clients ----------------------------------------------------

genai.configure(api_key=GEMINI_API_KEY)
_GEMINI = genai.GenerativeModel("models/gemini-2.0-flash")
_TAVILY = TavilyClient(api_key=TAVILY_API_KEY)

app = Flask(__name__)
# ➡️  Enable CORS for all routes & origins
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------------------------------------------------------------------
# 🔎  Helper functions (with error handling)
# ---------------------------------------------------------------------------

def _safe_gemini(prompt: str) -> Tuple[bool, str]:
    """Call Gemini and return (success, text)."""
    try:
        resp = _GEMINI.generate_content(prompt, safety_settings={})
        return True, resp.text.strip()
    except Exception as e:
        return False, f"Gemini error: {e}"


def _summarise_claim(claim: str) -> Tuple[bool, str]:
    prompt = (
        "You are an assistant that converts verbose claims into concise news-search "
        "queries. Keep key entities, numbers, places, and verbs. ≤ 8 words.\n\n"
        f"Claim: \"{claim}\"\nSearch query:"
    )
    return _safe_gemini(prompt)


def _search_articles(query: str, k: int = 5) -> Tuple[bool, List[str] | str]:
    try:
        res = _TAVILY.search(query, timeout=TIMEOUT)
        results = res.get("results", [])[:k]
        articles = [f"{r['title']} - {r.get('content', '')}" for r in results]
        return True, articles
    except Exception as e:
        return False, f"Tavily error: {e}"


def _build_verdict_prompt(claim: str, sources: List[str]) -> str:
    sources_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sources))
    return (
        f"Claim:\n\"{claim}\"\n\nSources:\n{sources_text}\n\n"
        "Task:\nBased on these sources, is the claim **likely true** or **likely fake**?\n"
        "Respond in JSON with keys `verdict` (Likely True | Likely Fake | Unverifiable) "
        "and `explanation` (one‑sentence reason)."
    )

# ---------------------------------------------------------------------------
# 🌐  Flask route
# ---------------------------------------------------------------------------

@app.route("/detect", methods=["POST"])
def detect():
    """Main endpoint → returns JSON verdict structure or error."""
    try:
        data = request.get_json(silent=True) or {}
        claim = data.get("claim") or request.args.get("claim")
        if not claim:
            return jsonify({"error": "Missing 'claim' field"}), 400

        # 1️⃣  Summarise claim ➜ search query
        ok, search_q_or_err = _summarise_claim(claim)
        if not ok:
            return jsonify({"error": search_q_or_err}), 502
        search_query: str = search_q_or_err

        # 2️⃣  Retrieve sources with Tavily
        ok, sources_or_err = _search_articles(search_query)
        if not ok:
            return jsonify({"error": sources_or_err, "search_query": search_query}), 502
        sources: List[str] = sources_or_err
        if not sources:
            return jsonify({
                "verdict": "Unverifiable",
                "explanation": "No relevant sources found",
                "sources": [],
                "search_query": search_query,
            })

        # 3️⃣  Ask Gemini for verdict
        prompt = _build_verdict_prompt(claim, sources)
        ok, verdict = _safe_gemini(prompt)
        if not ok:
            return jsonify({"error": verdict}), 502

        # 4️⃣  Parse JSON if possible
        try:
            verdict_struct = json.loads(verdict)
        except Exception:
            try:
                # Attempt to parse JSON from inside the "raw" string
                inner_json = json.loads(verdict.strip("```json").strip("```").strip())
                verdict_struct = inner_json
            except Exception:
                verdict_struct = {"raw": verdict}

        verdict_struct.setdefault("sources", sources)
        verdict_struct.setdefault("search_query", search_query)
        return jsonify(verdict_struct)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {e}"}), 500

# ---------------------------------------------------------------------------
# 🚀  Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
