import json
from typing import List, Tuple

from flask import Blueprint, current_app, jsonify, request
from google import genai
from tavily import TavilyClient

from .auth import login_required

detect_bp = Blueprint("detect", __name__)

_gemini_client: genai.Client | None = None
_tavily_client: TavilyClient | None = None


def _ensure_clients() -> Tuple[bool, str | None]:
    global _gemini_client, _tavily_client
    cfg = current_app.config
    gemini_key = cfg.get("GEMINI_API_KEY")
    tavily_key = cfg.get("TAVILY_API_KEY")

    if not gemini_key or not tavily_key:
        return False, "GEMINI_API_KEY and TAVILY_API_KEY are required"

    if _gemini_client is None or _tavily_client is None:
        _gemini_client = genai.Client(api_key=gemini_key)
        _tavily_client = TavilyClient(api_key=tavily_key)
    return True, None


def _safe_gemini(prompt: str) -> Tuple[bool, str]:
    try:
        resp = _gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=prompt
        )
        return True, (resp.text or "").strip()
    except Exception as exc:
        return False, f"Gemini error: {exc}"


def _summarize_claim(claim: str) -> Tuple[bool, str]:
    prompt = (
        "Convert this claim into a concise news search query (<= 8 words). "
        "Keep names, numbers, locations, and verbs.\n\n"
        f"Claim: \"{claim}\"\nSearch query:"
    )
    return _safe_gemini(prompt)


def _search_articles(query: str, k: int = 5) -> Tuple[bool, List[str] | str]:
    try:
        res = _tavily_client.search(query, timeout=current_app.config["TIMEOUT"])
        results = res.get("results", [])[:k]
        articles = [f"{r['title']} - {r.get('content', '')}" for r in results]
        return True, articles
    except Exception as exc:
        return False, f"Tavily error: {exc}"


def _verdict_prompt(claim: str, sources: List[str]) -> str:
    sources_text = "\n".join(f"{idx + 1}. {src}" for idx, src in enumerate(sources))
    return (
        f"Claim:\n\"{claim}\"\n\nSources:\n{sources_text}\n\n"
        "Decide if the claim is Likely True, Likely Fake, or Unverifiable. "
        "Respond in JSON with keys `verdict` and `explanation`."
    )


@detect_bp.post("/api/detect")
@login_required
def detect():
    ok, err = _ensure_clients()
    if not ok:
        return jsonify({"error": err}), 500

    data = request.get_json(silent=True) or {}
    claim = (data.get("claim") or "").strip()
    if not claim:
        return jsonify({"error": "Claim is required"}), 400

    ok, search_query = _summarize_claim(claim)
    if not ok:
        return jsonify({"error": search_query}), 502

    ok, sources_or_err = _search_articles(search_query)
    if not ok:
        return jsonify({"error": sources_or_err, "search_query": search_query}), 502
    sources: List[str] = sources_or_err  # type: ignore[assignment]

    if not sources:
        return jsonify(
            {
                "verdict": "Unverifiable",
                "explanation": "No relevant sources found",
                "sources": [],
                "search_query": search_query,
            }
        )

    prompt = _verdict_prompt(claim, sources)
    ok, verdict = _safe_gemini(prompt)
    if not ok:
        return jsonify({"error": verdict}), 502

    try:
        verdict_struct = json.loads(verdict)
    except Exception:
        try:
            verdict_struct = json.loads(verdict.strip("```json").strip("```").strip())
        except Exception:
            verdict_struct = {"raw": verdict}

    verdict_struct.setdefault("sources", sources)
    verdict_struct.setdefault("search_query", search_query)
    return jsonify(verdict_struct)
