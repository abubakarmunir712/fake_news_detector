import json
import logging
from typing import List, Tuple, Dict, Any, Optional

from flask import Blueprint, current_app, jsonify, request
from google import genai
from tavily import TavilyClient

from .auth import login_required

# Create blueprint for detection routes
detect_bp = Blueprint("detect", __name__)

# Global clients (lazy initialized)
_gemini_client: Optional[genai.Client] = None
_tavily_client: Optional[TavilyClient] = None

# Configure logger
logger = logging.getLogger(__name__)


def _ensure_clients() -> Tuple[bool, Optional[str]]:
    """Initialize API clients if not already active."""
    global _gemini_client, _tavily_client
    cfg = current_app.config
    gemini_key = cfg.get("GEMINI_API_KEY")
    tavily_key = cfg.get("TAVILY_API_KEY")

    if not gemini_key or not tavily_key:
        return False, "GEMINI_API_KEY and TAVILY_API_KEY are required"

    try:
        if _gemini_client is None:
            _gemini_client = genai.Client(api_key=gemini_key)
        if _tavily_client is None:
            _tavily_client = TavilyClient(api_key=tavily_key)
        return True, None
    except Exception as e:
        logger.error(f"Client initialization failed: {e}")
        return False, str(e)


def _search_articles(query: str, k: int = 5) -> Tuple[bool, List[Dict[str, str]] | str]:
    """Search for articles using Tavily API."""
    try:
        res = _tavily_client.search(query, timeout=current_app.config["TIMEOUT"])
        results = res.get("results", [])[:k]

        # Format sources for both AI analysis and frontend display
        articles = []
        for r in results:
            url = r.get("url", "")
            # Ensure URL is valid for frontend display
            if not url or not url.startswith("http"):
                continue

            articles.append(
                {
                    "title": r.get("title", "Unknown Title"),
                    "content": r.get("content", ""),
                    "url": url,
                    "score": r.get("score", 0),
                }
            )

        return True, articles
    except Exception as exc:
        logger.error(f"Tavily search error: {exc}")
        return False, f"Tavily error: {exc}"


def _analyze_claim(claim: str, sources: List[Dict[str, str]]) -> Tuple[bool, str]:
    """
    Perform single-pass analysis using Gemini.
    Asks for search query generation (if needed), verification, and explanation in one go
    to reduce latency and API calls.
    """
    try:
        sources_text = "\n".join(
            f"{i+1}. [{src['title']}]({src['url']}): {src['content'][:300]}..."
            for i, src in enumerate(sources)
        )

        prompt = f"""
        Analyze the following text against the provided news sources.
        
        Text: "{claim}"
        
        Sources:
        {sources_text}
        
        Task:
        1. FIRST, determine if the input is a verifiable claim or news topic.
           - If it is a greeting, general question (e.g. "how are you"), or unrelated to news/facts, reject it.
           - Set 'verdict' to "Out of Scope".
           - Set 'explanation' to "I verify news and claims. Please ask me to check a rumor, headline, or fact."
           - Set 'confidence_score' to 0.
           - STOP here.
        
        2. If it IS a claim:
           - Determine if it is 'Likely True', 'Likely Fake'.
           - If the provided sources do not contain enough information to verify or debunk the claim, set 'verdict' to "Insufficient Info".
           - Provide a clear, concise explanation citing the sources.
           - Assign a confidence score (0-100).
        
        Return a valid JSON object with the following structure:
        {{
            "verdict": "Likely True" | "Likely Fake" | "Insufficient Info" | "Out of Scope",
            "explanation": "Your explanation here...",
            "confidence_score": 85
        }}
        """

        resp = _gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        return True, (resp.text or "").strip()
    except Exception as exc:
        logger.error(f"Gemini analysis error: {exc}")
        return False, f"Gemini error: {exc}"


@detect_bp.post("/api/detect")
@login_required
def detect():
    """
    Main detection endpoint.
    Flow:
    1. Validate input
    2. Search for information (using claim as query directly first)
    3. Analyze with AI
    4. Return structured result
    """
    # 1. Initialize clients
    ok, err = _ensure_clients()
    if not ok:
        return jsonify({"error": err}), 500

    # 2. Parse input
    data = request.get_json(silent=True) or {}
    claim = (data.get("claim") or "").strip()
    if not claim:
        return jsonify({"error": "Claim is required"}), 400

    # 3. Search (using claim directly as query for simplicity/speed)
    # In a more complex flow, we could ask AI to generate a query first,
    # but for "one api call" requirement, we'll try to keep it streamlined.
    ok, sources_or_err = _search_articles(claim)
    if not ok:
        return jsonify({"error": sources_or_err}), 502

    sources: List[Dict[str, str]] = sources_or_err  # type: ignore

    if not sources:
        return jsonify(
            {
                "verdict": "Unverifiable",
                "explanation": "No relevant sources found to verify this claim.",
                "confidence_score": 0,
                "sources": [],
            }
        )

    # 4. Analyze
    ok, result_json = _analyze_claim(claim, sources)
    if not ok:
        return jsonify({"error": result_json}), 502

    try:
        # Parse AI response
        # Gemini 2.0 Flash is good at returning JSON when requested
        result = json.loads(result_json)

        # Add sources to the response for the frontend
        result["sources"] = sources
        result["search_query"] = claim

        return jsonify(result)

    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return (
            jsonify(
                {
                    "verdict": "Error",
                    "explanation": "Failed to parse AI response",
                    "raw_response": result_json,
                }
            ),
            500,
        )
