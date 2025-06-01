import os
from dotenv import load_dotenv
import google.generativeai as genai
from tavily import TavilyClient

load_dotenv()

# Configure Gemini (LLM)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.0-flash")

# Configure Tavily (search engine‑style retriever)
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def summarize_claim_for_search(claim: str) -> str:
    """Use Gemini to turn a raw claim into a tight search query."""
    prompt = f"""
    You are a helpful assistant. Convert the claim below into a concise news‑search query.
    • Keep key entities, numbers, places, and actions.
    • Remove filler/stop words. ≤ 8 words.

    Claim: "{claim}"
    Search query:
    """
    resp = model.generate_content(prompt)
    # Gemini sometimes returns extra lines; keep first line only
    return resp.text.splitlines()[0].strip()


def fetch_news_articles(query: str, max_results: int = 5):
    """Retrieve news snippets from Tavily."""
    response = tavily_client.search(query)
    results = response.get("results", [])[:max_results]
    return [f"{r['title']} - {r.get('content', '')}" for r in results]


def make_prompt(claim: str, sources: list[str]) -> str:
    sources_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sources))
    return f"""
    Claim:\n"{claim}"

    Sources:\n{sources_text}

    Task:\nBased on these sources, is the claim **likely true** or **likely fake**?\nRespond in the following JSON format:\n{{\n  \"verdict\": \"Likely True | Likely Fake | Unverifiable\",\n  \"explanation\": \"<one‑sentence reason>\"\n}}
    """


def get_verdict(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text.strip()


def detect_fake_news(claim: str):
    print("📝 Summarising claim for search…")
    search_query = summarize_claim_for_search(claim)
    print(f"🔎 Search query → {search_query}")

    print("🔍 Fetching articles via Tavily…")
    sources = fetch_news_articles(search_query)
    if not sources:
        return "⚠️ No relevant news articles found. Can't verify."

    print("🧠 Building prompt for Gemini…")
    prompt = make_prompt(claim, sources)

    print("🤖 Asking Gemini…")
    verdict_json = get_verdict(prompt)
    return verdict_json


if __name__ == "__main__":
    user_claim = input("Enter a news claim: ")
    result = detect_fake_news(user_claim)
    print("\n🧾 Result:\n", result)
