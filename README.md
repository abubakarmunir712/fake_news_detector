# 📰 Fake News Detector

A lightweight **Flask-based AI microservice** that detects whether a news **claim** is *Likely True*, *Likely Fake*, or *Unverifiable* using real-time articles and Gemini 2.0 Flash.

---

## 🚀 Features

- 📩 Accepts claims via simple `POST /detect` API  
- 🔍 Fetches real articles using **Tavily API**  
- 🤖 Uses **Gemini 2.0 Flash** for reasoning  
- 📦 Returns: `verdict`, `explanation`, `sources`  
- 🌐 CORS-enabled – frontend-ready  
- 🧪 Minimal HTML UI included for local testing  

---

## 📦 Requirements

- 🐍 Python 3.8+
- API Keys from:
  - 🔑 Google AI Studio (Gemini)
  - 🔑 Tavily Search

---

## 🔐 Environment Setup

```bash
GEMINI_API_KEY=your_google_key  
TAVILY_API_KEY=your_tavily_key  
PORT=8000           # Optional (default: 8000)  
TIMEOUT=8           # Optional request timeout (in seconds)
```
---

## ⚙️ Installation & Usage

```bash
# 1. Install dependencies
pip install flask flask-cors google-generativeai tavily python-dotenv

# 2. Run the server
python main.py

# Server runs at:
# http://localhost:8000
```

---

## 📮 API Endpoint

**POST /detect**

### Request Body:

```json
{
  "claim": "NASA confirms aliens exist"
}
```

### Sample Response:

```json
{
  "verdict": "Likely Fake",
  "explanation": "No credible sources support the claim.",
  "sources": [
    "NASA statement on UAPs - 'There is no confirmed evidence...'",
    "Scientific American - Experts debunk alien claims"
  ],
  "search_query": "NASA aliens confirmation"
}
```

---

## 💻 Local Frontend

Open the `index.html` file in your browser for a simple UI to test the API.

---

## 📌 Example cURL

```bash
curl -X POST http://localhost:8000/detect \
     -H "Content-Type: application/json" \
     -d '{"claim":"NASA confirms aliens exist"}'
```

---

## 🧠 How It Works

1. 🔎 Summarizes the input claim into a short query  
2. 🌐 Searches live news using **Tavily**  
3. 🧠 Sends claim + sources to **Gemini 2.0 Flash**  
4. 📤 Returns JSON with verdict, explanation, and sources  

---

## 🛡️ Disclaimer

This project is for **educational/demo purposes only**. Verdict accuracy may vary based on live data and model responses. Always double-check important claims.
