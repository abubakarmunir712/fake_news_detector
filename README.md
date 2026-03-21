# TruthLens | AI-Powered Fact Verification

TruthLens is a premium, AI-powered claim verification system. It uses **Tavily Search** and **Gemini 2.0 Flash** to intelligently fact-check news headlines and rumors, providing detailed analysis and referenced sources.

## ✨ Features
- 🔐 **Secure Auth**: Hashed passwords and session-based authentication.
- 🎨 **Premium UI**: Modern glassmorphic design with mesh gradients and custom typography (Outfit).
- 🌗 **Adaptive Theme**: Refined light and dark modes with a dedicated toggle.
- 🤖 **Smart AI Analysis**: Strict logic that identifies vague claims and requests more detail instead of guessing.
- 🔍 **Real-time Research**: Deep integration with Tavily for up-to-the-minute source verification.
- 🚢 **Docker Ready**: Optimized multi-stage build for a minimal production footprint.

## 🚀 Quick Start (Docker)

The fastest way to get TruthLens running is using the automated deployment script:

```bash
chmod +x deploy.sh
./deploy.sh
```
*Note: Depending on your system, you may need to run docker commands with `sudo`.*

**Access TruthLens at**: [http://localhost:8000](http://localhost:8000)

---

## 🛠️ Manual Setup

### 1. Backend Configuration
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Google Gemini and Tavily API keys.
```

### 2. Frontend Configuration
```bash
cd frontend
npm install
cp .env.example .env
# Default VITE_API_URL=http://localhost:8000 is usually correct.
```

### 3. Running Locally (Development)
- **Backend**: `python -m backend.main` (runs on :8000)
- **Frontend**: `npm run dev` (runs on :5173 with hot-reload)

---

## 📋 Deployment Checklist

- [ ] **Security**: Generate a unique `FLASK_SECRET_KEY` in `backend/.env`.
- [ ] **API Keys**: Ensure valid `GEMINI_API_KEY` and `TAVILY_API_KEY` are set.
- [ ] **CORS**: Update `FRONTEND_ORIGIN` in production for proper security.
- [ ] **Health Check**: Monitor the `/health` endpoint for system status.
- [ ] **Backups**: Regularly backup the `instance/` directory (SQLite database).

## 🐳 Docker Commands (Reference)

```bash
# Build optimized multi-stage image
docker build -t truthlens .

# Run with environment file
docker run -p 8000:8000 --env-file backend/.env truthlens

# Check logs
docker logs -f truthlens
```

## 🛠️ Project Structure
```
fake_news_detector/
├── backend/           # Flask API & AI Logic
├── frontend/          # React App & Design System
├── Dockerfile         # Multi-stage build config
├── deploy.sh          # Automation script
└── README.md          # Project documentation
```

---
*Created with ❤️ for Truth and Clarity.*
