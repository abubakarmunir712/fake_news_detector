# Fake News Detector

AI-powered claim verification system using Tavily search and Gemini for intelligent fact-checking. Users authenticate via SQLite-backed sessions to submit claims for analysis.

## Stack
- **Backend**: Flask, SQLite, flask-cors, python-dotenv
- **Frontend**: React 19 (Vite), shadcn-inspired UI with dark/light mode
- **AI/Search**: Gemini 2.0 Flash via `google-genai`, Tavily search API

## Project Structure
```
fake_news_detector/
├── backend/           # Flask API
│   ├── .env.example   # Environment template
│   ├── main.py        # Entry point
│   ├── config.py      # Centralized config
│   ├── auth.py        # Authentication routes
│   ├── detect.py      # Detection logic
│   ├── db.py          # Database setup
│   └── requirements.txt
├── frontend/          # React SPA
│   ├── .env.example   # Frontend env template
│   ├── src/
│   │   ├── components/ui/  # Reusable UI components
│   │   ├── App.jsx         # Main app
│   │   └── config.js       # API URL config
│   └── package.json
└── README.md
```

## Setup

### Backend Setup
```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
# - FLASK_SECRET_KEY (generate with: python -c "import secrets; print(secrets.token_hex(32))")
# - GEMINI_API_KEY (from Google AI Studio)
# - TAVILY_API_KEY (from Tavily)
```

### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Default VITE_API_URL=http://localhost:8000 is fine for local dev
```

## Development

### Run Backend
```bash
# From project root or backend/ directory
source backend/venv/bin/activate
python -m backend.main
# Backend runs at http://localhost:8000
```

### Run Frontend
```bash
cd frontend
npm run dev
# Frontend runs at http://localhost:5173
```

## Production Deployment

### Option 1: Automated Script (Recommended)
The included `deploy.sh` script handles setup, building, and environment checks.

```bash
chmod +x deploy.sh
./deploy.sh
```

### Option 2: Docker
Requires `npm run build` to be run first, or update Dockerfile for multi-stage build.

1. **Build Frontend:**
   ```bash
   cd frontend && npm install && npm run build && cd ..
   ```

2. **Build & Run Docker:**
   ```bash
   docker build -t fake-news-detector .
   docker run -p 8000:8000 --env-file backend/.env fake-news-detector
   ```

### Option 3: Manual Production Setup

1. Set production environment variables in `backend/.env`:
   ```
   FLASK_SECRET_KEY=<secure-random-key>
   GEMINI_API_KEY=<your-key>
   TAVILY_API_KEY=<your-key>
   PORT=8000
   FRONTEND_ORIGIN=https://yourdomain.com
   ```

2. Use a production WSGI server (gunicorn):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8000 backend.main:app
   ```

3. Set up reverse proxy (nginx/caddy) for SSL termination

### Frontend Production Build
1. Set `VITE_API_URL` in `frontend/.env`:
   ```
   VITE_API_URL=https://api.yourdomain.com
   ```

2. Build and deploy:
   ```bash
   cd frontend
   npm run build
   # Deploy dist/ folder to static hosting or serve via Flask
   ```

### Recommended Production Stack
- **Web Server:** Nginx or Caddy (reverse proxy + SSL)
- **WSGI Server:** Gunicorn (included in requirements)
- **Database:** SQLite (built-in) or upgrade to PostgreSQL
- **Process Manager:** systemd or supervisord
- **Monitoring:** Setup health check monitoring at `/health`

## API Endpoints

### Authentication
- `POST /api/signup` - Create account `{username, password}`
- `POST /api/login` - Sign in `{username, password}`
- `POST /api/logout` - Sign out
- `GET /api/me` - Get current user

### Detection
- `POST /api/detect` - Analyze claim (auth required)
  ```json
  {
    "claim": "NASA confirms aliens exist"
  }
  ```
  
  Response:
  ```json
  {
    "verdict": "Likely Fake",
    "explanation": "Analysis...",
    "sources": ["Source 1...", "Source 2..."],
    "search_query": "NASA aliens confirmation"
  }
  ```

## Features
- 🔐 Secure authentication with hashed passwords
- 🎨 Modern UI with dark/light theme toggle
- 🤖 AI-powered fact-checking via Gemini
- 🔍 Real-time web search via Tavily
- 📱 Fully responsive design
- ⚡ Fast React + Vite development

## Security Notes
- Never commit `.env` files
- Use strong secret keys in production
- Enable HTTPS in production
- Configure CORS origins properly
- Regularly update dependencies
