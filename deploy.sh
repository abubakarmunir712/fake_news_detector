#!/bin/bash
set -e

echo "🚀 Starting TruthLens Deployment..."

# Ensure we're in the project root
cd "$(dirname "$0")"

# --- Environment Check ---
if [ ! -f backend/.env ]; then
    echo "⚠️  backend/.env not found. Copying from example..."
    cp backend/.env.example backend/.env
    echo "📝 PLEASE UPDATE backend/.env WITH YOUR API KEYS!"
fi

# --- Docker Automation ---
echo "🐳 Building TruthLens Docker Image (Optimized Multi-Stage)..."
docker build -t truthlens .

echo "🛑 Cleaning up old containers..."
docker stop truthlens 2>/dev/null || true
docker rm truthlens 2>/dev/null || true

echo "🚢 Launching TruthLens..."
docker run -d --name truthlens -p 8000:8000 --env-file backend/.env truthlens

echo "✅ Deployment Complete!"
echo "📍 Access TruthLens at: http://localhost:8000"
echo "📜 View logs with:      docker logs -f truthlens"
echo ""
echo "Note: If you need to run in development mode (hot-reloading):"
echo "  Backend: cd backend && source venv/bin/activate && python main.py"
echo "  Frontend: cd frontend && npm run dev"
