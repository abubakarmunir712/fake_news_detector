#!/bin/bash
set -e

echo "🚀 Starting production deployment..."

# Load environment
if [ ! -f backend/.env ]; then
    echo "❌ Error: backend/.env not found. Copy backend/.env.example and configure it."
    exit 1
fi

# Activate virtual environment
if [ ! -d .venv ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install backend dependencies
echo "📦 Installing backend dependencies..."
pip install -r backend/requirements.txt

# Build frontend
echo "🎨 Building frontend..."
cd frontend
if [ ! -d node_modules ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi
npm run build
cd ..

echo "✅ Deployment complete!"
echo ""
echo "To start the server:"
echo "  source .venv/bin/activate"
echo "  gunicorn -c backend/gunicorn.conf.py backend.main:app"
echo ""
echo "Or for development:"
echo "  python -m backend.main"
