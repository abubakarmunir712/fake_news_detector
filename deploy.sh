#!/bin/bash
set -e

echo "🚀 Starting production deployment..."

# Ensure we're in the project root
cd "$(dirname "$0")"

# --- Backend Setup ---
echo "🔧 Setting up backend..."

# Check environment variables
if [ ! -f backend/.env ]; then
    echo "⚠️  backend/.env not found. Copying from example..."
    cp backend/.env.example backend/.env
    echo "📝 Please update backend/.env with your API keys!"
fi

# Create/Activate virtual environment
if [ ! -d backend/venv ]; then
    echo "📦 Creating virtual environment in backend/venv..."
    python3 -m venv backend/venv
fi

source backend/venv/bin/activate

# Install dependencies
echo "📦 Installing backend dependencies..."
pip install -r backend/requirements.txt

# --- Frontend Setup ---
echo "🎨 Setting up frontend..."
cd frontend

# Check environment variables
if [ ! -f .env ]; then
    echo "⚠️  frontend/.env not found. Creating from example..."
    cp .env.example .env
fi

# Install dependencies
if [ ! -d node_modules ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

# Build
echo "🏗️  Building frontend..."
npm run build

cd ..

echo "✅ Deployment complete!"
echo ""
echo "To start the server:"
echo "  source backend/venv/bin/activate"
echo "  gunicorn -c backend/gunicorn.conf.py backend.main:app"
echo ""
echo "Or for development:"
echo "  # Terminal 1 (Backend):"
echo "  source backend/venv/bin/activate"
echo "  python -m backend.main"
echo ""
echo "  # Terminal 2 (Frontend):"
echo "  cd frontend && npm run dev"
