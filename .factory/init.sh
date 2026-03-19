#!/bin/bash
# MLB F5 Prediction System - Environment Initialization
# This script is idempotent - safe to run multiple times

set -e

echo "=== MLB F5 Prediction System Initialization ==="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate 2>/dev/null || . .venv/Scripts/activate 2>/dev/null

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]" --quiet

# Create data directories
echo "Creating data directories..."
mkdir -p data/raw/statcast
mkdir -p data/models
mkdir -p data/logs
mkdir -p data/training

# Check for .env file
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found. Copy .env.example and fill in API keys."
    if [ -f ".env.example" ]; then
        echo "  cp .env.example .env"
    fi
fi

# Initialize database
echo "Initializing database..."
python3 -c "from src.db import init_db; init_db('data/mlb.db')" 2>/dev/null || echo "  (Database init requires src/db.py to exist)"

echo "=== Initialization complete ==="
echo "Next steps:"
echo "  1. Copy .env.example to .env and add API keys"
echo "  2. Run: pytest tests/ -v"
echo "  3. Run: python -m src.pipeline.daily --date today --mode prod --dry-run"
