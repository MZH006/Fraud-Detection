#!/bin/bash
# Phase 1 Setup Script for Fraud Detection System

set -e  # Exit on error

echo "=================================="
echo "Fraud Detection System - Phase 1 Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing production dependencies..."
pip install -r requirements.txt

echo ""
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "Created .env file"
    echo "Edit .env and add your SUPABASE_URL and SUPABASE_KEY"
else
    echo ".env file already exists"
fi

# Install pre-commit hooks
echo ""
echo "Installing pre-commit hooks..."
pre-commit install
echo "Pre-commit hooks installed"

# Run pre-commit on all files
echo ""
echo "Running code quality checks..."
pre-commit run --all-files || true  # Don't fail if there are formatting issues

echo ""
echo "=================================="
echo "Setup complete."
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your credentials:"
echo "   - SUPABASE_URL"
echo "   - SUPABASE_KEY"
echo ""
echo "2. Activate virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "3. Train the model and start the API:"
echo "   python scripts/train.py --save"
echo "   uvicorn src.app:app --reload"
echo ""
echo "4. Run the tests:"
echo "   pytest"
echo ""
