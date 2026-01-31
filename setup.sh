#!/bin/bash

# AeroGuard Setup Script
echo "=================================================="
echo "  AeroGuard - Setup Script"
echo "=================================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p models/saved_models
mkdir -p notebooks

# Generate synthetic data
echo "Generating synthetic data..."
python utils/data_generator.py

# Train models (optional - can be skipped for quick start)
read -p "Do you want to train models now? This may take 10-15 minutes. (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Training models..."
    python models/train_models.py
else
    echo "Skipping model training. You can train later with: python models/train_models.py"
fi

echo ""
echo "=================================================="
echo "  Setup Complete!"
echo "=================================================="
echo ""
echo "To run the application:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Run app: streamlit run app.py"
echo ""
echo "Access the app at: http://localhost:8501"
echo "=================================================="
