#!/bin/bash
set -e

echo "Starting build process..."

# Upgrade pip
pip install --upgrade pip

# Install core dependencies first
echo "Installing core dependencies..."
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install python-multipart==0.0.6
pip install aiofiles==23.2.1

# Install data processing libraries
echo "Installing data processing libraries..."
pip install --only-binary=all pandas==2.1.4
pip install --only-binary=all numpy==1.25.2
pip install --only-binary=all scikit-learn==1.3.2
pip install --only-binary=all joblib==1.3.2

# Install visualization
echo "Installing visualization libraries..."
pip install --only-binary=all plotly==5.17.0

# Try to install optional dependencies
echo "Attempting to install optional dependencies..."
pip install google-generativeai==0.3.2 || echo "Google AI not installed, will use fallback"

echo "Build completed successfully!"