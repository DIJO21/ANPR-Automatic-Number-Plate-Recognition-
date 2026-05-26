#!/bin/bash
# Forensic ANPR Ecosystem - Directory and Dependencies Setup Script
echo "=========================================================="
echo "Starting Forensic ANPR Setup..."
echo "=========================================================="

# Create required directories if they don't exist
mkdir -p datasets/license_plates/train/images datasets/license_plates/train/labels
mkdir -p datasets/license_plates/val/images datasets/license_plates/val/labels
mkdir -p checkpoints
mkdir -p reports
mkdir -p logs
mkdir -p outputs/forensic outputs/detected
mkdir -p forensic
mkdir -p decoder
mkdir -p ui
mkdir -p utils
mkdir -p notebooks
mkdir -p tests

echo "[✓] Directory structure initialized successfully."

# Install required packages
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    echo "[✓] Python dependencies installed successfully."
else
    echo "[!] requirements.txt not found, skipping package installation."
fi

echo "=========================================================="
echo "Forensic ANPR Ecosystem Setup Complete!"
echo "=========================================================="
