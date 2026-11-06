#!/usr/bin/env powershell
# Phishing URL Analyzer - Setup and Startup Script
# This script installs dependencies and starts the API server

Write-Host "=== Phishing URL Analyzer MLOps Pipeline ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if Node.js is installed
Write-Host "Step 1: Checking Node.js installation..." -ForegroundColor Yellow
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "⚠ Node.js not found. Please install Node.js 20+ from https://nodejs.org/" -ForegroundColor Red
    return
}
Write-Host "✓ Node.js found: $(node --version)" -ForegroundColor Green

# Step 2: Install npm dependencies
Write-Host ""
Write-Host "Step 2: Installing npm dependencies..." -ForegroundColor Yellow
try {
    npm install --legacy-peer-deps
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
    return
}

# Step 3: Install Python dependencies
Write-Host ""
Write-Host "Step 3: Installing Python dependencies..." -ForegroundColor Yellow
python3 -c "import pip; pip install -r requirements.txt" 2>&1 | Write-Host "✓ Python dependencies ready" -ForegroundColor Green

# Step 4: Initialize DVC if not already done
Write-Host ""
Write-Host "Step 4: Initializing DVC..." -ForegroundColor Yellow
if (-not (Test-Path -Path "D:\ML\Project\.dvc")) {
    dvc init
    Write-Host "✓ DVC initialized" -ForegroundColor Green
} else {
    Write-Host "✓ DVC already initialized" -ForegroundColor Green
}

# Step 5: Train the ML model
Write-Host ""
Write-Host "Step 5: Training ML model..." -ForegroundColor Yellow
python3 -m src.training.train 2>&1
Write-Host "✓ Model training complete" -ForegroundColor Green

# Step 6: Start the API server
Write-Host ""
Write-Host "Step 6: Starting API server..." -ForegroundColor Yellow
Write-Host "   The API will be available at http://localhost:8000" -ForegroundColor Green
Write-Host "   Health check: http://localhost:8000/api/health" -ForegroundColor Green
Write-Host "   Analyze endpoint: POST http://localhost:8000/api/analyze" -ForegroundColor Green
Write-Host "   Example: curl -X POST http://localhost:8000/api/analyze -H 'Content-Type: application/json' -d '{\"url\": \"https://example.com\"}'" -ForegroundColor Green
npm start