#!/bin/bash

# AI Agent Backend Startup Script
echo "🤖 Starting SentinelAI Epsilon AI Agent Backend..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python3."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found. Please run this script from the ai_agent_backend directory."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "ai_agent_venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv ai_agent_venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source ai_agent_venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🚀 Starting AI Agent Backend on port 8003..."
echo "🌐 Access the API documentation at: http://127.0.0.1:8003/docs"
echo "🔗 Health check at: http://127.0.0.1:8003/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 main.py
