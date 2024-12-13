#!/bin/bash

echo "🚀 Installing Trading Bot..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Make scripts executable
chmod +x start.sh stop.sh

# Create logs directory
mkdir -p logs

echo "✅ Installation complete!"
echo ""
echo "To start the bot:"
echo "1. source venv/bin/activate"
echo "2. ./start.sh"
echo ""
echo "To stop the bot:"
echo "  ./stop.sh"
echo ""
echo "To view logs:"
echo "  tail -f trading_bot.log" 