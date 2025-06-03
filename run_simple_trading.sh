#!/bin/bash

# Simple Trading Bot Runner Script
echo "╔══════════════════════════════════════════════╗"
echo "║          🚀 SIMPLE TRADING BOT 🚀            ║"
echo "║                                              ║"
echo "║  Multi-ticker live trading for Binance       ║"
echo "║  Focus: Setup, Entry, Stop Loss             ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found!"
    echo "Setting up configuration..."
    python3 start_simple_trading.py --setup
    echo ""
    echo "📝 Please edit .env file with your settings, then run this script again."
    exit 1
fi

# Menu function
show_menu() {
    echo "🔧 Select an option:"
    echo "1) 🧪 Start in TEST mode (recommended)"
    echo "2) 🔴 Start in LIVE mode (real trading)"
    echo "3) 📋 Show current configuration"
    echo "4) ⚙️  Setup/Edit configuration"
    echo "5) 🚪 Exit"
    echo ""
    read -p "Choose option (1-5): " choice
}

# Main menu loop
while true; do
    show_menu
    
    case $choice in
        1)
            echo ""
            echo "🧪 Starting in TEST mode..."
            python3 start_simple_trading.py --start --test
            break
            ;;
        2)
            echo ""
            echo "⚠️  WARNING: This will use real money!"
            read -p "Are you absolutely sure? (type 'yes'): " confirm
            if [ "$confirm" = "yes" ]; then
                echo "🔴 Starting in LIVE mode..."
                python3 start_simple_trading.py --start --live
            else
                echo "❌ Cancelled"
            fi
            break
            ;;
        3)
            echo ""
            python3 start_simple_trading.py --config
            echo ""
            read -p "Press Enter to continue..."
            ;;
        4)
            echo ""
            python3 start_simple_trading.py --setup
            echo ""
            read -p "Press Enter to continue..."
            ;;
        5)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid option. Please choose 1-5."
            echo ""
            ;;
    esac
done 