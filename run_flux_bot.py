#!/usr/bin/env python3
"""
Simple Flux Trading Bot Runner
Quick launcher for flux_live_trading.py
"""

import os
import sys
from dotenv import load_dotenv


def print_banner():
    banner = """
╔════════════════════════════════════════════════════╗
║          🚀 FLUX ORDER BLOCK TRADING BOT 🚀        ║
║                                                    ║
║  Pine Script Accurate Order Block Detection       ║
║  Advanced Entry Evaluation & Risk Management      ║
╚════════════════════════════════════════════════════╝
    """
    print(banner)


def check_setup():
    """Check if setup is complete"""
    if not os.path.exists('.env'):
        print("❌ No .env file found!")
        print("\n🔧 Quick Setup:")
        print("1. cp flux_trading_config.example .env")
        print("2. Edit .env with your Binance API keys")
        print("3. Run this script again")
        return False

    load_dotenv()

    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')

    if not api_key or 'your_' in api_key:
        print("❌ Please set your BINANCE_API_KEY in .env file")
        return False

    if not api_secret or 'your_' in api_secret:
        print("❌ Please set your BINANCE_API_SECRET in .env file")
        return False

    return True


def show_current_config():
    """Show current configuration"""
    load_dotenv()

    print("\n📋 Current Configuration:")
    print("-" * 40)

    test_mode = os.getenv('TEST_MODE', 'true').lower() == 'true'
    print(f"Mode: {'🧪 TEST' if test_mode else '🔴 LIVE'}")
    print(
        f"Symbols: {os.getenv('TRADING_SYMBOLS', 'BTCUSDT,ETHUSDT,ADAUSDT')}")
    print(f"Interval: {os.getenv('TRADING_INTERVAL', '15m')}")
    print(
        f"Entry Evaluation: {'✅' if os.getenv('USE_ENTRY_EVALUATION', 'true').lower() == 'true' else '❌'}")
    print(f"Entry Threshold: {os.getenv('ENTRY_THRESHOLD', '45')}")
    print(f"Risk per Trade: {os.getenv('RISK_PER_TRADE_PCT', '2.0')}%")
    print(f"Leverage: {os.getenv('LEVERAGE', '10')}x")

    telegram_enabled = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
    print(f"Telegram: {'✅' if telegram_enabled else '❌'}")


def main():
    print_banner()

    # Check setup
    if not check_setup():
        return

    # Show config
    show_current_config()

    # Confirm start
    print("\n" + "="*50)
    test_mode = os.getenv('TEST_MODE', 'true').lower() == 'true'

    if test_mode:
        print("🧪 Starting in TEST MODE (no real money)")
        confirm = input("Press Enter to start, or 'q' to quit: ")
        if confirm.lower() == 'q':
            return
    else:
        print("🔴 WARNING: LIVE MODE - REAL MONEY!")
        confirm = input("Type 'START' to confirm live trading: ")
        if confirm != 'START':
            print("❌ Cancelled")
            return

    print("\n🚀 Starting Flux Trading Bot...")
    print("📝 Check flux_trading.log for detailed logs")
    print("📱 Check Telegram for notifications")
    print("Press Ctrl+C to stop")
    print("-" * 50)

    # Start the bot
    try:
        from flux_live_trading import main
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Bot stopped by user")
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure flux_live_trading.py exists")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
