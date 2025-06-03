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
    test_capital = float(os.getenv('TEST_CAPITAL', '1000'))
    print(f"Mode: {'🧪 TEST' if test_mode else '🔴 LIVE'}")
    print(f"Capital: ${test_capital:.0f}")

    symbols = os.getenv('TRADING_SYMBOLS',
                        'BTCUSDT,ETHUSDT,ADAUSDT').split(',')
    capital_per_symbol = test_capital / len(symbols)
    print(f"Symbols: {', '.join(symbols)} ({len(symbols)} symbols)")
    print(f"Capital per Symbol: ${capital_per_symbol:.0f}")

    print(f"Interval: {os.getenv('TRADING_INTERVAL', '15m')}")
    print(
        f"Entry Evaluation: {'✅' if os.getenv('USE_ENTRY_EVALUATION', 'true').lower() == 'true' else '❌'}")
    print(f"Entry Threshold: {os.getenv('ENTRY_THRESHOLD', '45')}")

    # Position sizing details
    print("\n💰 Position Sizing:")
    capital_usage = float(os.getenv('CAPITAL_USAGE_PCT', '15.0'))
    leverage = int(os.getenv('LEVERAGE', '10'))
    max_risk = float(os.getenv('MAX_RISK_PER_TRADE_PCT', '8.0'))

    base_position = capital_per_symbol * (capital_usage / 100)
    leveraged_position = base_position * leverage
    margin_required = base_position

    print(f"Capital Usage: {capital_usage}% per trade")
    print(f"Base Position: ${base_position:.0f}")
    print(f"Leverage: {leverage}x")
    print(f"Position Size: ${leveraged_position:.0f}")
    print(f"Margin Required: ${margin_required:.0f}")
    print(f"Max Risk: {max_risk}% per trade")

    # Quality multipliers
    print(f"\n🎯 Quality Multipliers:")
    print(f"Excellent (80+): ${leveraged_position * 1.3:.0f}")
    print(f"Good (65+): ${leveraged_position * 1.15:.0f}")
    print(f"Moderate (<50): ${leveraged_position * 0.7:.0f}")

    telegram_enabled = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
    print(f"\n📱 Telegram: {'✅' if telegram_enabled else '❌'}")

    if capital_usage >= 25:
        print(
            f"\n⚠️  HIGH RISK: {capital_usage}% capital usage is aggressive!")
    elif capital_usage >= 20:
        print(f"\n⚠️  MEDIUM RISK: {capital_usage}% capital usage")
    else:
        print(f"\n✅ CONSERVATIVE: {capital_usage}% capital usage")


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
