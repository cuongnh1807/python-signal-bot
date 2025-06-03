#!/usr/bin/env python3
"""
Simple Trading Bot Launcher
Provides easy interface to start the trading bot with different configurations
"""

import os
import sys
import argparse
from dotenv import load_dotenv


def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════╗
║          🚀 SIMPLE TRADING BOT 🚀            ║
║                                              ║
║  Multi-ticker live trading for Binance       ║
║  Focus: Setup, Entry, Stop Loss             ║
╚══════════════════════════════════════════════╝
    """
    print(banner)


def check_requirements():
    """Check if required files and modules exist"""
    required_files = [
        'futures_strategy.py',
        'binance_data_fetcher.py',
        'helpers/price.py'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    # Check required modules
    try:
        import pandas
        import numpy
        import requests
        from binance.client import Client
        from dotenv import load_dotenv
    except ImportError as e:
        print(f"❌ Missing required module: {e}")
        return False

    return True


def check_config():
    """Check configuration file"""
    if not os.path.exists('.env'):
        print("⚠️  No .env file found!")
        print("1. Copy: cp simple_trading_config.example .env")
        print("2. Edit .env with your API keys")
        return False

    load_dotenv()

    required_env = ['BINANCE_API_KEY', 'BINANCE_API_SECRET']
    missing_env = []

    for env_var in required_env:
        if not os.getenv(env_var) or os.getenv(env_var) == f'your_{env_var.lower()}_here':
            missing_env.append(env_var)

    if missing_env:
        print("❌ Missing or invalid environment variables:")
        for var in missing_env:
            print(f"   - {var}")
        print("\nPlease update your .env file with actual values.")
        return False

    return True


def show_config():
    """Show current configuration"""
    load_dotenv()

    print("\n📋 Current Configuration:")
    print("=" * 50)

    # Trading settings
    print("🔧 Trading Settings:")
    print(f"   Test Mode: {os.getenv('TEST_MODE', 'true')}")
    print(f"   Symbols: {os.getenv('TRADING_SYMBOLS', 'Not set')}")
    print(f"   Interval: {os.getenv('TRADING_INTERVAL', '15m')}")
    print(f"   Leverage: {os.getenv('LEVERAGE', '10')}x")

    # Risk management
    print("\n⚠️  Risk Management:")
    print(
        f"   Max Risk/Trade: {float(os.getenv('MAX_RISK_PER_TRADE', '0.02')) * 100}%")
    print(f"   Min Setup Quality: {os.getenv('MIN_SETUP_QUALITY', '70')}%")
    print(f"   Max Distance: {os.getenv('MAX_DISTANCE_PCT', '3')}%")

    # Telegram
    print("\n📱 Telegram:")
    telegram_enabled = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
    print(f"   Enabled: {telegram_enabled}")
    if telegram_enabled:
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        print(
            f"   Bot Token: {'✅ Set' if bot_token and 'your_' not in bot_token else '❌ Not set'}")
        print(
            f"   Chat ID: {'✅ Set' if chat_id and 'your_' not in chat_id else '❌ Not set'}")


def quick_setup():
    """Interactive quick setup"""
    print("\n🚀 Quick Setup")
    print("=" * 30)

    if not os.path.exists('.env'):
        import shutil
        shutil.copy('simple_trading_config.example', '.env')
        print("✅ Created .env file from template")

    print("\n📝 Please edit .env file with your settings:")
    print("1. Add your Binance API key and secret")
    print("2. Configure trading symbols")
    print("3. Set up Telegram (optional)")
    print("\nAfter editing .env, run: python start_simple_trading.py --start")


def main():
    parser = argparse.ArgumentParser(description='Simple Trading Bot Launcher')
    parser.add_argument('--start', action='store_true',
                        help='Start the trading bot')
    parser.add_argument('--config', action='store_true',
                        help='Show current configuration')
    parser.add_argument('--setup', action='store_true',
                        help='Interactive setup')
    parser.add_argument('--test', action='store_true', help='Force test mode')
    parser.add_argument('--live', action='store_true',
                        help='Force live mode (dangerous!)')
    parser.add_argument('--symbols', type=str,
                        help='Override trading symbols (comma-separated)')

    args = parser.parse_args()

    print_banner()

    # Show config
    if args.config:
        if check_config():
            show_config()
        return

    # Quick setup
    if args.setup:
        quick_setup()
        return

    # Start bot
    if args.start:
        print("🔍 Checking requirements...")

        if not check_requirements():
            print("\n❌ Requirements check failed!")
            return

        if not check_config():
            print("\n❌ Configuration check failed!")
            print("Run: python start_simple_trading.py --setup")
            return

        print("✅ All checks passed!")

        # Set environment overrides
        if args.test:
            os.environ['TEST_MODE'] = 'true'
            print("🧪 Forced TEST MODE")
        elif args.live:
            confirm = input("\n⚠️  LIVE MODE - Are you sure? (type 'yes'): ")
            if confirm.lower() != 'yes':
                print("❌ Cancelled")
                return
            os.environ['TEST_MODE'] = 'false'
            print("🔴 LIVE MODE ACTIVATED")

        if args.symbols:
            os.environ['TRADING_SYMBOLS'] = args.symbols
            print(f"📊 Override symbols: {args.symbols}")

        # Show final config
        show_config()

        # Confirm start
        print("\n" + "="*50)
        mode = "TEST" if os.getenv(
            'TEST_MODE', 'true').lower() == 'true' else "LIVE"
        print(f"🚀 Starting Simple Trading Bot in {mode} mode...")

        if mode == "LIVE":
            confirm = input(
                "⚠️  Final confirmation for LIVE trading (type 'START'): ")
            if confirm != 'START':
                print("❌ Cancelled")
                return

        print("\n🤖 Bot starting... Press Ctrl+C to stop")
        print("📝 Check simple_trading.log for detailed logs")
        print("-" * 50)

        # Import and start bot
        try:
            from simple_live_trading import main
            main()
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
        except ImportError as e:
            print(f"\n❌ Import error: {e}")
            print("Make sure simple_live_trading.py exists")
        except Exception as e:
            print(f"\n❌ Error starting bot: {e}")

    else:
        # Show help
        print("🔧 Usage:")
        print("   python start_simple_trading.py --setup    # Interactive setup")
        print("   python start_simple_trading.py --config   # Show configuration")
        print("   python start_simple_trading.py --start    # Start bot")
        print("   python start_simple_trading.py --start --test  # Force test mode")
        print("\n📖 For detailed guide, see: README_SIMPLE_TRADING.md")


if __name__ == "__main__":
    main()
