import os
import time
import argparse
import logging
from dotenv import load_dotenv
from multi_ticker_manager import MultiTickerManager

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-ticker trading bot')
    parser.add_argument(
        '--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Override config path from environment if available
    config_path = os.environ.get('CONFIG_PATH', args.config)

    # Create and start the multi-ticker manager
    manager = MultiTickerManager(config_path=config_path)

    # Override test mode if specified in command line
    if args.test:
        manager.test_mode = True

    try:
        # Start all bots
        manager.start_all()

        # Keep the main thread running
        logger.info("All bots started. Press Ctrl+C to stop.")

        while True:
            # Check status every 15 minutes
            time.sleep(60 * 15)
            status = manager.get_status()
            logger.info(
                f"Manager status: {status['total_bots']} bots, {status['running_bots']} running")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
        manager.stop_all()

    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        manager.stop_all()


if __name__ == "__main__":
    main()
