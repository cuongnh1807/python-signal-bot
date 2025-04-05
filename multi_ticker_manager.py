import os
import time
import json
import logging
import concurrent.futures
import threading
from typing import Dict, List
from live_trading_bot import LiveTradingBot, TelegramNotifier
from binance.client import Client
# from telegram_notifier import TelegramNotifier

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_ticker_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_symbol_precision(client, symbols):
    info = client.futures_exchange_info()
    objs = {}
    for item in info['symbols']:
        if item['symbol'] in symbols:
            objs[item['symbol']] = {
                'pricePrecision': item['pricePrecision'],
                'quantityPrecision': item['quantityPrecision'],
                'tickSize': item['filters'][0]['tickSize']
            }
    return objs


class MultiTickerManager:
    def __init__(self, config_path: str, telegram_config: Dict = None):

        self.config_path = config_path
        self.bots = {}
        self.telegram_config = telegram_config
        self.telegram = None
        self.bots_lock = threading.Lock()  # Add lock for thread safety

        # Tải cấu hình
        self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # Khởi tạo Telegram notifier
            self.telegram = TelegramNotifier(
                bot_token=os.environ.get('TELEGRAM_BOT_TOKEN', ''),
                chat_id=os.environ.get('TELEGRAM_CHAT_ID', ''),
                enabled=os.environ.get(
                    'TELEGRAM_ENABLED', 'true').lower() == 'true',
                orders_topic_id=os.environ.get('TELEGRAM_ORDERS_TOPIC_ID', ''),
                signals_topic_id=os.environ.get(
                    'TELEGRAM_SIGNALS_TOPIC_ID', '')
            )

            # Cấu hình chung
            self.api_secret = os.environ.get('BINANCE_API_SECRET')
            self.test_mode = os.environ.get(
                'TEST_MODE', 'true').lower() == 'true'
            self.client = Client(os.environ.get(
                'BINANCE_API_KEY'), os.environ.get('BINANCE_API_SECRET'))

            # Cấu hình cho từng ticker
            self.ticker_configs = config.get('tickers', [])
            self.symbol_precision = get_symbol_precision(
                self.client, [ticker_config.get('symbol') for ticker_config in self.ticker_configs])

            logger.info(
                f"Loaded configuration with {len(self.ticker_configs)} tickers")

        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def _start_bot_worker(self, ticker_config):
        """Start a single bot in a separate thread"""
        try:
            symbol = ticker_config.get('symbol')
            if not symbol:
                logger.warning("Skipping ticker config without symbol")
                return None, None

            # Check if bot is already running (with lock)
            with self.bots_lock:
                if symbol in self.bots:
                    logger.warning(
                        f"Bot for {symbol} already running, skipping")
                    return None, None

            # Create new bot
            bot = LiveTradingBot(
                client=self.client,
                symbol=symbol,
                symbol_precision=self.symbol_precision[symbol],
                interval=ticker_config.get('interval', '15m'),
                max_risk_per_trade=ticker_config.get(
                    'max_risk_per_trade', 0.02),
                leverage=ticker_config.get('leverage', 20),
                window_size=ticker_config.get('window_size', 100),
                min_setup_quality=ticker_config.get('min_setup_quality', 70.0),
                min_volume_ratio=ticker_config.get('min_volume_ratio', 3.0),
                max_distance_to_current_price=ticker_config.get(
                    'max_distance_to_current_price', 5.0),
                test_mode=self.test_mode,
                telegram=self.telegram
            )

            # Start the bot
            bot.start()

            logger.info(f"Started bot for {symbol}")

            return symbol, bot

        except Exception as e:
            logger.error(
                f"Error starting bot for {ticker_config.get('symbol', 'unknown')}: {str(e)}")
            return None, None

    def start_all(self):
        """Start all bots in parallel using ThreadPool"""
        # Use ThreadPoolExecutor to start bots in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all bot start tasks to the executor
            future_to_config = {executor.submit(
                self._start_bot_worker, config): config for config in self.ticker_configs}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_config):
                symbol, bot = future.result()
                if symbol and bot:
                    # Add bot to the dictionary with lock protection
                    with self.bots_lock:
                        self.bots[symbol] = bot

        logger.info(f"Started {len(self.bots)} bots")

    def stop_all(self):
        """Stop all trading bots"""
        with self.bots_lock:
            bots_to_stop = list(self.bots.items())

        for symbol, bot in bots_to_stop:
            try:
                bot.stop()
                logger.info(f"Stopped bot for {symbol}")
            except Exception as e:
                logger.error(f"Error stopping bot for {symbol}: {str(e)}")

        with self.bots_lock:
            self.bots = {}
        logger.info("All bots stopped")

    def restart_bot(self, symbol: str):
        """Restart a bot for a specific ticker"""
        bot_to_restart = None

        with self.bots_lock:
            if symbol in self.bots:
                bot_to_restart = self.bots[symbol]

        if bot_to_restart:
            try:
                # Stop current bot
                bot_to_restart.stop()

                # Find configuration for this ticker
                ticker_config = next(
                    (config for config in self.ticker_configs if config.get('symbol') == symbol), None)

                if not ticker_config:
                    logger.error(f"Cannot find configuration for {symbol}")
                    return

                bot = LiveTradingBot(
                    client=self.client,
                    symbol=symbol,
                    symbol_precision=self.symbol_precision[symbol],
                    interval=ticker_config.get('interval', '15m'),
                    max_risk_per_trade=ticker_config.get(
                        'max_risk_per_trade', 0.02),
                    leverage=ticker_config.get('leverage', 20),
                    window_size=ticker_config.get('window_size', 100),
                    min_setup_quality=ticker_config.get(
                        'min_setup_quality', 70.0),
                    min_volume_ratio=ticker_config.get(
                        'min_volume_ratio', 3.0),
                    max_distance_to_current_price=ticker_config.get(
                        'max_distance_to_current_price', 5.0),
                    test_mode=self.test_mode,
                    telegram=self.telegram
                )

                bot.start()

                with self.bots_lock:
                    self.bots[symbol] = bot

                logger.info(f"Restarted bot for {symbol}")

            except Exception as e:
                logger.error(f"Error restarting bot for {symbol}: {str(e)}")
        else:
            logger.warning(f"No bot running for {symbol}")

    def update_config(self, new_config_path: str = None):
        """Cập nhật cấu hình và khởi động lại các bot"""
        if new_config_path:
            self.config_path = new_config_path

        self.stop_all()

        self.load_config()

        self.start_all()

        logger.info("Configuration updated and all bots restarted")

    def get_status(self) -> Dict:
        with self.bots_lock:
            status = {
                'total_bots': len(self.bots),
                'running_bots': sum(1 for bot in self.bots.values() if bot.running),
                'test_mode': self.test_mode,
                'bots': {}
            }

            for symbol, bot in self.bots.items():
                status['bots'][symbol] = bot.get_status()

        return status

    def get_open_positions(self) -> Dict[str, List]:
        positions = {}

        with self.bots_lock:
            bot_items = list(self.bots.items())

        for symbol, bot in bot_items:
            positions[symbol] = bot.get_open_positions()

        return positions

    def get_active_orders(self) -> Dict[str, List]:
        orders = {}

        with self.bots_lock:
            bot_items = list(self.bots.items())

        for symbol, bot in bot_items:
            orders[symbol] = bot.get_active_orders()

        return orders
