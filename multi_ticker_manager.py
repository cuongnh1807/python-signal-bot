import os
import time
import json
import logging
from typing import Dict, List
from live_trading_bot import LiveTradingBot, TelegramNotifier
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


class MultiTickerManager:
    """
    Quản lý nhiều bot giao dịch cho các ticker khác nhau
    """

    def __init__(self, config_path: str, telegram_config: Dict = None):
        """
        Khởi tạo MultiTickerManager

        Parameters:
        -----------
        config_path: Đường dẫn đến file cấu hình JSON
        telegram_config: Cấu hình Telegram (nếu không có sẽ lấy từ file cấu hình)
        """
        self.config_path = config_path
        self.bots = {}
        self.telegram_config = telegram_config
        self.telegram = None

        # Tải cấu hình
        self.load_config()

    def load_config(self):
        """Tải cấu hình từ file JSON"""
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
            self.api_key = os.environ.get('BINANCE_API_KEY')
            self.api_secret = os.environ.get('BINANCE_API_SECRET')
            self.test_mode = os.environ.get(
                'TEST_MODE', 'true').lower() == 'true'

            # Cấu hình cho từng ticker
            self.ticker_configs = config.get('tickers', [])

            logger.info(
                f"Loaded configuration with {len(self.ticker_configs)} tickers")

        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def start_all(self):
        """Khởi động tất cả các bot giao dịch"""
        for ticker_config in self.ticker_configs:
            try:
                symbol = ticker_config.get('symbol')
                if not symbol:
                    logger.warning("Skipping ticker config without symbol")
                    continue

                # Kiểm tra xem bot đã tồn tại chưa
                if symbol in self.bots:
                    logger.warning(
                        f"Bot for {symbol} already running, skipping")
                    continue

                # Tạo bot mới
                bot = LiveTradingBot(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    symbol=symbol,
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

                # Khởi động bot
                bot.start()

                # Lưu bot vào danh sách
                self.bots[symbol] = bot

                logger.info(f"Started bot for {symbol}")

                # Đợi một chút trước khi khởi động bot tiếp theo để tránh quá tải API
                time.sleep(2)

            except Exception as e:
                logger.error(
                    f"Error starting bot for {ticker_config.get('symbol', 'unknown')}: {str(e)}")

        logger.info(f"Started {len(self.bots)} bots")

    def stop_all(self):
        """Dừng tất cả các bot giao dịch"""
        for symbol, bot in list(self.bots.items()):
            try:
                bot.stop()
                logger.info(f"Stopped bot for {symbol}")
            except Exception as e:
                logger.error(f"Error stopping bot for {symbol}: {str(e)}")

        self.bots = {}
        logger.info("All bots stopped")

    def restart_bot(self, symbol: str):
        """Khởi động lại bot cho một ticker cụ thể"""
        if symbol in self.bots:
            try:
                # Dừng bot hiện tại
                self.bots[symbol].stop()

                # Tìm cấu hình cho ticker này
                ticker_config = next(
                    (config for config in self.ticker_configs if config.get('symbol') == symbol), None)

                if not ticker_config:
                    logger.error(f"Cannot find configuration for {symbol}")
                    return

                # Tạo bot mới
                bot = LiveTradingBot(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    symbol=symbol,
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

                # Khởi động bot
                bot.start()

                # Cập nhật bot trong danh sách
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

        # Dừng tất cả các bot hiện tại
        self.stop_all()

        # Tải cấu hình mới
        self.load_config()

        # Khởi động lại tất cả các bot
        self.start_all()

        logger.info("Configuration updated and all bots restarted")

    def get_status(self) -> Dict:
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
        """Lấy danh sách vị thế mở của tất cả các bot"""
        positions = {}
        for symbol, bot in self.bots.items():
            positions[symbol] = bot.get_open_positions()

        return positions

    def get_active_orders(self) -> Dict[str, List]:
        """Lấy danh sách lệnh đang hoạt động của tất cả các bot"""
        orders = {}
        for symbol, bot in self.bots.items():
            orders[symbol] = bot.get_active_orders()

        return orders
