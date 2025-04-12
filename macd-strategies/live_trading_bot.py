import time
import logging
import json
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import os
from binance_data_fetcher import BinanceDataFetcher

from .strategy import MacdRsiStrategy
from binance.client import Client
from time_synchronizer import initialize_time_sync, get_time_synchronizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("macd_trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MacdTradingBot:
    """
    Live trading bot that implements the MACD-RSI strategy on Binance.
    Uses WebSockets for real-time data and executes trades automatically.
    """

    def __init__(self,
                 client: Client,
                 symbol: str,
                 interval: str = '15m',
                 max_risk_per_trade: float = 0.02,
                 leverage: int = 20,
                 window_size: int = 100,
                 min_setup_quality: float = 70.0,
                 min_volume_ratio: float = 3.0,
                 max_distance_to_current_price: float = 5.0,
                 test_mode: bool = True,
                 telegram: Optional[object] = None,
                 time_synchronizer=None,
                 symbol_precision: Dict = None):
        """
        Initialize the MACD-RSI trading bot.
        """
        self.symbol = symbol
        self.interval = interval
        self.max_risk_per_trade = max_risk_per_trade
        self.leverage = leverage
        self.window_size = window_size
        self.min_setup_quality = min_setup_quality
        self.min_volume_ratio = min_volume_ratio
        self.test_mode = test_mode
        self.symbol_precision = symbol_precision
        self.last_signal = None
        self.last_analysis_time = None
        self.in_position = False
        self.tickerInfo = None

        # Initialize time synchronizer
        if time_synchronizer:
            self.time_sync = time_synchronizer
        else:
            try:
                self.time_sync = get_time_synchronizer()
            except RuntimeError:
                self.time_sync = initialize_time_sync(
                    os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

        # Initialize Binance client
        self.client = client

        # Initialize data fetcher
        self.data_fetcher = BinanceDataFetcher(client=client)

        # Get USDT balance from Futures account
        self.initial_capital = self._get_usdt_balance()
        self.tickerInfo = self.client.get_symbol_info(self.symbol)

        # Initialize MACD-RSI strategy
        self.strategy = MacdRsiStrategy(
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            volume_ma_period=20,
            min_volume_ratio=2.0
        )

        # Initialize data storage
        self.historical_data = pd.DataFrame()
        self.current_price = 0.0

        # Initialize order tracking
        self.active_orders = {}
        self.open_positions = {}

        # Initialize control flags
        self.running = False
        self.analysis_interval_seconds = self._get_interval_seconds(interval)

        # Initialize Telegram notifier
        self.telegram = telegram

        # New parameter
        self.max_distance_to_current_price = max_distance_to_current_price

        logger.info(
            f"Initialized MACD Trading Bot for {symbol} on {interval} timeframe")

    def _get_usdt_balance(self) -> float:
        """Get USDT balance from Futures account"""
        try:
            if self.test_mode:
                logger.info("Test mode: Using default balance of 1000 USDT")
                return 1000.0

            account_info = self.client.futures_account_balance()
            usdt_balance = 0.0
            usdt_available_balance = 0.0

            for asset in account_info:
                if asset['asset'] == 'USDT':
                    usdt_balance = float(asset['balance'])
                    usdt_available_balance = float(asset['availableBalance'])
                    break

            logger.info(
                f"Current USDT balance: {usdt_balance}, Available: {usdt_available_balance}")

            if usdt_available_balance < 10.0:
                warning_msg = f"WARNING: Low available USDT balance ({usdt_available_balance})"
                logger.warning(warning_msg)
                if self.telegram:
                    self.telegram.notify_error(warning_msg)

            return min(usdt_available_balance, 650)
        except Exception as e:
            error_msg = f"Error getting USDT balance: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.notify_error(error_msg)
            return 1000.0

    def _process_kline_message(self, msg):
        """Process kline message from WebSocket"""
        try:
            if 'k' in msg:
                kline = msg['k']
                is_candle_closed = kline['x']

                if not is_candle_closed:
                    self.current_price = float(kline['c'])
                    return

                # Extract candle data
                timestamp = datetime.fromtimestamp(kline['t'] / 1000)
                new_candle = pd.DataFrame({
                    'open': [float(kline['o'])],
                    'high': [float(kline['h'])],
                    'low': [float(kline['l'])],
                    'close': [float(kline['c'])],
                    'volume': [float(kline['v'])]
                }, index=[timestamp])

                # Update historical data
                self.historical_data = pd.concat(
                    [self.historical_data, new_candle])

                if len(self.historical_data) > self.window_size + 10:
                    self.historical_data = self.historical_data.iloc[-(
                        self.window_size + 10):]

                self.current_price = float(kline['c'])
                logger.info(
                    f"New candle closed: {timestamp}, Close: {self.current_price}")

                # Run analysis on new candle
                self._run_analysis()

        except Exception as e:
            error_msg = f"Error processing kline message: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.notify_error(error_msg)

    def _run_analysis(self):
        """Run strategy analysis and generate orders"""
        try:
            if len(self.historical_data) < self.window_size:
                logger.warning(
                    f"Not enough data for analysis. Have {len(self.historical_data)}, need {self.window_size}")
                return

            # Get analysis from MACD-RSI strategy
            analysis = self.strategy.analyze_market(self.historical_data)

            # Process signals
            for signal in analysis['signals']:
                self._process_trading_signal(signal)

            # Send analysis to Telegram
            if self.telegram:
                self.telegram.notify_common_indicators(analysis)

        except Exception as e:
            error_msg = f"Error running analysis: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.notify_error(error_msg)

    def _process_trading_signal(self, signal: Dict):
        """Process trading signal and create order"""
        try:
            # Calculate position size and risk
            entry_price = signal['price']

            # Skip if price is too far from current price
            price_distance_percent = abs(
                (entry_price / self.current_price) - 1) * 100
            if price_distance_percent > self.max_distance_to_current_price:
                logger.info(
                    f"Skipping signal: Entry price too far from current price ({price_distance_percent:.2f}%)")
                return

            # Calculate stop loss (2% from entry)
            stop_loss = entry_price * \
                0.98 if signal['signal_type'] == 'BUY' else entry_price * 1.02

            # Calculate take profit levels
            take_profit = {
                'tp1': entry_price * 1.02 if signal['signal_type'] == 'BUY' else entry_price * 0.98,
                'tp2': entry_price * 1.04 if signal['signal_type'] == 'BUY' else entry_price * 0.96,
                'tp3': entry_price * 1.06 if signal['signal_type'] == 'BUY' else entry_price * 0.94
            }

            # Calculate position size based on risk
            risk_amount = self.initial_capital * self.max_risk_per_trade
            position_size = (risk_amount * self.leverage) / \
                abs(entry_price - stop_loss)

            # Create order
            order = {
                'symbol': self.symbol,
                'side': 'LONG' if signal['signal_type'] == 'BUY' else 'SHORT',
                'entry_type': 'MARKET',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'leverage': self.leverage,
                'setup_quality': signal['strength'],
                'setup_type': 'MACD_RSI',
                'timestamp': datetime.now(),
                'status': 'PENDING',
                'volume_ratio': signal['volume_ratio'],
                'rsi': signal['rsi'],
                'macd': signal['macd']
            }

            # Place order
            if not self.test_mode:
                self._place_order_on_exchange(order)
            else:
                logger.info(f"TEST MODE: Would place order: {order}")

            # Notify about new order
            if self.telegram:
                self.telegram.notify_order_created(order, self.current_price)

        except Exception as e:
            error_msg = f"Error processing trading signal: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.notify_error(error_msg)

    # ... [Keep all other methods from live_trading_bot.py unchanged] ...
    # Including _place_order_on_exchange, _check_order_status, start, stop, etc.

    def start(self):
        """Start the trading bot"""
        if self.running:
            logger.warning("Trading bot is already running")
            return

        try:
            logger.info("Starting MACD-RSI trading bot...")
            self.running = True

            # Fetch initial data
            self._fetch_initial_data()

            # Start order status checking thread
            self.status_thread = threading.Thread(
                target=self._status_check_loop)
            self.status_thread.daemon = True
            self.status_thread.start()

            # Start market analysis thread
            self.analysis_thread = threading.Thread(
                target=self._analysis_loop)
            self.analysis_thread.daemon = True
            self.analysis_thread.start()

            logger.info(
                f"MACD-RSI trading bot started for {self.symbol} on {self.interval}")

            if self.telegram:
                self.telegram.send_message(
                    f"🚀 <b>MACD-RSI Trading Bot Started</b>\n\n"
                    f"Symbol: <b>{self.symbol}</b>\n"
                    f"Timeframe: <b>{self.interval}</b>\n"
                    f"Current Price: <b>${self.current_price:.2f}</b>\n"
                    f"Mode: <b>{'Test' if self.test_mode else 'Live'}</b>"
                )

        except Exception as e:
            error_msg = f"Error starting trading bot: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.notify_error(error_msg)
            self.stop()

    def stop(self):
        """Stop the trading bot"""
        if not self.running:
            return

        logger.info("Stopping MACD-RSI trading bot...")
        self.running = False

        if self.telegram:
            self.telegram.send_message("🛑 <b>MACD-RSI Trading Bot Stopped</b>")

        logger.info("Trading bot stopped")


# Main execution
if __name__ == "__main__":
    load_dotenv()

    # Load configuration from environment variables
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')
    symbol = os.environ.get('TRADING_SYMBOL', 'BTCUSDT')
    interval = os.environ.get('TRADING_INTERVAL', '15m')
    max_risk_per_trade = float(os.environ.get('MAX_RISK_PER_TRADE', '0.02'))
    leverage = int(os.environ.get('LEVERAGE', '20'))
    window_size = int(os.environ.get('WINDOW_SIZE', '100'))
    min_setup_quality = float(os.environ.get('MIN_SETUP_QUALITY', '70.0'))
    min_volume_ratio = float(os.environ.get('MIN_VOLUME_RATIO', '3.0'))
    test_mode = os.environ.get('TEST_MODE', 'true').lower() == 'true'

    # Initialize Binance client
    client = Client(api_key, api_secret)

    # Create and start trading bot
    bot = MacdTradingBot(
        client=client,
        symbol=symbol,
        interval=interval,
        max_risk_per_trade=max_risk_per_trade,
        leverage=leverage,
        window_size=window_size,
        min_setup_quality=min_setup_quality,
        min_volume_ratio=min_volume_ratio,
        test_mode=test_mode
    )

    # Start the bot
    bot.start()

    # Keep the main thread running
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
        bot.stop()
