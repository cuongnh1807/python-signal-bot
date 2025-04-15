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
from binance.helpers import round_step_size
from helpers.price import adjust_precision
from binance_data_fetcher import BinanceDataFetcher

from macd_strategies.strategy import MacdRsiStrategy
from binance.client import Client
from telegram import TelegramNotifier
from time_synchronizer import initialize_time_sync, get_time_synchronizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
                 time_synchronizer=None,
                 telegram=None,
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
        self.telegram = TelegramNotifier(
            bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
            chat_id=os.getenv('TELEGRAM_CHAT_ID'),
            enabled=os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true',
            orders_topic_id=os.getenv('TELEGRAM_ORDERS_TOPIC_ID'),
            signals_topic_id=os.getenv('TELEGRAM_SIGNALS_TOPIC_ID')
        )

        # New parameter
        self.max_distance_to_current_price = max_distance_to_current_price

        # Add multi-timeframe parameters
        self.use_multi_timeframe = True
        self.higher_timeframe = '1h'
        self.lower_timeframe = self.interval  # Current timeframe

        # Add storage for multi-timeframe data
        self.htf_data = pd.DataFrame()  # Higher timeframe data

        logger.info(
            f"Initialized MACD Trading Bot for {symbol} on {interval} timeframe")

    def _get_interval_seconds(self, interval: str) -> int:
        """Convert interval string to seconds"""
        unit = interval[-1]
        value = int(interval[:-1])

        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 60 * 60
        elif unit == 'd':
            return value * 24 * 60 * 60
        elif unit == 'w':
            return value * 7 * 24 * 60 * 60
        else:
            return 3600

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
                    self.telegram.send_message(warning_msg)

            return min(usdt_available_balance, 650)
        except Exception as e:
            error_msg = f"Error getting USDT balance: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_message(error_msg)
            return 1000.0

        except Exception as e:
            error_msg = f"Error processing kline message: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_message(error_msg)

    def _run_analysis(self):
        """Run strategy analysis with multi-timeframe support"""
        try:

            if len(self.historical_data) < self.window_size:
                logger.warning(
                    f"Not enough data for analysis. Have {len(self.historical_data)}, need {self.window_size}")
                return

            # Basic single-timeframe analysis
            analysis = self.strategy.analyze_market(self.historical_data)

            # Multi-timeframe analysis if enabled
            mtf_analysis = None
            if self.use_multi_timeframe:
                # Fetch higher timeframe data if needed
                if len(self.htf_data) < self.window_size:
                    self._fetch_higher_timeframe_data()
                else:
                    # Update the last candle of HTF data if needed
                    current_htf_time = self._get_current_candle_time(
                        self.higher_timeframe)
                    if len(self.htf_data) > 0 and self.htf_data.index[-1] < current_htf_time:
                        self._fetch_higher_timeframe_data()

                # Run multi-timeframe analysis if we have data for both timeframes
                if len(self.htf_data) >= self.window_size:
                    data_dict = {
                        self.lower_timeframe: self.historical_data,
                        self.higher_timeframe: self.htf_data
                    }
                    mtf_analysis = self.strategy.analyze_multi_timeframe(
                        data_dict)
                    # Use best entries from multi-timeframe analysis if available
                    if mtf_analysis and mtf_analysis.get('best_entries'):
                        for entry in mtf_analysis['best_entries']:
                            self._process_trading_signal(entry)
                        return  # Skip processing signals from single timeframe

            # Check for reversals specifically (even without multi-timeframe)
            reversals = self.strategy.detect_timeframe_reversals(
                self.historical_data)
            if reversals['direction'] != 'NEUTRAL' and reversals['strength'] >= 8:
                logger.info(
                    f"Strong {reversals['direction']} reversal detected with strength {reversals['strength']}")

                # Create a trading signal based on the reversal
                signal = {
                    'signal_type': 'BUY' if reversals['direction'] == 'BULLISH' else 'SELL',
                    'price': self.current_price,
                    'strength': reversals['strength'],
                    'reason': f"Strong {reversals['direction']} reversal detected",
                    'reversal_details': reversals[f"{reversals['direction'].lower()}_reversals"]
                }

                # Process the reversal signal
                self._process_trading_signal(signal)

            for signal in analysis['signals']:
                self._process_trading_signal(signal)

        except Exception as e:
            error_msg = f"Error running analysis: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_message(error_msg)

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
                0.985 if signal['signal_type'] == 'BUY' else entry_price * 1.015

            # Calculate take profit level (only tp1)
            take_profit = {
                'tp1': entry_price * 1.015 if signal['signal_type'] == 'BUY' else entry_price * 0.985,
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
                'setup_type': 'MACD_RSI_EMA',
                'timestamp': datetime.now(),
                'status': 'PENDING',
                'volume_ratio': signal.get('volume_ratio', 0),
                'rsi': signal.get('rsi', 0),
                'ema': signal.get('ema', {})
            }

            # Print signal information
            self._print_signal_info(signal)

            # Place order
            if not self.test_mode:
                self._place_order_on_exchange(order)
                self._print_order_info(order, "PROD")

            else:
                logger.info(f"TEST MODE: Would place order: {order}")
                self._print_order_info(order, "TEST")

        except Exception as e:
            error_msg = f"Error processing trading signal: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_message(error_msg)

    def _place_order_on_exchange(self, order: Dict):
        """Place an order on Binance with proper time synchronization"""
        try:
            # Get timestamp and recvWindow parameters
            time_params = self.time_sync.get_timestamp_with_recvwindow()

            # Set leverage first
            self.client.futures_change_leverage(
                symbol=self.symbol,
                leverage=order['leverage'],
                **time_params  # Add timestamp and recvWindow
            )

            # Get updated timestamp for the next request
            time_params = self.time_sync.get_timestamp_with_recvwindow()

            # Calculate and adjust quantity precision
            quantity = adjust_precision(
                order['position_size'] / order['entry_price'], self.symbol_precision['quantityPrecision'])

            if quantity != 0:
                # Determine order type and parameters
                if order['entry_type'] == 'MARKET':
                    # Place market order
                    response = self.client.futures_create_order(
                        symbol=self.symbol,
                        side='BUY' if order['side'] == 'LONG' else 'SELL',
                        type='MARKET',
                        quantity=quantity,
                        **time_params  # Add timestamp and recvWindow
                    )

                    # Get filled price
                    order['actual_entry_price'] = order['entry_price'] or float(
                        response['avgPrice'])
                    order['order_id'] = response['orderId']
                    order['status'] = 'ACTIVE'

                    # Log order placement
                    logger.info(
                        f"Market order placed: {response['orderId']} at {order['actual_entry_price']}")

                    # Print order information
                    self._print_order_info(order, "PLACED")

                    # Send notification if telegram is configured
                    if self.telegram:
                        self.telegram.send_message(
                            f"Order placed: {order['side']} {self.symbol} at {order['actual_entry_price']}")

                    # Place stop loss
                    self._place_stop_loss(order)

                    # Place take profit
                    self._place_take_profit(order)

                    # Add to open positions
                    self.open_positions[response['orderId']] = order

                else:  # LIMIT order
                    # Place limit order
                    response = self.client.futures_create_order(
                        symbol=self.symbol,
                        side='BUY' if order['side'] == 'LONG' else 'SELL',
                        type='LIMIT',
                        quantity=quantity,
                        timeInForce='GTC',
                        price=round_step_size(order['entry_price'], float(
                            self.symbol_precision['tickSize'])),
                        **time_params  # Add timestamp and recvWindow
                    )

                    order['order_id'] = response['orderId']
                    order['status'] = 'PENDING'

                    # Log order placement
                    logger.info(
                        f"Limit order placed: {response['orderId']} at {order['entry_price']}")

                    # Print order information
                    self._print_order_info(order, "PENDING")

                    # Add to active orders
                    self.active_orders[response['orderId']] = order

        except Exception as e:
            error_msg = f"Error placing order on exchange: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_message(error_msg)

    def _place_stop_loss(self, order: Dict):
        """Place stop loss order"""
        try:
            # Calculate quantity
            if not order.get('actual_entry_price'):
                order['actual_entry_price'] = self.current_price

            quantity = order['position_size'] / order['actual_entry_price']
            quantity = adjust_precision(
                quantity, self.symbol_precision['quantityPrecision'])

            # Adjust price precision
            stop_price = round_step_size(
                order['stop_loss'], float(self.symbol_precision['tickSize']))

            # Place stop loss order
            response = self.client.futures_create_order(
                symbol=self.symbol,
                side='SELL' if order['side'] == 'LONG' else 'BUY',
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=stop_price,
                closePosition=True,
                **self.time_sync.get_timestamp_with_recvwindow()
            )

            order['stop_loss_order_id'] = response['orderId']
            logger.info(f"Stop loss placed at {stop_price}")

        except Exception as e:
            error_msg = f"Error placing stop loss: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_message(error_msg)

    def _place_take_profit(self, order: Dict):
        """Place take profit order (only tp1)"""
        try:
            # Calculate quantity
            if not order.get('actual_entry_price'):
                order['actual_entry_price'] = self.current_price

            quantity = order['position_size'] / order['actual_entry_price']
            quantity = adjust_precision(
                quantity, self.symbol_precision['quantityPrecision'])

            # Get tp1 price
            tp_price = order['take_profit']['tp1']
            tp_price = round_step_size(
                tp_price, float(self.symbol_precision['tickSize']))

            # Place take profit order
            response = self.client.futures_create_order(
                symbol=self.symbol,
                side='SELL' if order['side'] == 'LONG' else 'BUY',
                type='TAKE_PROFIT_MARKET',
                quantity=quantity,
                stopPrice=tp_price,
                **self.time_sync.get_timestamp_with_recvwindow()
            )

            order['take_profit_order_id'] = response['orderId']
            logger.info(f"Take profit placed at {tp_price}")

        except Exception as e:
            error_msg = f"Error placing take profit: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_message(error_msg)

    def _adjust_price_precision(self, price):
        """Adjust price precision according to symbol rules"""
        if not self.symbol_precision:
            return price

        tick_size = float(self.symbol_precision.get('tickSize', 0.1))
        return round(price / tick_size) * tick_size

    def _adjust_quantity_precision(self, quantity):
        """Adjust quantity precision according to symbol rules"""
        if not self.symbol_precision:
            return quantity

        step_size = float(self.symbol_precision.get('stepSize', 0.001))
        return round(quantity / step_size) * step_size

    def _print_order_info(self, order, status):
        """Print information about an order"""
        # Create a formatted string for order details
        entry_price = order.get('actual_entry_price', order['entry_price'])
        order_type = "MARKET" if order.get(
            'entry_type') == 'MARKET' else "LIMIT"

        order_info = (
            f"\n{'='*50}\n"
            f"ORDER {status}: {order['side']} {self.symbol}\n"
            f"Type: {order_type}\n"
            f"Entry Price: {entry_price}\n"
            f"Stop Loss: {order['stop_loss']}\n"
            f"Take Profit: {order['take_profit']['tp1']}\n"
            f"Position Size: {order['position_size']}\n"
            f"Leverage: {order['leverage']}x\n"
            f"Setup Quality: {order.get('strength', 0)}\n"
            f"Reason: {order.get('reason', 'N/A')}\n"
            f"{'='*50}\n"
        )

        logger.info(order_info)
        if status != "TEST":
            self.telegram.send_message(order_info, topic_id=os.getenv(
                'TELEGRAM_ORDERS_TOPIC_ID'))

    def _print_signal_info(self, signal):
        """Print information about a trading signal"""
        # Create a formatted string for signal details
        signal_type = signal['signal_type']
        price = signal['price']
        strength = signal.get('strength', 0)

        signal_info = (
            f"\n{'='*50}\n"
            f"SIGNAL DETECTED: {signal_type} {self.symbol}\n"
            f"Price: {price}\n"
            f"Strength: {strength:.2f}\n"
            f"RSI: {signal.get('rsi', 'N/A')}\n"
            f"Volume Ratio: {signal.get('volume_ratio', 'N/A')}\n"
            f"Reason: {signal.get('reason', 'N/A')}\n"
            f"{'='*50}\n"
        )

        self.telegram.send_message(signal_info)

    def _fetch_higher_timeframe_data(self):
        """Fetch higher timeframe data for multi-timeframe analysis"""
        logger.info(
            f"Fetching higher timeframe data ({self.higher_timeframe}) for {self.symbol}")

        try:
            # Calculate start time for higher timeframe
            htf_interval_seconds = self._get_interval_seconds(
                self.higher_timeframe)
            start_time = datetime.now() - timedelta(
                seconds=htf_interval_seconds * (self.window_size + 5))

            # Fetch data
            self.htf_data = self.data_fetcher.get_historical_klines(
                symbol=self.symbol,
                interval=self.higher_timeframe,
                start_time=start_time,
            )

            logger.info(
                f"Fetched {len(self.htf_data)} higher timeframe candles")

            return self.htf_data
        except Exception as e:
            error_msg = f"Error fetching higher timeframe data: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_message(error_msg)
            return pd.DataFrame()

    def _get_current_candle_time(self, interval):
        """Get the timestamp of the current candle for a given interval"""
        now = datetime.now()

        if interval.endswith('m'):
            mins = int(interval[:-1])
            return now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % mins)
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=now.hour % hours)
        elif interval.endswith('d'):
            return now.replace(hour=0, minute=0, second=0, microsecond=0)

        return now

    def _fetch_initial_data(self):
        """Fetch initial historical data for all timeframes"""
        logger.info(f"Fetching initial data for {self.symbol}")

        try:
            # Fetch data for current timeframe
            start_time = datetime.now() - timedelta(
                seconds=self.analysis_interval_seconds * (self.window_size + 10))

            self.historical_data = self.data_fetcher.get_historical_klines(
                symbol=self.symbol,
                interval=self.interval,
                start_time=start_time,
            )

            logger.info(
                f"Fetched {len(self.historical_data)} {self.interval} candles")

            # Fetch higher timeframe data if multi-timeframe is enabled
            if self.use_multi_timeframe:
                self._fetch_higher_timeframe_data()

            # Set current price
            if not self.historical_data.empty:
                self.current_price = self.historical_data['close'].iloc[-1]

        except Exception as e:
            error_msg = f"Error fetching initial data: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_message(error_msg)
            raise

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
            # self.status_thread = threading.Thread(
            #     target=self._status_check_loop)
            # self.status_thread.daemon = True
            # self.status_thread.start()

            # Start market analysis thread
            self.analysis_thread = threading.Thread(
                target=self._cronjob_analysis_loop)
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
                self.telegram.send_message(error_msg)
            self.stop()

    def _update_capital(self):
        """Update capital based on current USDT balance"""
        try:
            if not self.test_mode:
                # Get new USDT balance
                new_balance = self._get_usdt_balance()

                # Update capital in strategy
                self.strategy.capital = new_balance
                self.initial_capital = new_balance

                logger.info(f"Updated capital to {new_balance} USDT")

        except Exception as e:
            error_msg = f"Error updating capital: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_message(error_msg)

    def _cronjob_analysis_loop(self):
        """Run market analysis based on cronjob schedule (every 3 minutes)"""
        while self.running:
            try:
                self._update_capital()

                now = datetime.now()
                current_minute = now.minute

                # Check if current minute is divisible by 3 (0,3,6,9,12,15,18,21,...57)
                if current_minute % 5 == 0:
                    logger.info(f"Running analysis at minute {current_minute}")

                # Fetch newest data
                    self._fetch_latest_data()

                # Run analysis
                    self._run_analysis()

                # Sleep for 3 minutes to avoid multiple runs in the same minute
                    time.sleep(180)  # 3 minutes = 180 seconds
                else:
                    # Calculate time until next 3-minute interval
                    minutes_to_next = 3 - (current_minute % 3)
                    seconds_to_next = minutes_to_next * 60 - now.second

                    # Add a small buffer
                    seconds_to_next += 2

                    logger.info(f"Next analysis in {seconds_to_next} seconds")
                    time.sleep(seconds_to_next)

            except Exception as e:
                error_msg = f"Error in cronjob analysis loop: {str(e)}"
                logger.error(error_msg)
                self.telegram.send_message(error_msg)
                time.sleep(60)  # Wait a minute before trying again

    def _fetch_latest_data(self):
        """Fetch the latest market data"""
        try:
            # Calculate start time based on window size
            start_time = datetime.now() - timedelta(
                seconds=self._get_interval_seconds(self.interval) * (self.window_size))

            # Fetch data
            self.historical_data = self.data_fetcher.get_historical_klines(
                symbol=self.symbol,
                interval=self.interval,
                start_time=start_time,
            )

            # Update current price
            if not self.historical_data.empty:
                self.current_price = self.historical_data['close'].iloc[-1]

            logger.info(
                f"Updated historical data. Current price: {self.current_price}")
        except Exception as e:
            logger.error(f"Error fetching latest data: {str(e)}")

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
