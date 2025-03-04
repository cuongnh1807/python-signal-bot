import time
import logging
import json
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import os
import signal
import sys
import requests
import asyncio
from binance import AsyncClient, BinanceSocketManager

# Import strategy components
from binance_data_fetcher import BinanceDataFetcher
from futures_strategy import FuturesStrategy
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Handles sending notifications to Telegram.
    """

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        """
        Initialize the Telegram notifier.

        Parameters:
        -----------
        bot_token: Telegram bot token
        chat_id: Telegram chat ID to send messages to
        enabled: Whether notifications are enabled
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.base_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        if enabled:
            self.send_message("🤖 Trading Bot initialized and ready.")
            logger.info("Telegram notifications enabled")
        else:
            logger.info("Telegram notifications disabled")

    def send_message(self, message: str, parse_mode: str = "HTML"):
        """Send a message to the Telegram chat"""
        if not self.enabled:
            return

        try:
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }

            response = requests.post(self.base_url, data=data)

            if response.status_code != 200:
                logger.error(
                    f"Failed to send Telegram message: {response.text}")

        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")

    def notify_order_created(self, order: Dict):
        """Send notification about a new order"""
        if not self.enabled:
            return

        side = order['side']
        symbol = order['symbol']
        entry_type = order['entry_type']
        entry_price = order['entry_price']
        stop_loss = order['stop_loss']
        setup_quality = order['setup_quality']
        setup_type = order['setup_type']
        volume_ratio = order.get('volume_ratio', 'N/A')
        position_size = order.get('position_size', 0)
        leverage = order.get('leverage', 20)
        margin = order.get('margin_amount', 0)

        # Calculate risk-reward for TP2
        risk = abs(entry_price - stop_loss)
        tp2 = order['take_profit'].get('tp2', 0)
        rr = abs(tp2 - entry_price) / risk if risk > 0 else 0

        message = (
            f"🔔 <b>New {side} Order Created</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Type: <b>{entry_type}</b>\n"
            f"Setup: <b>{setup_type}</b> (Quality: {setup_quality:.1f}%)\n"
            f"Entry: <b>${entry_price:.2f}</b>\n"
            f"Stop Loss: <b>${stop_loss:.2f}</b>\n"
            f"Risk-Reward (TP2): <b>{rr:.2f}</b>\n"
            f"Volume Ratio: <b>{volume_ratio:.1f}x</b>\n"
            f"Position Size: <b>${position_size:.2f}</b> ({leverage}x)\n"
            f"Margin: <b>${margin:.2f}</b>"
        )

        self.send_message(message)

    def notify_order_filled(self, order: Dict):
        """Send notification about a filled order"""
        if not self.enabled:
            return

        side = order['side']
        symbol = order['symbol']
        entry_price = order['actual_entry_price']
        position_size = order.get('position_size', 0)

        message = (
            f"✅ <b>{side} Order Filled</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Entry Price: <b>${entry_price:.2f}</b>\n"
            f"Position Size: <b>${position_size:.2f}</b>"
        )

        self.send_message(message)

    def notify_position_closed(self, position: Dict):
        """Send notification about a closed position"""
        if not self.enabled:
            return

        side = position['side']
        symbol = position['symbol']
        entry_price = position['actual_entry_price']
        exit_price = position['exit_price']
        exit_reason = position['exit_reason']
        profit = position.get('profit', 0)
        profit_percent = position.get('profit_percent', 0)

        # Determine emoji based on profit
        if profit > 0:
            emoji = "🟢"
        elif profit < 0:
            emoji = "🔴"
        else:
            emoji = "⚪"

        message = (
            f"{emoji} <b>{side} Position Closed</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Reason: <b>{exit_reason}</b>\n"
            f"Entry: <b>${entry_price:.2f}</b>\n"
            f"Exit: <b>${exit_price:.2f}</b>\n"
            f"P&L: <b>${profit:.2f}</b> ({profit_percent:.2f}%)"
        )

        self.send_message(message)

    def notify_error(self, error_message: str):
        """Send notification about an error"""
        if not self.enabled:
            return

        message = f"⚠️ <b>Error</b>\n\n{error_message}"
        self.send_message(message)

    def notify_status(self, status: Dict):
        """Send notification about bot status"""
        if not self.enabled:
            return

        running = status['running']
        symbol = status['symbol']
        interval = status['interval']
        current_price = status['current_price']
        active_orders = status['active_orders']
        open_positions = status['open_positions']
        test_mode = status['test_mode']

        message = (
            f"📊 <b>Bot Status Update</b>\n\n"
            f"Status: <b>{'Running' if running else 'Stopped'}</b>\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Interval: <b>{interval}</b>\n"
            f"Current Price: <b>${current_price:.2f}</b>\n"
            f"Active Orders: <b>{active_orders}</b>\n"
            f"Open Positions: <b>{open_positions}</b>\n"
            f"Mode: <b>{'Test' if test_mode else 'Live'}</b>"
        )

        self.send_message(message)


class LiveTradingBot:
    """
    Live trading bot that implements the futures strategy on Binance.
    Uses WebSockets for real-time data and executes trades automatically.
    """

    def __init__(self,
                 api_key: str,
                 api_secret: str,
                 symbol: str = "BTCUSDT",
                 interval: str = "15m",
                 initial_capital: float = 1000.0,
                 max_risk_per_trade: float = 0.02,
                 leverage: int = 20,
                 window_size: int = 400,
                 min_setup_quality: float = 70.0,
                 min_volume_ratio: float = 3.0,
                 test_mode: bool = True,
                 telegram_config: Optional[Dict] = None):
        """
        Initialize the live trading bot.

        Parameters:
        -----------
        api_key: Binance API key
        api_secret: Binance API secret
        symbol: Trading symbol (e.g., "BTCUSDT")
        interval: Timeframe interval (e.g., "4h", "1d")
        initial_capital: Initial capital for trading
        max_risk_per_trade: Maximum risk per trade as percentage of capital
        leverage: Fixed leverage to use
        window_size: Number of candles to use for analysis
        min_setup_quality: Minimum setup quality to consider
        min_volume_ratio: Minimum volume ratio to consider
        test_mode: If True, run in test mode (no real orders)
        telegram_config: Configuration for Telegram notifications
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.interval = interval
        self.initial_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.leverage = leverage
        self.window_size = window_size
        self.min_setup_quality = min_setup_quality
        self.min_volume_ratio = min_volume_ratio
        self.test_mode = test_mode

        # Initialize Binance client
        self.client = Client(api_key, api_secret)

        # Initialize data fetcher
        self.data_fetcher = BinanceDataFetcher()

        # Initialize strategy
        self.strategy = FuturesStrategy(
            initial_capital=initial_capital,
            max_risk_per_trade=max_risk_per_trade,
            default_leverage=leverage
        )

        # Initialize data storage
        self.historical_data = pd.DataFrame()
        self.current_price = 0.0

        # Initialize order tracking
        self.active_orders = {}
        self.open_positions = {}

        # Initialize control flags
        self.running = False
        self.last_analysis_time = None
        self.analysis_interval_seconds = self._get_interval_seconds(interval)

        # Initialize Telegram notifier
        if telegram_config and telegram_config.get('enabled', False):
            self.telegram = TelegramNotifier(
                bot_token=telegram_config.get('bot_token', ''),
                chat_id=telegram_config.get('chat_id', ''),
                enabled=True
            )
        else:
            self.telegram = TelegramNotifier('', '', enabled=False)

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(
            f"Initialized LiveTradingBot for {symbol} on {interval} timeframe")

    def _signal_handler(self, sig, frame):
        """Handle termination signals for graceful shutdown"""
        logger.info("Received termination signal. Shutting down...")
        self.stop()
        sys.exit(0)

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
            return 3600  # Default to 1 hour

    def _fetch_initial_data(self):
        """Fetch initial historical data"""
        logger.info(
            f"Fetching initial historical data for {self.symbol} {self.interval}")

        try:
            # Calculate start time based on window size
            start_time = datetime.now() - timedelta(
                seconds=self.analysis_interval_seconds * (self.window_size + 10))

            # Fetch data
            self.historical_data = self.data_fetcher.get_historical_klines(
                symbol=self.symbol,
                interval=self.interval,
                start_time=start_time,
                limit=self.window_size + 10
            )

            logger.info(f"Fetched {len(self.historical_data)} initial candles")

            # Set current price
            if not self.historical_data.empty:
                self.current_price = self.historical_data['close'].iloc[-1]

        except Exception as e:
            error_msg = f"Error fetching initial data: {str(e)}"
            logger.error(error_msg)
            self.telegram.notify_error(error_msg)
            raise

    def _process_kline_message(self, msg):
        """Process kline message from WebSocket"""
        try:
            # Kiểm tra xem có phải là thông báo kline không
            if 'k' in msg:
                kline = msg['k']

                # Kiểm tra nến đã đóng chưa
                is_candle_closed = kline['x']

                if not is_candle_closed:
                    # Update current price from the open candle
                    self.current_price = float(kline['c'])
                    return

                # Extract candle data
                timestamp = datetime.fromtimestamp(kline['t'] / 1000)
                open_price = float(kline['o'])
                high_price = float(kline['h'])
                low_price = float(kline['l'])
                close_price = float(kline['c'])
                volume = float(kline['v'])

                # Create new candle data
                new_candle = pd.DataFrame({
                    'open': [open_price],
                    'high': [high_price],
                    'low': [low_price],
                    'close': [close_price],
                    'volume': [volume]
                }, index=[timestamp])

                # Update historical data
                self.historical_data = pd.concat(
                    [self.historical_data, new_candle])

                # Keep only the most recent window_size + 10 candles
                if len(self.historical_data) > self.window_size + 10:
                    self.historical_data = self.historical_data.iloc[-(
                        self.window_size + 10):]

                # Update current price
                self.current_price = close_price

                logger.info(
                    f"New candle closed: {timestamp}, Close: {close_price}")

                # Run analysis on new candle
                self._run_analysis()

        except Exception as e:
            error_msg = f"Error processing kline message: {str(e)}"
            logger.error(error_msg)
            self.telegram.notify_error(error_msg)

    def _run_analysis(self):
        """Run strategy analysis and generate orders"""
        try:
            # Check if we have enough data
            if len(self.historical_data) < self.window_size:
                logger.warning(
                    f"Not enough data for analysis. Have {len(self.historical_data)}, need {self.window_size}")
                return

            # Get analysis window
            analysis_window = self.historical_data.iloc[-self.window_size:].copy()

            # Run market analysis
            analysis = self.strategy.analyze_market(analysis_window)

            # Generate new orders
            new_orders = self.strategy.generate_orders(
                analysis=analysis,
                min_setup_quality=self.min_setup_quality,
                min_volume_ratio=self.min_volume_ratio,
                respect_pressure=True,
                respect_warnings=True
            )

            # Process new orders
            if new_orders:
                logger.info(f"Generated {len(new_orders)} new orders")
                self._process_new_orders(new_orders)
            else:
                logger.info("No new orders generated")

            # Update last analysis time
            self.last_analysis_time = datetime.now()

        except Exception as e:
            error_msg = f"Error running analysis: {str(e)}"
            logger.error(error_msg)
            self.telegram.notify_error(error_msg)

    def _process_new_orders(self, new_orders: List[Dict]):
        """Process new orders and place them on the exchange"""
        for order in new_orders:
            try:
                # Add timestamp
                order['created_time'] = datetime.now()
                order['symbol'] = self.symbol

                # Calculate price to entry percentage
                order['price_to_entry_percent'] = (
                    (order['entry_price'] / self.current_price) - 1) * 100

                # Log order details
                logger.info(f"New {order['side']} order: {order['setup_type']}, "
                            f"Entry: {order['entry_price']}, Stop: {order['stop_loss']}, "
                            f"Quality: {order['setup_quality']}, Volume: {order['volume_ratio']}")

                # Send Telegram notification
                self.telegram.notify_order_created(order)

                # Place order on exchange
                if not self.test_mode:
                    self._place_order_on_exchange(order)
                else:
                    logger.info(
                        f"TEST MODE: Would place {order['side']} order at {order['entry_price']}")

                    # In test mode, simulate order placement
                    order_id = f"test_{int(time.time())}_{len(self.active_orders)}"
                    order['order_id'] = order_id
                    self.active_orders[order_id] = order

            except Exception as e:
                error_msg = f"Error processing order: {str(e)}"
                logger.error(error_msg)
                self.telegram.notify_error(error_msg)

    def _place_order_on_exchange(self, order: Dict):
        """Place an order on Binance"""
        try:
            # Set leverage first
            self.client.futures_change_leverage(
                symbol=self.symbol,
                leverage=order['leverage']
            )

            # Calculate quantity
            quantity = order['position_size'] / order['entry_price']
            quantity = self._round_step_size(quantity)

            # Determine order type and parameters
            if order['entry_type'] == 'MARKET':
                # Place market order
                response = self.client.futures_create_order(
                    symbol=self.symbol,
                    side='BUY' if order['side'] == 'LONG' else 'SELL',
                    type='MARKET',
                    quantity=quantity
                )

                # Get filled price
                order['actual_entry_price'] = float(response['avgPrice'])
                order['order_id'] = response['orderId']
                order['status'] = 'ACTIVE'

                # Send notification
                self.telegram.notify_order_filled(order)

                # Place stop loss
                self._place_stop_loss(order)

                # Place take profits
                self._place_take_profits(order)

                # Add to open positions
                self.open_positions[response['orderId']] = order

                logger.info(f"Market order placed: {response['orderId']}")

            else:  # LIMIT order
                # Place limit order
                response = self.client.futures_create_order(
                    symbol=self.symbol,
                    side='BUY' if order['side'] == 'LONG' else 'SELL',
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity=quantity,
                    price=order['entry_price']
                )

                order['order_id'] = response['orderId']
                order['status'] = 'PENDING'

                # Add to active orders
                self.active_orders[response['orderId']] = order

                logger.info(f"Limit order placed: {response['orderId']}")

        except BinanceAPIException as e:
            error_msg = f"Binance API error placing order: {str(e)}"
            logger.error(error_msg)
            self.telegram.notify_error(error_msg)
        except Exception as e:
            error_msg = f"Error placing order on exchange: {str(e)}"
            logger.error(error_msg)
            self.telegram.notify_error(error_msg)

    def _place_stop_loss(self, order: Dict):
        """Place stop loss order"""
        try:
            # Calculate quantity
            quantity = order['position_size'] / order['actual_entry_price']
            quantity = self._round_step_size(quantity)

            # Place stop loss order
            response = self.client.futures_create_order(
                symbol=self.symbol,
                side='SELL' if order['side'] == 'LONG' else 'BUY',
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=order['stop_loss'],
                closePosition=True
            )

            order['stop_loss_order_id'] = response['orderId']
            logger.info(f"Stop loss placed at {order['stop_loss']}")

        except Exception as e:
            error_msg = f"Error placing stop loss: {str(e)}"
            logger.error(error_msg)
            self.telegram.notify_error(error_msg)

    def _place_take_profits(self, order: Dict):
        """Place take profit orders"""
        try:
            # Calculate quantity
            quantity = order['position_size'] / order['actual_entry_price']
            quantity = self._round_step_size(quantity)

            # Place take profit orders
            tp_order_ids = {}

            # Split quantity among take profit levels
            tp_levels = len(order['take_profit'])
            qty_per_level = quantity / tp_levels
            qty_per_level = self._round_step_size(qty_per_level)

            for tp_name, tp_price in order['take_profit'].items():
                response = self.client.futures_create_order(
                    symbol=self.symbol,
                    side='SELL' if order['side'] == 'LONG' else 'BUY',
                    type='TAKE_PROFIT_MARKET',
                    quantity=qty_per_level,
                    stopPrice=tp_price
                )

                tp_order_ids[tp_name] = response['orderId']
                logger.info(f"Take profit {tp_name} placed at {tp_price}")

            order['take_profit_order_ids'] = tp_order_ids

        except Exception as e:
            error_msg = f"Error placing take profits: {str(e)}"
            logger.error(error_msg)
            self.telegram.notify_error(error_msg)

    def _round_step_size(self, quantity: float) -> float:
        """Round quantity to valid step size"""
        try:
            # Get symbol info
            info = self.client.get_symbol_info(self.symbol)

            # Find the quantity filter
            filters = info['filters']
            step_size = None

            for f in filters:
                if f['filterType'] == 'LOT_SIZE':
                    step_size = float(f['stepSize'])
                    break

            if step_size:
                # Calculate precision
                precision = 0
                if step_size < 1:
                    precision = len(str(step_size).split('.')[-1].rstrip('0'))

                # Round to precision
                return round(quantity - (quantity % step_size), precision)
            else:
                # Default to 5 decimals if no step size found
                return round(quantity, 5)

        except Exception as e:
            logger.error(f"Error rounding quantity: {str(e)}")
            # Default to 5 decimals
            return round(quantity, 5)

    def _check_order_status(self):
        """Check status of active orders and update accordingly"""
        if self.test_mode:
            # In test mode, simulate order execution
            self._simulate_order_execution()
            return

        try:
            # Get all open orders
            open_orders = self.client.futures_get_open_orders(
                symbol=self.symbol)
            open_order_ids = [str(order['orderId']) for order in open_orders]

            # Check for filled orders
            for order_id in list(self.active_orders.keys()):
                if order_id not in open_order_ids:
                    # Order is no longer open, check if it was filled
                    order = self.active_orders[order_id]

                    # Get order status
                    order_info = self.client.futures_get_order(
                        symbol=self.symbol,
                        orderId=order_id
                    )

                    if order_info['status'] == 'FILLED':
                        # Order was filled
                        logger.info(f"Order {order_id} was filled")

                        # Update order details
                        order['status'] = 'ACTIVE'
                        order['actual_entry_price'] = float(
                            order_info['avgPrice'])
                        order['filled_time'] = datetime.fromtimestamp(
                            order_info['updateTime'] / 1000)

                        # Send notification
                        self.telegram.notify_order_filled(order)

                        # Place stop loss
                        self._place_stop_loss(order)

                        # Place take profits
                        self._place_take_profits(order)

                        # Move to open positions
                        self.open_positions[order_id] = order
                        del self.active_orders[order_id]

                    elif order_info['status'] in ['CANCELED', 'EXPIRED', 'REJECTED']:
                        # Order was canceled or expired
                        logger.info(
                            f"Order {order_id} was {order_info['status'].lower()}")
                        del self.active_orders[order_id]

            # Check for closed positions
            positions = self.client.futures_position_information(
                symbol=self.symbol)

            for position in positions:
                position_amount = float(position['positionAmt'])

                # If position amount is 0, the position is closed
                if position_amount == 0:
                    for position_id in list(self.open_positions.keys()):
                        # Check if this position was closed
                        # We need to check recent trades to get the exit price and reason
                        trades = self.client.futures_account_trades(
                            symbol=self.symbol, limit=10)

                        for trade in trades:
                            if trade['orderId'] == position_id:
                                position = self.open_positions[position_id]

                                # Update position details
                                position['status'] = 'CLOSED'
                                position['exit_time'] = datetime.fromtimestamp(
                                    trade['time'] / 1000)
                                position['exit_price'] = float(trade['price'])

                                # Determine exit reason
                                if trade['orderId'] == position.get('stop_loss_order_id'):
                                    position['exit_reason'] = 'STOP_LOSS'
                                else:
                                    # Check if it matches any take profit order
                                    for tp_name, tp_id in position.get('take_profit_order_ids', {}).items():
                                        if trade['orderId'] == tp_id:
                                            position['exit_reason'] = f'TAKE_PROFIT_{tp_name.upper()}'
                                            break
                                    else:
                                        position['exit_reason'] = 'MANUAL_CLOSE'

                                # Calculate profit
                                entry_price = position['actual_entry_price']
                                exit_price = position['exit_price']
                                position_size = position['position_size']

                                if position['side'] == 'LONG':
                                    profit = position_size * \
                                        (exit_price - entry_price) / entry_price
                                else:  # SHORT
                                    profit = position_size * \
                                        (entry_price - exit_price) / entry_price

                                position['profit'] = profit
                                position['profit_percent'] = (
                                    profit / position.get('margin_amount', 1)) * 100

                                # Send notification
                                self.telegram.notify_position_closed(position)

                                # Remove from open positions
                                del self.open_positions[position_id]
                                break

        except Exception as e:
            error_msg = f"Error checking order status: {str(e)}"
            logger.error(error_msg)
            self.telegram.notify_error(error_msg)

    def _simulate_order_execution(self):
        """Simulate order execution in test mode"""
        # Process active orders
        for order_id in list(self.active_orders.keys()):
            order = self.active_orders[order_id]

            # Check if limit order should be filled
            if order['entry_type'] == 'LIMIT':
                if (order['side'] == 'LONG' and self.current_price <= order['entry_price']) or \
                   (order['side'] == 'SHORT' and self.current_price >= order['entry_price']):
                    # Simulate order fill
                    order['status'] = 'ACTIVE'
                    order['actual_entry_price'] = order['entry_price']
                    order['filled_time'] = datetime.now()

                    # Send notification
                    self.telegram.notify_order_filled(order)

                    # Move to open positions
                    self.open_positions[order_id] = order
                    del self.active_orders[order_id]

                    logger.info(
                        f"TEST MODE: Order {order_id} filled at {order['actual_entry_price']}")

        # Process open positions
        for position_id in list(self.open_positions.keys()):
            position = self.open_positions[position_id]

            # Check if stop loss hit
            if (position['side'] == 'LONG' and self.current_price <= position['stop_loss']) or \
               (position['side'] == 'SHORT' and self.current_price >= position['stop_loss']):
                # Simulate stop loss
                position['status'] = 'CLOSED'
                position['exit_time'] = datetime.now()
                position['exit_price'] = position['stop_loss']
                position['exit_reason'] = 'STOP_LOSS'

                # Calculate profit/loss
                if position['side'] == 'LONG':
                    profit = position['position_size'] * (
                        position['stop_loss'] - position['actual_entry_price']) / position['actual_entry_price']
                else:
                    profit = position['position_size'] * (
                        position['actual_entry_price'] - position['stop_loss']) / position['actual_entry_price']

                position['profit'] = profit
                position['profit_percent'] = (
                    profit / position.get('margin_amount', 1)) * 100

                # Send notification
                self.telegram.notify_position_closed(position)

                logger.info(
                    f"TEST MODE: Position {position_id} stopped out at {position['stop_loss']} with PNL: {profit:.2f}")

                # Remove from open positions
                del self.open_positions[position_id]
                continue

            # Check if take profits hit
            for tp_name, tp_price in position['take_profit'].items():
                if (position['side'] == 'LONG' and self.current_price >= tp_price) or \
                   (position['side'] == 'SHORT' and self.current_price <= tp_price):
                    # Simulate take profit
                    position['status'] = 'CLOSED'
                    position['exit_time'] = datetime.now()
                    position['exit_price'] = tp_price
                    position['exit_reason'] = f'TAKE_PROFIT_{tp_name.upper()}'

                    # Calculate profit
                    if position['side'] == 'LONG':
                        profit = position['position_size'] * (
                            tp_price - position['actual_entry_price']) / position['actual_entry_price']
                    else:
                        profit = position['position_size'] * (
                            position['actual_entry_price'] - tp_price) / position['actual_entry_price']

                    position['profit'] = profit
                    position['profit_percent'] = (
                        profit / position.get('margin_amount', 1)) * 100

                    # Send notification
                    self.telegram.notify_position_closed(position)

                    logger.info(
                        f"TEST MODE: Position {position_id} took profit at {tp_price} with PNL: {profit:.2f}")

                    # Remove from open positions
                    del self.open_positions[position_id]
                    break

    async def _start_socket(self):
        """Khởi động WebSocket connection với asyncio"""
        try:
            # Khởi tạo AsyncClient
            client = await AsyncClient.create(self.api_key, self.api_secret)

            # Khởi tạo BinanceSocketManager
            bsm = BinanceSocketManager(client)

            # Bắt đầu kline socket
            kline_socket = bsm.kline_socket(
                symbol=self.symbol, interval=self.interval)

            # Xử lý dữ liệu từ socket
            async with kline_socket as stream:
                while self.running:
                    msg = await stream.recv()
                    self._process_kline_message(msg)

            # Đóng client khi kết thúc
            await client.close_connection()

        except Exception as e:
            error_msg = f"Error in WebSocket connection: {str(e)}"
            logger.error(error_msg)
            self.telegram.notify_error(error_msg)

    def start(self):
        """Start the trading bot"""
        if self.running:
            logger.warning("Trading bot is already running")
            return

        try:
            logger.info("Starting trading bot...")
            self.running = True

            # Fetch initial data
            self._fetch_initial_data()

            # Khởi động WebSocket trong một thread riêng
            self.socket_thread = threading.Thread(
                target=lambda: asyncio.run(self._start_socket()))
            self.socket_thread.daemon = True
            self.socket_thread.start()

            logger.info(f"Started WebSocket connection for {self.symbol}")

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
                f"Trading bot started for {self.symbol} on {self.interval} timeframe")

            # Send notification
            self.telegram.send_message(
                f"🚀 <b>Trading Bot Started</b>\n\n"
                f"Symbol: <b>{self.symbol}</b>\n"
                f"Timeframe: <b>{self.interval}</b>\n"
                f"Current Price: <b>${self.current_price:.2f}</b>\n"
                f"Mode: <b>{'Test' if self.test_mode else 'Live'}</b>"
            )

            # Run initial analysis
            self._run_analysis()

        except Exception as e:
            error_msg = f"Error starting trading bot: {str(e)}"
            logger.error(error_msg)
            self.telegram.notify_error(error_msg)
            self.stop()

    def _status_check_loop(self):
        """Continuously check order status"""
        while self.running:
            try:
                self._check_order_status()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                error_msg = f"Error in status check loop: {str(e)}"
                logger.error(error_msg)
                self.telegram.notify_error(error_msg)
                time.sleep(30)  # Wait longer on error

    def _analysis_loop(self):
        """Continuously run market analysis at interval boundaries"""
        while self.running:
            try:
                # Calculate time until next candle closes
                now = datetime.now()
                seconds_in_interval = self._get_interval_seconds(self.interval)

                # Calculate time until next candle
                if self.interval.endswith('m'):
                    # For minute-based intervals
                    minutes = int(self.interval[:-1])
                    current_minute = now.minute
                    minutes_to_next = minutes - (current_minute % minutes)
                    if minutes_to_next == 0:
                        minutes_to_next = minutes
                    seconds_to_next = minutes_to_next * 60 - now.second
                elif self.interval.endswith('h'):
                    # For hour-based intervals
                    hours = int(self.interval[:-1])
                    current_hour = now.hour
                    hours_to_next = hours - (current_hour % hours)
                    if hours_to_next == 0:
                        hours_to_next = hours
                    seconds_to_next = hours_to_next * \
                        3600 - (now.minute * 60 + now.second)
                else:
                    # Default to 1 minute if interval format is unknown
                    seconds_to_next = 60 - now.second

                # Add a small buffer to ensure the candle has closed
                seconds_to_next += 2

                # Sleep until next candle
                logger.info(f"Next analysis in {seconds_to_next} seconds")
                time.sleep(seconds_to_next)

                # Run analysis if we're still running
                if self.running:
                    self._run_analysis()

            except Exception as e:
                error_msg = f"Error in analysis loop: {str(e)}"
                logger.error(error_msg)
                self.telegram.notify_error(error_msg)
                time.sleep(60)  # Wait a minute before trying again

    def stop(self):
        """Stop the trading bot"""
        if not self.running:
            return

        logger.info("Stopping trading bot...")
        self.running = False

        # WebSocket sẽ tự đóng khi self.running = False

        # Send notification
        self.telegram.send_message("🛑 <b>Trading Bot Stopped</b>")

        logger.info("Trading bot stopped")

    def get_status(self) -> Dict:
        """Get current status of the trading bot"""
        status = {
            'running': self.running,
            'symbol': self.symbol,
            'interval': self.interval,
            'current_price': self.current_price,
            'active_orders': len(self.active_orders),
            'open_positions': len(self.open_positions),
            'last_analysis': self.last_analysis_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_analysis_time else None,
            'test_mode': self.test_mode,
            'capital': self.strategy.capital
        }

        # Send status notification periodically
        if hasattr(self, 'telegram'):
            self.telegram.notify_status(status)

        return status

    def get_open_positions(self) -> List[Dict]:
        """Get list of open positions"""
        return list(self.open_positions.values())

    def get_active_orders(self) -> List[Dict]:
        """Get list of active orders"""
        return list(self.active_orders.values())


if __name__ == "__main__":
    load_dotenv()
    # Load configuration from environment variables first, then fallback to config file
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')
    print(api_key, api_secret)
    symbol = os.environ.get('TRADING_SYMBOL', 'BTCUSDT')
    interval = os.environ.get('TRADING_INTERVAL', '15m')
    initial_capital = float(os.environ.get('INITIAL_CAPITAL', '1000.0'))
    max_risk_per_trade = float(os.environ.get('MAX_RISK_PER_TRADE', '0.02'))
    leverage = int(os.environ.get('LEVERAGE', '20'))
    window_size = int(os.environ.get('WINDOW_SIZE', '100'))
    min_setup_quality = float(os.environ.get('MIN_SETUP_QUALITY', '70.0'))
    min_volume_ratio = float(os.environ.get('MIN_VOLUME_RATIO', '3.0'))
    test_mode = os.environ.get('TEST_MODE', 'true').lower() == 'true'

    # Telegram settings
    telegram_enabled = os.environ.get(
        'TELEGRAM_ENABLED', 'false').lower() == 'true'
    telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')

    # Validate required configuration
    # if not api_key or not api_secret:
    #     logger.error(
    #         "API key and secret are required. Set them in environment variables or config file.")
    #     sys.exit(1)

    # Create and start trading bot
    bot = LiveTradingBot(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        interval=interval,
        initial_capital=initial_capital,
        max_risk_per_trade=max_risk_per_trade,
        leverage=leverage,
        window_size=window_size,
        min_setup_quality=min_setup_quality,
        min_volume_ratio=min_volume_ratio,
        test_mode=test_mode
    )

    # Initialize Telegram notifier
    if telegram_enabled and telegram_token and telegram_chat_id:
        bot.telegram = TelegramNotifier(
            bot_token=telegram_token,
            chat_id=telegram_chat_id,
            enabled=True
        )
    else:
        # Create a dummy notifier that does nothing
        bot.telegram = TelegramNotifier(
            bot_token="",
            chat_id="",
            enabled=False
        )

    # Start the bot
    bot.start()

    # Keep the main thread running
    try:
        while True:
            time.sleep(60 * 15)  # Check status every 15 minutes
            status = bot.get_status()
            logger.info(f"Bot status: {status}")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
        bot.stop()
