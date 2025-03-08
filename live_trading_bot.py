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

from strategy import calculate_rsi

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
    Handles sending notifications to Telegram with support for topics.
    """

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True, orders_topic_id: str = None, signals_topic_id: str = None):
        """
        Initialize the Telegram notifier with topic support.

        Parameters:
        -----------
        bot_token: Telegram bot token
        chat_id: Telegram chat ID to send messages to
        enabled: Whether notifications are enabled
        orders_topic_id: Topic ID for orders notifications
        signals_topic_id: Topic ID for signals notifications
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.orders_topic_id = orders_topic_id
        self.signals_topic_id = signals_topic_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        if enabled:
            self.send_message("🤖 Trading Bot initialized and ready.")
            logger.info("Telegram notifications enabled")
        else:
            logger.info("Telegram notifications disabled")

    def send_message(self, message: str, parse_mode: str = "HTML", topic_id: str = None):
        """Send a message to the Telegram chat with optional topic"""
        if not self.enabled:
            return

        try:
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }

            # Add message_thread_id for topic if provided
            if topic_id:
                data["message_thread_id"] = topic_id

            response = requests.post(self.base_url, data=data)

            if response.status_code != 200:
                logger.error(
                    f"Failed to send Telegram message: {response.text}")

        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")

    def notify_signal(self, signal_data: Dict):
        """Send notification about a new trading signal to signals topic"""
        if not self.enabled:
            return

        symbol = signal_data['symbol']
        signal_type = signal_data['signal_type']
        price = signal_data['price']
        ema_fast = signal_data.get('ema_fast', 0)
        ema_slow = signal_data.get('ema_slow', 0)
        volume_ratio = signal_data.get('volume_ratio', 0)
        rsi = signal_data.get('rsi', 0)

        # Determine emoji based on signal type
        if signal_type == "BUY":
            emoji = "🟢"
        elif signal_type == "SELL":
            emoji = "🔴"
        else:
            emoji = "⚠️"

        message = (
            f"{emoji} <b>{signal_type} Signal Detected</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Price: <b>${price:.2f}</b>\n"
            f"EMA Fast: <b>{ema_fast:.2f}</b>\n"
            f"EMA Slow: <b>{ema_slow:.2f}</b>\n"
            f"RSI: <b>{rsi:.2f}</b>\n"
            f"Volume Ratio: <b>{volume_ratio:.2f}x</b>"
        )

        if 'support' in signal_data:
            message += f"\nSupport: <b>${signal_data['support']:.2f}</b>"

        if 'resistance' in signal_data:
            message += f"\nResistance: <b>${signal_data['resistance']:.2f}</b>"

        # Send to signals topic if configured, otherwise to main chat
        self.send_message(message, topic_id=self.signals_topic_id)

    def notify_order_created(self, order: Dict):
        """Send notification about a new order to orders topic"""
        if not self.enabled:
            return

        side = order['side']
        symbol = order['symbol']
        entry_type = order.get('entry_type', '')
        entry_price = order['entry_price']
        stop_loss = order['stop_loss']
        setup_quality = order.get('setup_quality', 0)
        setup_type = order.get('setup_type', '')
        volume_ratio = order.get('volume_ratio', 'N/A')
        position_size = order.get('position_size', 0)
        leverage = order.get('leverage', 20)
        margin = order.get('margin_amount', 0)

        # Calculate risk-reward for TP2
        risk = abs(entry_price - stop_loss)
        tp2 = order['take_profit'].get('tp2', 0) if isinstance(
            order.get('take_profit', {}), dict) else 0
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

        # Send to orders topic if configured, otherwise to main chat
        self.send_message(message, topic_id=self.orders_topic_id)

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
                 telegram: TelegramNotifier = None,
                 # EMA Crossover params
                 fast_ema: int = 8,
                 slow_ema: int = 21,
                 volume_threshold: float = 2.0):
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
        fast_ema: Fast EMA period for crossover strategy
        slow_ema: Slow EMA period for crossover strategy
        volume_threshold: Volume threshold for crossover strategy
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

        # EMA Crossover parameters
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.volume_threshold = volume_threshold
        self.last_signal = None
        self.last_analysis_time = None
        self.in_position = False

        # Support & resistance levels
        self.support_levels = []
        self.resistance_levels = []

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
        self.analysis_interval_seconds = self._get_interval_seconds(interval)

        # Initialize Telegram notifier
        self.telegram = telegram

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
                limit=self.window_size
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

                # Calculate indicators for EMA crossover strategy
                self.historical_data = self._calculate_indicators(
                    self.historical_data)

                # Run both analyses on new candle
                self._run_analysis()  # Futures strategy analysis
                self._generate_signals()  # EMA crossover analysis

        except Exception as e:
            error_msg = f"Error processing kline message: {str(e)}"
            logger.error(error_msg)
            self.telegram.notify_error(error_msg)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for EMA crossover strategy and market analysis"""
        try:
            df = df.copy()

            # Calculate EMA values for crossover
            df['ema_fast'] = df['close'].ewm(
                span=self.fast_ema, adjust=False).mean()
            df['ema_slow'] = df['close'].ewm(
                span=self.slow_ema, adjust=False).mean()

            # Calculate basic MA for trend determination
            df['ma'] = df['close'].rolling(window=50).mean()

            # Calculate RSI
            df['rsi'] = calculate_rsi(df)

            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            # Avoid division by zero
            rs = avg_gain / avg_loss.replace(0, 0.001)
            df['rsi'] = 100 - (100 / (1 + rs))

            # Calculate ATR for volatility
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()

            # Calculate volume ratio
            df['avg_volume'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['avg_volume']

            # Calculate crossover signals
            df['ema_diff'] = df['ema_fast'] - df['ema_slow']
            df['ema_diff_prev'] = df['ema_diff'].shift(1)

            # Create crossover column (1 for bullish, -1 for bearish, 0 for no crossover)
            df['crossover'] = 0
            df.loc[(df['ema_diff'] > 0) & (df['ema_diff_prev'] <= 0),
                   'crossover'] = 1  # Bullish crossover
            df.loc[(df['ema_diff'] < 0) & (df['ema_diff_prev'] >= 0),
                   'crossover'] = -1  # Bearish crossover

            # Calculate support/resistance levels
            self._identify_support_resistance(df)

            return df

        except Exception as e:
            error_msg = f"Error calculating indicators: {str(e)}"
            logger.error(error_msg)
            if hasattr(self, 'telegram'):
                self.telegram.notify_error(error_msg)
            return df

    def _identify_support_resistance(self, df: pd.DataFrame, num_levels: int = 5, window: int = 10):
        """Identify important support and resistance levels"""
        try:
            # Create empty lists for support and resistance
            support_levels = []
            resistance_levels = []

            # Find local minima (support) and maxima (resistance)
            for i in range(window, len(df) - window):
                # Check if this is a local low (support)
                if all(df.iloc[i-window:i]['low'] >= df.iloc[i]['low']) and all(df.iloc[i+1:i+window+1]['low'] >= df.iloc[i]['low']):
                    support_levels.append(df.iloc[i]['low'])

                # Check if this is a local high (resistance)
                if all(df.iloc[i-window:i]['high'] <= df.iloc[i]['high']) and all(df.iloc[i+1:i+window+1]['high'] <= df.iloc[i]['high']):
                    resistance_levels.append(df.iloc[i]['high'])

            # Remove duplicates and sort
            support_levels = sorted(
                list(set([round(level, 2) for level in support_levels])))
            resistance_levels = sorted(
                list(set([round(level, 2) for level in resistance_levels])))

            # Keep only the strongest levels (most recent and significant)
            current_price = df.iloc[-1]['close']

            # Filter levels close to current price
            support_levels = [
                level for level in support_levels if level < current_price]
            resistance_levels = [
                level for level in resistance_levels if level > current_price]

            # Sort by distance to current price and take top levels
            support_levels = sorted(
                support_levels, key=lambda x: current_price - x)[:num_levels]
            resistance_levels = sorted(
                resistance_levels, key=lambda x: x - current_price)[:num_levels]

            # Store the levels
            self.support_levels = support_levels
            self.resistance_levels = resistance_levels

        except Exception as e:
            logger.error(f"Error identifying support/resistance: {str(e)}")

    def _is_near_support(self, price: float) -> bool:
        """Check if price is near a support level"""
        if not self.support_levels:
            return False

        threshold = price * 0.01  # 1% of current price
        return any(abs(price - level) < threshold for level in self.support_levels)

    def _is_near_resistance(self, price: float) -> bool:
        """Check if price is near a resistance level"""
        if not self.resistance_levels:
            return False

        threshold = price * 0.01  # 1% of current price
        return any(abs(price - level) < threshold for level in self.resistance_levels)

    def _get_closest_support(self, price: float) -> Optional[float]:
        """Get the closest support level below current price"""
        if not self.support_levels:
            return None

        supports_below = [
            level for level in self.support_levels if level < price]
        if supports_below:
            return max(supports_below)  # Highest support below price
        return None

    def _get_closest_resistance(self, price: float) -> Optional[float]:
        """Get the closest resistance level above current price"""
        if not self.resistance_levels:
            return None

        resistances_above = [
            level for level in self.resistance_levels if level > price]
        if resistances_above:
            return min(resistances_above)  # Lowest resistance above price
        return None

    def _generate_signals(self):
        """Generate trading signals based on EMA crossover and other indicators"""
        try:
            # Check if we have enough data
            if len(self.historical_data) < 50:  # Need at least 50 candles for reliable indicators
                logger.warning("Not enough data for signal generation")
                return

            # Get the latest data point
            latest = self.historical_data.iloc[-1]

            # Check if we have all necessary indicators
            if any(pd.isna([latest['ema_fast'], latest['ema_slow'], latest['ma'], latest['volume_ratio']])):
                logger.warning(
                    "Missing indicator data, skipping signal generation")
                return

            # Extract current values
            current_price = latest['close']
            ema_fast = latest['ema_fast']
            ema_slow = latest['ema_slow']
            ma = latest['ma']
            volume_ratio = latest['volume_ratio']
            crossover = latest['crossover']
            rsi = latest['rsi']

            # Signal variables
            signal = None
            signal_strength = 0

            # Check for buy signal
            if crossover == 1:  # Bullish crossover
                # Check volume confirmation
                if volume_ratio >= self.volume_threshold:
                    # Check if price is above MA (uptrend)
                    if current_price > ma:
                        # Check proximity to support
                        near_support = self._is_near_support(current_price)

                        if near_support:
                            signal = "BUY"
                            # Strong signal (all conditions met)
                            signal_strength = 3
                        else:
                            signal = "BUY"
                            # Medium signal (not near support)
                            signal_strength = 2
                    else:
                        signal = "BUY"
                        signal_strength = 1  # Weak signal (not in uptrend)

            # Check for sell signal
            elif crossover == -1:  # Bearish crossover
                # Check volume confirmation
                if volume_ratio >= self.volume_threshold:
                    # Check if price is below MA (downtrend)
                    if current_price < ma:
                        # Check proximity to resistance
                        near_resistance = self._is_near_resistance(
                            current_price)

                        if near_resistance:
                            signal = "SELL"
                            # Strong signal (all conditions met)
                            signal_strength = 3
                        else:
                            signal = "SELL"
                            # Medium signal (not near resistance)
                            signal_strength = 2
                    else:
                        signal = "SELL"
                        signal_strength = 1  # Weak signal (not in downtrend)

            # Process signal if we have one
            if signal and signal_strength >= 2:  # Only act on medium or strong signals
                # Only notify if it's a new signal or we haven't seen one in a while
                if self.last_signal != signal or (datetime.now() - self.last_analysis_time > timedelta(hours=2)):
                    # Get closest support/resistance level for stop loss/take profit
                    closest_support = self._get_closest_support(current_price)
                    closest_resistance = self._get_closest_resistance(
                        current_price)

                    # Create signal data
                    signal_data = {
                        'symbol': self.symbol,
                        'signal_type': signal,
                        'price': current_price,
                        'ema_fast': ema_fast,
                        'ema_slow': ema_slow,
                        'volume_ratio': volume_ratio,
                        'signal_strength': signal_strength,
                        'timestamp': datetime.now(),
                        'rsi': rsi
                    }

                    # Add support/resistance if available
                    if closest_support:
                        signal_data['support'] = closest_support

                    if closest_resistance:
                        signal_data['resistance'] = closest_resistance

                    # Update last signal
                    self.last_signal = signal

                    # Send notification
                    logger.info(
                        f"Generated {signal} signal with strength {signal_strength}")
                    self.telegram.notify_signal(signal_data)

                    # Only place order based on signals if not in test mode and not already in position
                    if not self.test_mode and not self.in_position:
                        self._place_signal_order(signal_data)

        except Exception as e:
            error_msg = f"Error generating signals: {str(e)}"
            logger.error(error_msg)
            self.telegram.notify_error(error_msg)

    def _place_signal_order(self, signal_data: Dict):
        """Place an order based on EMA crossover signal"""
        try:
            # Extract data
            signal_type = signal_data['signal_type']
            price = signal_data['price']

            # Determine side
            side = "LONG" if signal_type == "BUY" else "SHORT"

            # Get latest data point
            latest = self.historical_data.iloc[-1]

            # Get ATR for volatility-based stop loss
            atr = latest['atr']

            # Calculate volatility as percentage
            volatility = (atr / price) * 100

            # Determine ATR multiplier based on volatility
            if volatility < 1:
                atr_multiplier = 3.0  # Lower volatility needs wider stops
            elif volatility < 2:
                atr_multiplier = 2.5
            elif volatility < 4:
                atr_multiplier = 2.0
            else:
                atr_multiplier = 1.5  # High volatility needs tighter stops

            # Calculate stop loss and take profit based on side
            if side == "LONG":
                # For longs: price - (ATR * multiplier)
                # Adjust based on nearest support level
                base_stop = price - (atr * atr_multiplier)

                # Check if there's a support level between price and base_stop
                close_supports = [
                    level for level in self.support_levels if level < price and level > base_stop]

                if close_supports:
                    # Use the highest support level that's below price but above base_stop
                    # Small buffer below support
                    stop_loss = max(close_supports) - (atr * 0.5)
                else:
                    stop_loss = base_stop

                # Take profit based on resistance or risk:reward ratio
                resistance = next(
                    (level for level in self.resistance_levels if level > price), None)
                risk = price - stop_loss

                if resistance:
                    # Target the nearest resistance
                    take_profit = resistance
                else:
                    # Default to 2:1 risk:reward if no resistance found
                    take_profit = price + (risk * 2)

            else:  # SELL/SHORT
                # For shorts: price + (ATR * multiplier)
                # Adjust based on nearest resistance level
                base_stop = price + (atr * atr_multiplier)

                # Check if there's a resistance level between price and base_stop
                close_resistances = [
                    level for level in self.resistance_levels if level > price and level < base_stop]

                if close_resistances:
                    # Use the lowest resistance level that's above price but below base_stop
                    # Small buffer above resistance
                    stop_loss = min(close_resistances) + (atr * 0.5)
                else:
                    stop_loss = base_stop

                # Take profit based on support or risk:reward ratio
                support = next(
                    (level for level in self.support_levels if level < price), None)
                risk = stop_loss - price

                if support:
                    # Target the nearest support
                    take_profit = support
                else:
                    # Default to 2:1 risk:reward if no support found
                    take_profit = price - (risk * 2)

            # Calculate risk-reward ratio
            risk_reward = abs(take_profit - price) / abs(price - stop_loss)

            # Only take trades with acceptable risk:reward
            if risk_reward < 1.5:
                logger.info(
                    f"Skipping {side} signal - insufficient risk:reward ratio ({risk_reward:.2f})")
                return None

            # Calculate position size based on risk
            risk_amount = self.strategy.capital * self.max_risk_per_trade
            price_risk = abs(price - stop_loss)
            position_size = (risk_amount / price_risk) * self.leverage

            # Create order object
            order = {
                'symbol': self.symbol,
                'side': side,
                'entry_type': 'MARKET',
                'entry_price': price,
                'stop_loss': stop_loss,
                'take_profit': {'tp1': take_profit},
                'risk_reward': risk_reward,
                'position_size': position_size,
                'leverage': self.leverage,
                'setup_type': f"EMA_CROSSOVER_{side}",
                # Convert to percentage
                'setup_quality': signal_data['signal_strength'] * 25,
                'volume_ratio': signal_data['volume_ratio'],
                'margin_amount': position_size / self.leverage
            }

            # Process the order
            self._process_new_orders([order])

            return order

        except Exception as e:
            error_msg = f"Error placing signal order: {str(e)}"
            logger.error(error_msg)
            self.telegram.notify_error(error_msg)
            return None

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

            # Track orders from the current analysis
            new_order_signatures = set()
            if new_orders:
                # Create signatures for new orders for later comparison
                for order in new_orders:
                    # Create a unique signature based on key order properties
                    signature = f"{order['side']}_{order['setup_type']}_{order['entry_price']:.2f}_{order['stop_loss']:.2f}"
                    new_order_signatures.add(signature)

                logger.info(f"Generated {len(new_orders)} new orders")
                self._process_new_orders(new_orders)
            else:
                logger.info("No new orders generated")

            # Check for obsolete orders that are no longer recommended
            if not self.test_mode:
                self._cancel_obsolete_orders(new_order_signatures)
            else:
                self._simulate_cancel_obsolete_orders(new_order_signatures)

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

            # Set margin type to ISOLATED
            self.client.futures_change_margin_type(
                symbol=self.symbol,
                marginType='ISOLATED'
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

    def _cancel_obsolete_orders(self, new_order_signatures: set):
        """Cancel orders that are no longer recommended by the latest analysis"""
        try:
            # Check each active order
            for order_id in list(self.active_orders.keys()):
                order = self.active_orders[order_id]

                # Skip orders that are not PENDING (i.e., already FILLED or processing)
                if order['status'] != 'PENDING':
                    continue

                # Create signature for this order
                order_signature = f"{order['side']}_{order['setup_type']}_{order['entry_price']:.2f}_{order['stop_loss']:.2f}"

                # If this order signature is not in the new recommendations, cancel it
                if order_signature not in new_order_signatures:
                    logger.info(
                        f"Canceling obsolete order {order_id} that is no longer recommended")

                    # Cancel the order on the exchange
                    self.client.futures_cancel_order(
                        symbol=self.symbol,
                        orderId=order_id
                    )

                    # Remove from our active orders
                    del self.active_orders[order_id]

                    # Send notification
                    self.telegram.send_message(
                        f"🚫 <b>Order Canceled</b>\n\n"
                        f"Symbol: <b>{self.symbol}</b>\n"
                        f"Side: <b>{order['side']}</b>\n"
                        f"Setup: <b>{order['setup_type']}</b>\n"
                        f"Entry: <b>${order['entry_price']:.2f}</b>\n"
                        f"Reason: <b>No longer recommended by strategy</b>"
                    )

        except Exception as e:
            error_msg = f"Error canceling obsolete orders: {str(e)}"
            logger.error(error_msg)
            self.telegram.notify_error(error_msg)

    def _simulate_cancel_obsolete_orders(self, new_order_signatures: set):
        """Simulate canceling orders in test mode"""
        # Check each active order
        for order_id in list(self.active_orders.keys()):
            order = self.active_orders[order_id]

            # Create signature for this order
            order_signature = f"{order['side']}_{order['setup_type']}_{order['entry_price']:.2f}_{order['stop_loss']:.2f}"

            # If this order signature is not in the new recommendations, cancel it
            if order_signature not in new_order_signatures:
                logger.info(
                    f"TEST MODE: Canceling obsolete order {order_id} that is no longer recommended")

                # Remove from our active orders
                del self.active_orders[order_id]

                # Send notification
                self.telegram.send_message(
                    f"🚫 <b>Order Canceled (Test Mode)</b>\n\n"
                    f"Symbol: <b>{self.symbol}</b>\n"
                    f"Side: <b>{order['side']}</b>\n"
                    f"Setup: <b>{order['setup_type']}</b>\n"
                    f"Entry: <b>${order['entry_price']:.2f}</b>\n"
                    f"Reason: <b>No longer recommended by strategy</b>"
                )

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
    telegram_orders_topic_id = os.environ.get(
        'TELEGRAM_ORDERS_TOPIC_ID', '6216')
    telegram_signals_topic_id = os.environ.get(
        'TELEGRAM_SIGNALS_TOPIC_ID', '6215')

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
        test_mode=test_mode,
        telegram=TelegramNotifier(
            bot_token=telegram_token,
            chat_id=telegram_chat_id,
            enabled=telegram_enabled,
            orders_topic_id=telegram_orders_topic_id,
            signals_topic_id=telegram_signals_topic_id
        )
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
