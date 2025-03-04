import time
import logging
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import os
import signal
import sys
import requests
import asyncio
from binance import AsyncClient, BinanceSocketManager
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crossover_bot.log"),
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
            self.send_message("🤖 EMA Crossover Bot initialized and ready.")
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

    def notify_signal(self, signal_data: Dict):
        """Send notification about a new trading signal"""
        if not self.enabled:
            return

        symbol = signal_data['symbol']
        signal_type = signal_data['signal_type']
        price = signal_data['price']
        ema_fast = signal_data.get('ema_fast', 0)
        ema_slow = signal_data.get('ema_slow', 0)
        volume_ratio = signal_data.get('volume_ratio', 0)

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
            f"Volume Ratio: <b>{volume_ratio:.2f}x</b>"
        )

        if 'support' in signal_data:
            message += f"\nSupport: <b>${signal_data['support']:.2f}</b>"

        if 'resistance' in signal_data:
            message += f"\nResistance: <b>${signal_data['resistance']:.2f}</b>"

        self.send_message(message)

    def notify_order_created(self, order: Dict):
        """Send notification about a new order"""
        if not self.enabled:
            return

        side = order['side']
        symbol = order['symbol']
        entry_price = order['entry_price']
        stop_loss = order['stop_loss']
        take_profit = order['take_profit']
        position_size = order.get('position_size', 0)
        leverage = order.get('leverage', 20)

        message = (
            f"🔔 <b>New {side} Order Created</b>\n\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Entry Price: <b>${entry_price:.2f}</b>\n"
            f"Stop Loss: <b>${stop_loss:.2f}</b>\n"
            f"Take Profit: <b>${take_profit:.2f}</b>\n"
            f"Risk/Reward: <b>{order['risk_reward']:.2f}</b>\n"
            f"Position Size: <b>${position_size:.2f}</b> ({leverage}x)"
        )

        self.send_message(message)

    def notify_error(self, error_message: str):
        """Send notification about an error"""
        if not self.enabled:
            return

        message = f"⚠️ <b>Error</b>\n\n{error_message}"
        self.send_message(message)


class EMACrossoverBot:
    """
    Trading bot that uses EMA crossovers, volume analysis, and support/resistance
    levels to generate trading signals.
    """

    def __init__(self,
                 api_key: str,
                 api_secret: str,
                 symbol: str = "BTCUSDT",
                 interval: str = "15m",
                 fast_ema: int = 9,
                 slow_ema: int = 21,
                 ma_period: int = 50,
                 volume_threshold: float = 1.5,
                 initial_capital: float = 1000.0,
                 max_risk_per_trade: float = 0.02,
                 leverage: int = 20,
                 test_mode: bool = True):
        """
        Initialize the EMA Crossover Trading Bot.

        Parameters:
        -----------
        api_key: Binance API key
        api_secret: Binance API secret
        symbol: Trading symbol (e.g., "BTCUSDT")
        interval: Trading timeframe (e.g., "15m", "1h")
        fast_ema: Fast EMA period
        slow_ema: Slow EMA period
        ma_period: Moving Average period for trend direction
        volume_threshold: Volume threshold as multiplier of average volume
        initial_capital: Starting capital
        max_risk_per_trade: Maximum risk per trade (percentage of capital)
        leverage: Trading leverage
        test_mode: Whether to run in test mode (no real trades)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.interval = interval
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.ma_period = ma_period
        self.volume_threshold = volume_threshold
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.leverage = leverage
        self.test_mode = test_mode

        # Initialize Binance client
        self.client = Client(
            api_key, api_secret) if api_key and api_secret else None

        # Configure properties for operation
        self.running = False
        self.current_price = 0.0
        self.historical_data = pd.DataFrame()
        self.last_analysis_time = None

        # Trade tracking
        self.active_orders = {}
        self.open_positions = {}
        self.trade_history = []

        # Support and resistance levels
        self.support_levels = []
        self.resistance_levels = []

        # Signal state tracking
        self.last_signal = None
        self.in_position = False

        # Telegram notification placeholder
        self.telegram = None

        logger.info(
            f"Initialized EMA Crossover Bot for {symbol} on {interval} timeframe")

    async def _start_socket(self):
        """Start WebSocket connection using asyncio"""
        try:
            # Initialize AsyncClient
            client = await AsyncClient.create(self.api_key, self.api_secret)

            # Initialize BinanceSocketManager
            bsm = BinanceSocketManager(client)

            # Start kline socket
            kline_socket = bsm.kline_socket(
                symbol=self.symbol, interval=self.interval)

            # Process data from socket
            async with kline_socket as stream:
                while self.running:
                    msg = await stream.recv()
                    self._process_kline_message(msg)

            # Close client when finished
            await client.close_connection()

        except Exception as e:
            error_msg = f"Error in WebSocket connection: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.notify_error(error_msg)

    def _fetch_initial_data(self):
        """Fetch historical data for initial analysis"""
        try:
            logger.info(f"Fetching historical data for {self.symbol}")

            # Calculate how many candles we need based on our longest indicator
            lookback_periods = max(self.slow_ema, self.ma_period) * 3

            # Get historical klines from Binance
            klines = self.client.get_historical_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=lookback_periods
            )

            # Convert to DataFrame
            data = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Convert types
            data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
            data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')

            for col in ['open', 'high', 'low', 'close', 'volume']:
                data[col] = data[col].astype(float)

            # Set index to timestamp
            data.set_index('open_time', inplace=True)

            # Keep only OHLCV columns
            self.historical_data = data[[
                'open', 'high', 'low', 'close', 'volume']]

            # Set current price from most recent close
            self.current_price = float(self.historical_data['close'].iloc[-1])

            logger.info(
                f"Fetched {len(self.historical_data)} candles. Current price: {self.current_price}")

            # Calculate indicators for initial data
            self._calculate_indicators()

            # Find support and resistance levels
            self._find_support_resistance_levels()

        except Exception as e:
            error_msg = f"Error fetching historical data: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.notify_error(error_msg)
            raise

    def _calculate_indicators(self):
        """Calculate technical indicators on historical data"""
        try:
            df = self.historical_data.copy()

            # Calculate EMAs
            df['ema_fast'] = df['close'].ewm(
                span=self.fast_ema, adjust=False).mean()
            df['ema_slow'] = df['close'].ewm(
                span=self.slow_ema, adjust=False).mean()
            df['ma'] = df['close'].rolling(window=self.ma_period).mean()

            # Calculate EMA crossover signals
            # 1 = bullish crossover, -1 = bearish crossover, 0 = no crossover
            df['ema_diff'] = df['ema_fast'] - df['ema_slow']
            df['ema_diff_prev'] = df['ema_diff'].shift(1)
            df['crossover'] = 0  # Default: no crossover

            # Bullish crossover: fast EMA crosses above slow EMA
            bullish_crossover = (df['ema_diff'] > 0) & (
                df['ema_diff_prev'] <= 0)
            df.loc[bullish_crossover, 'crossover'] = 1

            # Bearish crossover: fast EMA crosses below slow EMA
            bearish_crossover = (df['ema_diff'] < 0) & (
                df['ema_diff_prev'] >= 0)
            df.loc[bearish_crossover, 'crossover'] = -1

            # Calculate Average True Range for dynamic stop loss
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = abs(df['high'] - df['close'].shift())
            df['low_close'] = abs(df['low'] - df['close'].shift())
            df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=14).mean()

            # Volume analysis
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']

            # Momentum indicators
            df['rsi'] = self._calculate_rsi(df['close'], 14)

            # Volatility measure for adaptive stop loss
            df['volatility'] = df['close'].pct_change().rolling(20).std() * 100

            # Store calculated data
            self.historical_data = df

            # Print current indicator values
            self._print_indicators()

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise

    def _print_indicators(self):
        """Print current technical indicators"""
        if len(self.historical_data) < 2:
            return

        latest = self.historical_data.iloc[-1]
        prev = self.historical_data.iloc[-2]

        # Format indicator values
        ema_fast = latest['ema_fast']
        ema_slow = latest['ema_slow']
        ma = latest['ma']
        rsi = latest['rsi']
        volume_ratio = latest['volume_ratio']
        volatility = latest['volatility']
        atr = latest['atr']

        # Calculate EMA crossover status
        if latest['ema_fast'] > latest['ema_slow'] and prev['ema_fast'] <= prev['ema_slow']:
            crossover = "🟢 BULLISH CROSSOVER"
        elif latest['ema_fast'] < latest['ema_slow'] and prev['ema_fast'] >= prev['ema_slow']:
            crossover = "🔴 BEARISH CROSSOVER"
        elif latest['ema_fast'] > latest['ema_slow']:
            crossover = "↗️ Bullish Trend"
        else:
            crossover = "↘️ Bearish Trend"

        # Log indicators
        logger.info(f"Current Price: ${self.current_price:.2f}")
        logger.info(f"EMA Status: {crossover}")
        logger.info(
            f"EMA({self.fast_ema}): {ema_fast:.2f}, EMA({self.slow_ema}): {ema_slow:.2f}, MA({self.ma_period}): {ma:.2f}")
        logger.info(
            f"RSI: {rsi:.2f}, Volume Ratio: {volume_ratio:.2f}x, Volatility: {volatility:.2f}%")

        # Send to Telegram (limited frequency)
        if hasattr(self, 'telegram') and self.telegram.enabled:
            current_time = datetime.now()
            if not hasattr(self, 'last_indicator_notification') or \
               (current_time - self.last_indicator_notification > timedelta(minutes=30)):

                message = (
                    f"📈 <b>Technical Indicators</b>\n\n"
                    f"Symbol: <b>{self.symbol}</b>\n"
                    f"Current Price: <b>${self.current_price:.2f}</b>\n"
                    f"Status: <b>{crossover}</b>\n\n"
                    f"EMA({self.fast_ema}): <b>{ema_fast:.2f}</b>\n"
                    f"EMA({self.slow_ema}): <b>{ema_slow:.2f}</b>\n"
                    f"MA({self.ma_period}): <b>{ma:.2f}</b>\n"
                    f"RSI: <b>{rsi:.2f}</b>\n"
                    f"Volume Ratio: <b>{volume_ratio:.2f}x</b>\n"
                    f"Volatility: <b>{volatility:.2f}%</b>\n"
                    f"ATR: <b>{atr:.2f}</b>"
                )

                self.telegram.send_message(message)
                self.last_indicator_notification = current_time

    def _find_support_resistance_levels(self):
        """Identify support and resistance levels using swing highs/lows"""
        try:
            # Check if we have enough data
            if len(self.historical_data) < 30:
                logger.warning(
                    "Not enough data to identify support/resistance levels")
                return

            # Reset levels
            self.support_levels = []
            self.resistance_levels = []

            # Look for swing highs (resistance)
            for i in range(2, len(self.historical_data)-2):
                if (self.historical_data['high'].iloc[i] > self.historical_data['high'].iloc[i-1] and
                    self.historical_data['high'].iloc[i] > self.historical_data['high'].iloc[i-2] and
                    self.historical_data['high'].iloc[i] > self.historical_data['high'].iloc[i+1] and
                        self.historical_data['high'].iloc[i] > self.historical_data['high'].iloc[i+2]):
                    self.resistance_levels.append(
                        self.historical_data['high'].iloc[i])

            # Look for swing lows (support)
            for i in range(2, len(self.historical_data)-2):
                if (self.historical_data['low'].iloc[i] < self.historical_data['low'].iloc[i-1] and
                    self.historical_data['low'].iloc[i] < self.historical_data['low'].iloc[i-2] and
                    self.historical_data['low'].iloc[i] < self.historical_data['low'].iloc[i+1] and
                        self.historical_data['low'].iloc[i] < self.historical_data['low'].iloc[i+2]):
                    self.support_levels.append(
                        self.historical_data['low'].iloc[i])

            # Sort levels
            self.support_levels = sorted(self.support_levels)
            self.resistance_levels = sorted(self.resistance_levels)

            # Keep only significant levels
            self._filter_significant_levels()

            logger.info(
                f"Found {len(self.support_levels)} support and {len(self.resistance_levels)} resistance levels")

        except Exception as e:
            error_msg = f"Error finding support/resistance levels: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.notify_error(error_msg)

    def _filter_significant_levels(self):
        """Filter out less significant support/resistance levels to reduce noise"""
        # Group nearby levels (within 0.5% of price)
        grouped_support = self._group_nearby_levels(self.support_levels)
        grouped_resistance = self._group_nearby_levels(self.resistance_levels)

        # Replace with average price from each group
        self.support_levels = [sum(group)/len(group)
                               for group in grouped_support]
        self.resistance_levels = [sum(group)/len(group)
                                  for group in grouped_resistance]

        # Keep only levels within a reasonable range of current price
        current_price = self.current_price

        # Filter support levels (keep levels that are below current price)
        self.support_levels = [level for level in self.support_levels
                               if level < current_price and level > current_price * 0.8]

        # Filter resistance levels (keep levels that are above current price)
        self.resistance_levels = [level for level in self.resistance_levels
                                  if level > current_price and level < current_price * 1.2]

        # Limit to the 3 closest levels in each direction
        if len(self.support_levels) > 3:
            self.support_levels = sorted(
                self.support_levels, key=lambda x: abs(current_price - x))[:3]

        if len(self.resistance_levels) > 3:
            self.resistance_levels = sorted(
                self.resistance_levels, key=lambda x: abs(current_price - x))[:3]

        # Sort levels
        self.support_levels = sorted(self.support_levels)
        self.resistance_levels = sorted(self.resistance_levels)

    def _group_nearby_levels(self, levels, threshold_percent=0.5):
        """Group levels that are within threshold_percent of each other"""
        if not levels:
            return []

        # Convert percent to price amount
        threshold = self.current_price * (threshold_percent / 100)

        # Sort levels
        sorted_levels = sorted(levels)

        # Initialize groups
        groups = [[sorted_levels[0]]]

        # Group nearby levels
        for level in sorted_levels[1:]:
            if level - groups[-1][-1] <= threshold:
                groups[-1].append(level)
            else:
                groups.append([level])

        return groups

    def _process_kline_message(self, msg):
        """Process kline message from WebSocket"""
        try:
            # Check if it's a kline message
            if 'k' in msg:
                kline = msg['k']

                # Check if candle is closed
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

                # Keep only enough data for indicators
                max_lookback = max(self.slow_ema, self.ma_period, 40) * 3
                if len(self.historical_data) > max_lookback:
                    self.historical_data = self.historical_data.iloc[-max_lookback:]

                # Update current price
                self.current_price = close_price

                logger.info(
                    f"New candle closed: {timestamp}, Close: {close_price}")

                # Run analysis on new candle
                self._run_analysis()

        except Exception as e:
            error_msg = f"Error processing kline message: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.notify_error(error_msg)

    def _run_analysis(self):
        """Analyze market and generate trading signals"""
        try:
            self.last_analysis_time = datetime.now()

            # Calculate technical indicators
            self._calculate_indicators()

            # Update support and resistance levels
            self._find_support_resistance_levels()

            # Print support and resistance levels
            self._print_levels()

            # Generate trading signals
            self._generate_signals()

            # Check if we need to exit any positions
            if self.in_position:
                self._check_exit_conditions()

        except Exception as e:
            error_msg = f"Error running analysis: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.notify_error(error_msg)

    def _print_levels(self):
        """Print support and resistance levels with current price context"""
        current_price = self.current_price

        # Format support levels
        support_str = ", ".join(
            [f"${level:.2f}" for level in self.support_levels])
        resistance_str = ", ".join(
            [f"${level:.2f}" for level in self.resistance_levels])

        # Log to console
        logger.info(f"Current price: ${current_price:.2f}")
        logger.info(f"Support levels: {support_str}")
        logger.info(f"Resistance levels: {resistance_str}")

        # Send to Telegram if enabled (only every hour to avoid spam)
        if hasattr(self, 'telegram') and self.telegram.enabled:
            if not hasattr(self, 'last_levels_notification') or \
               (datetime.now() - self.last_levels_notification > timedelta(hours=1)):

                message = (
                    f"📊 <b>Price Levels Update</b>\n\n"
                    f"Symbol: <b>{self.symbol}</b>\n"
                    f"Current Price: <b>${current_price:.2f}</b>\n\n"
                    f"🟢 <b>Support Levels:</b>\n{support_str}\n\n"
                    f"🔴 <b>Resistance Levels:</b>\n{resistance_str}"
                )

                self.telegram.send_message(message)
                self.last_levels_notification = datetime.now()

    def _generate_signals(self):
        """Generate trading signals based on indicators and conditions"""
        try:
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
                        'timestamp': datetime.now()
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
                    if self.telegram:
                        self.telegram.notify_signal(signal_data)

                    # If not in test mode, place order
                    if not self.test_mode and not self.in_position:
                        self._place_order(signal_data)

        except Exception as e:
            error_msg = f"Error generating signals: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.notify_error(error_msg)

    def _is_near_support(self, price):
        """Check if price is near a support level (within 1%)"""
        if not self.support_levels:
            return False

        threshold = price * 0.01  # 1% of current price

        # Check if price is within threshold of any support level
        for level in self.support_levels:
            if abs(price - level) < threshold:
                return True

        return False

    def _is_near_resistance(self, price):
        """Check if price is near a resistance level (within 1%)"""
        if not self.resistance_levels:
            return False

        threshold = price * 0.01  # 1% of current price

        # Check if price is within threshold of any resistance level
        for level in self.resistance_levels:
            if abs(price - level) < threshold:
                return True

        return False

    def _get_closest_support(self, price):
        """Get the closest support level below current price"""
        if not self.support_levels:
            return None

        # Filter levels below price
        levels_below = [
            level for level in self.support_levels if level < price]

        if not levels_below:
            return None

        # Return the highest support level below price
        return max(levels_below)

    def _get_closest_resistance(self, price):
        """Get the closest resistance level above current price"""
        if not self.resistance_levels:
            return None

        # Filter levels above price
        levels_above = [
            level for level in self.resistance_levels if level > price]

        if not levels_above:
            return None

        # Return the lowest resistance level above price
        return min(levels_above)

    def _place_order(self, signal_data: Dict):
        """Place order based on trading signal"""
        try:
            side = signal_data['signal_type']
            price = self.current_price

            # Skip if we're already in position
            if self.in_position:
                logger.info(
                    f"Already in {self.last_signal} position, skipping {side} signal")
                return None

            # Enhanced dynamic stop loss based on ATR and volatility
            latest = self.historical_data.iloc[-1]
            atr = latest['atr']
            volatility = latest['volatility']

            # Base multiplier that adapts to market volatility
            # Higher volatility = wider stop loss
            # Range between 1 and 3
            volatility_factor = min(max(1.0, volatility / 5), 3.0)
            atr_multiplier = volatility_factor * 1.5

            # Calculate dynamic stop loss
            if side == "BUY":
                # For longs: price - (ATR * multiplier)
                # Adjust based on nearest support level
                base_stop = price - (atr * atr_multiplier)

                # Check if there's a support level between base_stop and price
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

            else:  # SELL
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
            risk_amount = self.capital * self.max_risk_per_trade
            price_risk = abs(price - stop_loss)
            position_size = (risk_amount / price_risk) * self.leverage

            # Create order object
            order = {
                'symbol': self.symbol,
                'side': side,
                'entry_price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': risk_reward,
                'position_size': position_size,
                'leverage': self.leverage,
                'atr': atr,
                'volatility': volatility,
                'indicators': {
                    'ema_fast': latest['ema_fast'],
                    'ema_slow': latest['ema_slow'],
                    'rsi': latest['rsi'],
                    'volume_ratio': latest['volume_ratio']
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'CREATED'
            }

            # In a real implementation, you would place the order with the exchange here
            logger.info(
                f"Created {side} order at ${price:.2f} with stop loss ${stop_loss:.2f} and take profit ${take_profit:.2f}")
            logger.info(
                f"Risk:Reward = {risk_reward:.2f}, ATR = {atr:.2f}, Volatility = {volatility:.2f}%")

            # Notify
            if self.telegram:
                # Enhanced order notification
                message = (
                    f"🔔 <b>New {side} Order Created</b>\n\n"
                    f"Symbol: <b>{self.symbol}</b>\n"
                    f"Entry Price: <b>${price:.2f}</b>\n"
                    f"Stop Loss: <b>${stop_loss:.2f}</b>\n"
                    f"Take Profit: <b>${take_profit:.2f}</b>\n"
                    f"Risk/Reward: <b>{risk_reward:.2f}</b>\n"
                    f"Position Size: <b>${position_size:.2f}</b> ({self.leverage}x)\n\n"
                    f"<b>Market Conditions:</b>\n"
                    f"EMA({self.fast_ema}): <b>{latest['ema_fast']:.2f}</b>\n"
                    f"EMA({self.slow_ema}): <b>{latest['ema_slow']:.2f}</b>\n"
                    f"RSI: <b>{latest['rsi']:.2f}</b>\n"
                    f"Volume Ratio: <b>{latest['volume_ratio']:.2f}x</b>"
                )
                self.telegram.send_message(message)

            # Set position flag
            self.in_position = True
            self.last_signal = side

            return order

        except Exception as e:
            error_msg = f"Error placing order: {str(e)}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.notify_error(error_msg)
            return None

    def _check_exit_conditions(self):
        """Check if we should exit positions based on indicators"""
        # This would check for exit signals for open positions
        # Simplified version - would need to integrate with actual order management
        latest = self.historical_data.iloc[-1]

        if self.last_signal == "BUY":
            # Check for exit signal for long position
            if latest['ema_fast'] < latest['ema_slow']:
                logger.info("Exit signal for long position")
                self.in_position = False
                self.last_signal = None

        elif self.last_signal == "SELL":
            # Check for exit signal for short position
            if latest['ema_fast'] > latest['ema_slow']:
                logger.info("Exit signal for short position")
                self.in_position = False
                self.last_signal = None

    def start(self):
        """Start the trading bot"""
        if self.running:
            logger.warning("Trading bot is already running")
            return

        try:
            logger.info("Starting EMA Crossover Trading Bot...")
            self.running = True

            # Fetch initial data
            self._fetch_initial_data()

            # Start WebSocket connection in a separate thread
            self.socket_thread = threading.Thread(
                target=lambda: asyncio.run(self._start_socket()))
            self.socket_thread.daemon = True
            self.socket_thread.start()

            logger.info(f"Started WebSocket connection for {self.symbol}")

            # Send notification
            if self.telegram:
                self.telegram.send_message(
                    f"🚀 <b>EMA Crossover Bot Started</b>\n\n"
                    f"Symbol: <b>{self.symbol}</b>\n"
                    f"Timeframe: <b>{self.interval}</b>\n"
                    f"Current Price: <b>${self.current_price:.2f}</b>\n"
                    f"Fast EMA: <b>{self.fast_ema}</b>\n"
                    f"Slow EMA: <b>{self.slow_ema}</b>\n"
                    f"Mode: <b>{'Test' if self.test_mode else 'Live'}</b>"
                )

            # Run initial analysis
            self._run_analysis()

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

        logger.info("Stopping EMA Crossover Bot...")
        self.running = False

        # WebSocket will close when self.running = False

        # Send notification
        if self.telegram:
            self.telegram.send_message("🛑 <b>EMA Crossover Bot Stopped</b>")

        logger.info("Trading bot stopped")

    def get_status(self) -> Dict:
        """Get current status of the trading bot"""
        status = {
            'running': self.running,
            'symbol': self.symbol,
            'interval': self.interval,
            'current_price': self.current_price,
            'fast_ema': self.fast_ema,
            'slow_ema': self.slow_ema,
            'last_signal': self.last_signal,
            'in_position': self.in_position,
            'last_analysis': self.last_analysis_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_analysis_time else None,
            'test_mode': self.test_mode,
            'capital': self.capital
        }

        return status

    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up/down if down != 0 else 0
            rsi[i] = 100. - 100./(1. + rs)

        return rsi


if __name__ == "__main__":
    load_dotenv()

    # Load configuration from environment variables
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')
    symbol = os.environ.get('TRADING_SYMBOL', 'BTCUSDT')
    interval = os.environ.get('TRADING_INTERVAL', '15m')
    fast_ema = int(os.environ.get('FAST_EMA', '9'))
    slow_ema = int(os.environ.get('SLOW_EMA', '21'))
    ma_period = int(os.environ.get('MA_PERIOD', '50'))
    volume_threshold = float(os.environ.get('VOLUME_THRESHOLD', '1.5'))
    initial_capital = float(os.environ.get('INITIAL_CAPITAL', '1000.0'))
    max_risk_per_trade = float(os.environ.get('MAX_RISK_PER_TRADE', '0.02'))
    leverage = int(os.environ.get('LEVERAGE', '20'))
    test_mode = os.environ.get('TEST_MODE', 'true').lower() == 'true'

    # Telegram settings
    telegram_enabled = os.environ.get(
        'TELEGRAM_ENABLED', 'false').lower() == 'true'
    telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')

    # Create trading bot
    bot = EMACrossoverBot(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        interval=interval,
        fast_ema=fast_ema,
        slow_ema=slow_ema,
        ma_period=ma_period,
        volume_threshold=volume_threshold,
        initial_capital=initial_capital,
        max_risk_per_trade=max_risk_per_trade,
        leverage=leverage,
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
            time.sleep(60)  # Check status every minute
            status = bot.get_status()
            logger.info(f"Bot status: {status}")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
        bot.stop()
