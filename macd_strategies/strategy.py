import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List
import os
from indicators.rsi import calculate_macd, calculate_rsi


# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("macd_rsi_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MacdRsiStrategy:
    def __init__(self,
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 volume_ma_period: int = 20,
                 min_volume_ratio: float = 2.0,
                 ema_short: int = 34,
                 ema_long: int = 89):
        """
        Initialize trading strategy based on MACD, RSI and EMA

        Parameters:
        -----------
        rsi_period: RSI period
        rsi_overbought: RSI overbought threshold
        rsi_oversold: RSI oversold threshold
        macd_fast: Fast MACD EMA
        macd_slow: Slow MACD EMA
        macd_signal: MACD signal period
        volume_ma_period: Volume MA period
        min_volume_ratio: Minimum volume ratio
        ema_short: Short-term EMA period (default: 34)
        ema_long: Long-term EMA period (default: 89)
        """
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.volume_ma_period = volume_ma_period
        self.min_volume_ratio = min_volume_ratio
        self.ema_short = ema_short
        self.ema_long = ema_long

    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """
        Analyze market and create trading signals with comprehensive analysis
        """
        analysis = {}

        # Calculate indicators
        df = self._calculate_indicators(data)

        # Analyze market structure and trend
        market_structure = self._analyze_market_structure(df)

        # Analyze support and resistance levels
        sr_levels = self._analyze_support_resistance(df)

        # Analyze candle patterns
        candle_patterns = self._analyze_candle_patterns(df)

        # Analyze MACD
        macd_signals = self._analyze_macd(df)

        # Analyze RSI
        rsi_signals = self._analyze_rsi(df)

        # Check for RSI divergences
        divergence = self._analyze_divergence(df)

        # Analyze volume
        volume_signals = self._analyze_volume(df)

        # Analyze EMA
        ema_signals = self._analyze_ema(df)

        # Analyze market volatility
        volatility = self._analyze_volatility(df)

        # Analyze market momentum
        momentum = self._analyze_momentum(df)

        # Combine signals
        signals = self._combine_signals(
            df,
            market_structure,
            sr_levels,
            candle_patterns,
            macd_signals,
            rsi_signals,
            divergence,
            volume_signals,
            ema_signals,
            volatility,
            momentum
        )

        analysis['signals'] = signals
        analysis['market_structure'] = market_structure
        analysis['sr_levels'] = sr_levels
        analysis['indicators'] = {
            'macd': macd_signals,
            'rsi': rsi_signals,
            'divergence': divergence,
            'volume': volume_signals,
            'patterns': candle_patterns,
            'ema': ema_signals,
            'volatility': volatility,
            'momentum': momentum
        }
        analysis['current_price'] = df['close'].iloc[-1]

        return analysis

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df = df.copy()

        # RSI calculation - fixed implementation
        delta = df['close'].diff()
        delta = delta.fillna(0)  # Fill NaN values to avoid calculation errors

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Use simple calculation for first periods to avoid NaN values
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()

        # Avoid division by zero
        avg_loss = avg_loss.replace(0, 0.000001)

        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD calculation
        ema_fast = df['close'].ewm(
            span=self.macd_fast, adjust=False, min_periods=1).mean()
        ema_slow = df['close'].ewm(
            span=self.macd_slow, adjust=False, min_periods=1).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(
            span=self.macd_signal, adjust=False, min_periods=1).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # EMA calculations with min_periods to avoid NaN
        df['ema8'] = df['close'].ewm(
            span=8, adjust=False, min_periods=1).mean()
        df['ema21'] = df['close'].ewm(
            span=21, adjust=False, min_periods=1).mean()
        df['ema34'] = df['close'].ewm(
            span=self.ema_short, adjust=False, min_periods=1).mean()
        df['ema89'] = df['close'].ewm(
            span=self.ema_long, adjust=False, min_periods=1).mean()
        df['ema200'] = df['close'].ewm(
            span=200, adjust=False, min_periods=1).mean()

        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(
            window=self.volume_ma_period, min_periods=1).mean()
        # Handle potential division by zero
        df['volume_ma'] = df['volume_ma'].replace(0, 0.000001)
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # On-Balance Volume (OBV) calculation
        df['obv'] = 0
        obv = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv += df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv -= df['volume'].iloc[i]
            df.loc[df.index[i], 'obv'] = obv  # Use index for safer assignment

        # Bollinger Bands with min_periods to avoid NaN values
        df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['close'].rolling(window=20, min_periods=1).std().fillna(0)
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        # Avoid division by zero
        df['bb_middle'] = df['bb_middle'].replace(0, 0.000001)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Average True Range (ATR) for volatility
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift().fillna(df['open']))
        low_close = abs(df['low'] - df['close'].shift().fillna(df['open']))

        # Fix potential NaN issues in ranges calculation
        ranges = pd.concat([high_low, high_close, low_close], axis=1).fillna(0)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14, min_periods=1).mean()

        # Stochastic Oscillator with error handling
        lowest_low = df['low'].rolling(window=14, min_periods=1).min()
        highest_high = df['high'].rolling(window=14, min_periods=1).max()
        # Avoid division by zero
        height_diff = (highest_high - lowest_low).replace(0, 0.000001)
        df['stoch_k'] = 100 * ((df['close'] - lowest_low) / height_diff)
        df['stoch_d'] = df['stoch_k'].rolling(window=3, min_periods=1).mean()

        # Rate of Change (ROC) with NaN handling
        df['roc'] = df['close'].pct_change(periods=10).fillna(0) * 100

        # Directional Movement Index (DMI) with NaN handling
        plus_dm = df['high'].diff().fillna(0)
        minus_dm = df['low'].diff(-1).abs().fillna(0)
        plus_dm = plus_dm.mask(plus_dm < 0, 0)
        minus_dm = minus_dm.mask(minus_dm < 0, 0)

        tr_sum = true_range.rolling(
            window=14, min_periods=1).sum().replace(0, 0.000001)
        plus_di = 100 * (plus_dm.rolling(window=14,
                         min_periods=1).sum() / tr_sum)
        minus_di = 100 * (minus_dm.rolling(window=14,
                          min_periods=1).sum() / tr_sum)

        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # Calculate ADX with error handling
        di_sum = (plus_di + minus_di).replace(0, 0.000001)
        df['adx'] = (abs(plus_di - minus_di) / di_sum *
                     100).rolling(window=14, min_periods=1).mean()

        # Fill remaining NaN values with 0 to ensure clean data
        df = df.fillna(0)

        return df  # Make sure to return the DataFrame

    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze overall market structure and trend"""
        # Get current price and EMAs
        current_price = df['close'].iloc[-1]
        print("current_price", current_price)
        ema8 = df['ema8'].iloc[-1]
        ema21 = df['ema21'].iloc[-1]
        ema34 = df['ema34'].iloc[-1]
        ema89 = df['ema89'].iloc[-1]
        ema200 = df['ema200'].iloc[-1]

        # ADX for trend strength
        adx = df['adx'].iloc[-1]

        # Determine short-term trend
        short_term_trend = "BULLISH" if ema8 > ema21 else "BEARISH"

        # Determine medium-term trend
        medium_term_trend = "BULLISH" if ema34 > ema89 else "BEARISH"

        # Determine long-term trend
        long_term_trend = "BULLISH" if current_price > ema200 else "BEARISH"

        # Check if all trends align for stronger signal
        trends_aligned = (short_term_trend ==
                          medium_term_trend == long_term_trend)

        # Check market trend strength
        trend_strength = "STRONG" if adx > 25 else "WEAK" if adx > 20 else "RANGING"

        # Check for trend changes
        trend_change = False
        if len(df) >= 5:
            prev_short_term = "BULLISH" if df['ema8'].iloc[-5] > df['ema21'].iloc[-5] else "BEARISH"
            if prev_short_term != short_term_trend:
                trend_change = True

        # Higher timeframe context
        above_key_emas = (current_price > ema34 and current_price > ema89)
        below_key_emas = (current_price < ema34 and current_price < ema89)

        return {
            'short_term_trend': short_term_trend,
            'medium_term_trend': medium_term_trend,
            'long_term_trend': long_term_trend,
            'trend_strength': trend_strength,
            'trends_aligned': trends_aligned,
            'trend_change': trend_change,
            'above_key_emas': above_key_emas,
            'below_key_emas': below_key_emas,
            'adx': adx
        }

    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Identify key support and resistance levels"""
        highs = df['high'].values
        lows = df['low'].values
        close = df['close'].values

        # Current price
        current_price = close[-1]

        # Identify swing highs and lows
        swing_highs = []
        swing_lows = []

        # Simple method to identify swings (more sophisticated methods can be used)
        window = 5
        for i in range(window, len(df) - window):
            # Check for swing high
            if highs[i] == max(highs[i-window:i+window+1]):
                swing_highs.append((i, highs[i]))

            # Check for swing low
            if lows[i] == min(lows[i-window:i+window+1]):
                swing_lows.append((i, lows[i]))

        # Filter recent swing points (last 30 candles)
        recent_swing_highs = [price for idx,
                              price in swing_highs if len(df) - idx <= 30]
        recent_swing_lows = [price for idx,
                             price in swing_lows if len(df) - idx <= 30]

        # Find closest support and resistance
        closest_resistance = min(
            [price for price in recent_swing_highs if price > current_price], default=None)
        closest_support = max(
            [price for price in recent_swing_lows if price < current_price], default=None)

        # Check if price is near support or resistance (within 1% range)
        near_support = closest_support and (
            current_price - closest_support) / closest_support < 0.01
        near_resistance = closest_resistance and (
            closest_resistance - current_price) / current_price < 0.01

        # Calculate distance to support/resistance as percentage
        distance_to_support = ((current_price - closest_support) /
                               current_price * 100) if closest_support else None
        distance_to_resistance = (
            (closest_resistance - current_price) / current_price * 100) if closest_resistance else None

        return {
            'closest_resistance': closest_resistance,
            'closest_support': closest_support,
            'near_support': near_support,
            'near_resistance': near_resistance,
            'distance_to_support': distance_to_support,
            'distance_to_resistance': distance_to_resistance
        }

    def _analyze_divergence(self, df: pd.DataFrame) -> Dict:
        """Detect divergences between price and oscillators (RSI, MACD)"""
        # Get price and oscillator data for the last 20 candles
        window = min(20, len(df)-1)

        prices = df['close'].iloc[-window:].values
        rsi_values = df['rsi'].iloc[-window:].values
        macd_hist = df['macd_hist'].iloc[-window:].values

        # Initialize divergence flags
        bullish_rsi_div = False
        bearish_rsi_div = False
        bullish_macd_div = False
        bearish_macd_div = False

        # Simple divergence detection (can be improved with more sophisticated methods)
        # Bullish divergence: Lower lows in price but higher lows in oscillator
        # Bearish divergence: Higher highs in price but lower highs in oscillator

        # Find local min/max in price and oscillators
        for i in range(2, window-2):
            # Check for price local minimum
            if prices[i] < prices[i-1] and prices[i] < prices[i-2] and prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                # RSI bullish divergence
                if rsi_values[i] > rsi_values[i-2] and prices[i] < prices[i-2]:
                    bullish_rsi_div = True
                # MACD bullish divergence
                if macd_hist[i] > macd_hist[i-2] and prices[i] < prices[i-2]:
                    bullish_macd_div = True

            # Check for price local maximum
            if prices[i] > prices[i-1] and prices[i] > prices[i-2] and prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                # RSI bearish divergence
                if rsi_values[i] < rsi_values[i-2] and prices[i] > prices[i-2]:
                    bearish_rsi_div = True
                # MACD bearish divergence
                if macd_hist[i] < macd_hist[i-2] and prices[i] > prices[i-2]:
                    bearish_macd_div = True

        return {
            'bullish_rsi_divergence': bullish_rsi_div,
            'bearish_rsi_divergence': bearish_rsi_div,
            'bullish_macd_divergence': bullish_macd_div,
            'bearish_macd_divergence': bearish_macd_div
        }

    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """Analyze market volatility"""
        # Get ATR and BB width for volatility measurement
        current_atr = df['atr'].iloc[-1]
        avg_atr = df['atr'].iloc[-20:].mean()
        bb_width = df['bb_width'].iloc[-1]
        avg_bb_width = df['bb_width'].iloc[-20:].mean()

        # Determine if volatility is high, low, or expanding/contracting
        high_volatility = current_atr > avg_atr * 1.5
        low_volatility = current_atr < avg_atr * 0.7
        expanding_volatility = df['bb_width'].iloc[-1] > df['bb_width'].iloc[-2] > df['bb_width'].iloc[-3]
        contracting_volatility = df['bb_width'].iloc[-1] < df['bb_width'].iloc[-2] < df['bb_width'].iloc[-3]

        # Check for volatility squeeze (potential breakout setup)
        volatility_squeeze = contracting_volatility and df[
            'bb_width'].iloc[-1] < df['bb_width'].iloc[-20:].min() * 1.2

        # Calculate percent change over recent periods
        daily_change = abs(df['close'].iloc[-1] /
                           df['close'].iloc[-2] - 1) * 100
        weekly_change = abs(df['close'].iloc[-1] / df['close'].iloc[-5] -
                            1) * 100 if len(df) >= 5 else daily_change

        return {
            'atr': current_atr,
            'bb_width': bb_width,
            'high_volatility': high_volatility,
            'low_volatility': low_volatility,
            'expanding_volatility': expanding_volatility,
            'contracting_volatility': contracting_volatility,
            'volatility_squeeze': volatility_squeeze,
            'daily_change': daily_change,
            'weekly_change': weekly_change
        }

    def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze market momentum"""
        # Get momentum indicators
        current_roc = df['roc'].iloc[-1]
        current_stoch_k = df['stoch_k'].iloc[-1]
        current_stoch_d = df['stoch_d'].iloc[-1]

        # Determine momentum direction and strength
        strong_bullish = current_roc > 5 and current_stoch_k > 80 and current_stoch_k > current_stoch_d
        strong_bearish = current_roc < - \
            5 and current_stoch_k < 20 and current_stoch_k < current_stoch_d

        # Check momentum alignment with price
        price_momentum_aligned = (df['close'].iloc[-1] > df['close'].iloc[-2] and current_roc > 0) or \
                                 (df['close'].iloc[-1] <
                                  df['close'].iloc[-2] and current_roc < 0)

        # Check for overbought/oversold conditions
        overbought = current_stoch_k > 80 and df['rsi'].iloc[-1] > 70
        oversold = current_stoch_k < 20 and df['rsi'].iloc[-1] < 30

        return {
            'roc': current_roc,
            'stoch_k': current_stoch_k,
            'stoch_d': current_stoch_d,
            'strong_bullish': strong_bullish,
            'strong_bearish': strong_bearish,
            'price_momentum_aligned': price_momentum_aligned,
            'overbought': overbought,
            'oversold': oversold
        }

    def _combine_signals(self, df: pd.DataFrame, market_structure: Dict, sr_levels: Dict,
                         candle_patterns: Dict, macd_signals: Dict, rsi_signals: Dict,
                         divergence: Dict, volume_signals: Dict, ema_signals: Dict,
                         volatility: Dict, momentum: Dict) -> List[Dict]:
        """Combine all signals with improved weighting and strategy context"""
        signals = []
        current_price = df['close'].iloc[-1]

        # ADVANCED BUY SIGNAL EVALUATION
        buy_score = 0
        buy_reasons = []

        # 1. Check trend alignment - most important factor (30%)
        if market_structure['short_term_trend'] == 'BULLISH':
            buy_score += 1.5
            buy_reasons.append("Short-term trend bullish")

        if market_structure['medium_term_trend'] == 'BULLISH':
            buy_score += 1.5
            buy_reasons.append("Medium-term trend bullish")

        if market_structure['trends_aligned'] and market_structure['short_term_trend'] == 'BULLISH':
            buy_score += 2.0
            buy_reasons.append("All timeframes aligned bullish")

        # 2. Check EMA signals (20%)
        if ema_signals['buy_signal']:
            buy_score += ema_signals['strength'] * 0.2
            if ema_signals['is_golden_cross']:
                buy_reasons.append("Golden Cross (EMA34 above EMA89)")
                buy_score += 1.0
            if ema_signals['pullback_to_ema34'] or ema_signals['pullback_to_ema89']:
                buy_reasons.append("Price pullback to EMA support")
                buy_score += 1.0

        # 3. Check indicators: MACD, RSI, etc. (15%)
        if macd_signals['buy_signal']:
            buy_score += macd_signals['strength'] * 0.15
            buy_reasons.append(
                f"MACD bullish (strength: {macd_signals['strength']:.1f})")

        if rsi_signals['oversold']:
            buy_score += rsi_signals['strength'] * 0.15
            buy_reasons.append(f"RSI oversold at {rsi_signals['current']:.1f}")

        # 4. Check divergences (high weight due to reliability) (10%)
        if divergence['bullish_rsi_divergence'] or divergence['bullish_macd_divergence']:
            buy_score += 1.0
            buy_reasons.append("Bullish divergence detected")

        # 5. Check candle patterns (10%)
        if candle_patterns['bullish']:
            buy_score += candle_patterns['strength'] * 0.1
            buy_reasons.append(
                f"Bullish {candle_patterns['pattern_name']} pattern")

        # 6. Check support/resistance (10%)
        if sr_levels['near_support']:
            buy_score += 1.0
            buy_reasons.append(
                f"Price near support level {sr_levels['closest_support']:.2f}")

        # 7. Check volume (5%)
        if volume_signals['bullish_volume']:
            buy_score += volume_signals['strength'] * 0.05
            buy_reasons.append(
                f"Strong volume ({volume_signals['volume_ratio']:.1f}x avg)")

        # 8. Check volatility and momentum together
        if volatility['volatility_squeeze'] and momentum['roc'] > 0:
            buy_score += 0.5
            buy_reasons.append("Volatility squeeze with positive momentum")

        if momentum['oversold'] and market_structure['medium_term_trend'] == 'BULLISH':
            buy_score += 1.0
            buy_reasons.append("Oversold in bullish trend")

        # ADVANCED SELL SIGNAL EVALUATION - follows similar pattern as buy
        sell_score = 0
        sell_reasons = []

        # 1. Check trend alignment (30%)
        if market_structure['short_term_trend'] == 'BEARISH':
            sell_score += 1.5
            sell_reasons.append("Short-term trend bearish")

        if market_structure['medium_term_trend'] == 'BEARISH':
            sell_score += 1.5
            sell_reasons.append("Medium-term trend bearish")

        if market_structure['trends_aligned'] and market_structure['short_term_trend'] == 'BEARISH':
            sell_score += 2.0
            sell_reasons.append("All timeframes aligned bearish")

        # 2. Check EMA signals (20%)
        if ema_signals['sell_signal']:
            sell_score += ema_signals['strength'] * 0.2
            if ema_signals['is_death_cross']:
                sell_reasons.append("Death Cross (EMA34 below EMA89)")
                sell_score += 1.0

        # 3. Check indicators: MACD, RSI, etc. (15%)
        if macd_signals['sell_signal']:
            sell_score += macd_signals['strength'] * 0.15
            sell_reasons.append(
                f"MACD bearish (strength: {macd_signals['strength']:.1f})")

        if rsi_signals['overbought']:
            sell_score += rsi_signals['strength'] * 0.15
            sell_reasons.append(
                f"RSI overbought at {rsi_signals['current']:.1f}")

        # 4. Check divergences (10%)
        if divergence['bearish_rsi_divergence'] or divergence['bearish_macd_divergence']:
            sell_score += 1.0
            sell_reasons.append("Bearish divergence detected")

        # 5. Check candle patterns (10%)
        if candle_patterns['bearish']:
            sell_score += candle_patterns['strength'] * 0.1
            sell_reasons.append(
                f"Bearish {candle_patterns['pattern_name']} pattern")

        # 6. Check support/resistance (10%)
        if sr_levels['near_resistance']:
            sell_score += 1.0
            sell_reasons.append(
                f"Price near resistance level {sr_levels['closest_resistance']:.2f}")

        # 7. Check volume (5%)
        if volume_signals['bearish_volume']:
            sell_score += volume_signals['strength'] * 0.05
            sell_reasons.append(
                f"Strong volume ({volume_signals['volume_ratio']:.1f}x avg)")

        # 8. Check volatility and momentum together
        if volatility['volatility_squeeze'] and momentum['roc'] < 0:
            sell_score += 0.5
            sell_reasons.append("Volatility squeeze with negative momentum")

        if momentum['overbought'] and market_structure['medium_term_trend'] == 'BEARISH':
            sell_score += 1.0
            sell_reasons.append("Overbought in bearish trend")

        # Threshold for generating signals
        threshold = 8  # Higher threshold for more stringent requirements

        # Create signals if score exceeds threshold
        if buy_score > threshold:
            signal = {
                'signal_type': 'BUY',
                'price': current_price,
                'strength': min(buy_score, 10),  # Cap at 10
                'macd': macd_signals,
                'rsi': rsi_signals['current'],
                'volume_ratio': volume_signals['volume_ratio'],
                'pattern': candle_patterns.get('pattern_name'),
                'ema': ema_signals['trend'] if ema_signals else None,
                'market_structure': market_structure['short_term_trend'],
                'support': sr_levels.get('closest_support'),
                'resistance': sr_levels.get('closest_resistance'),
                'reason': " | ".join(buy_reasons)
            }
            signals.append(signal)

        if sell_score > threshold:
            signal = {
                'signal_type': 'SELL',
                'price': current_price,
                'strength': min(sell_score, 10),  # Cap at 10
                'macd': macd_signals,
                'rsi': rsi_signals['current'],
                'volume_ratio': volume_signals['volume_ratio'],
                'pattern': candle_patterns.get('pattern_name'),
                'ema': ema_signals['trend'] if ema_signals else None,
                'market_structure': market_structure['short_term_trend'],
                'support': sr_levels.get('closest_support'),
                'resistance': sr_levels.get('closest_resistance'),
                'reason': " | ".join(sell_reasons)
            }
            signals.append(signal)

        return signals

    def _analyze_candle_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze candle patterns"""
        patterns = {
            'bullish': False,
            'bearish': False,
            'strength': 0,
            'pattern_name': None
        }

        # Get last 3 candles
        last_candles = df.iloc[-3:]

        # Check bullish candle pattern
        if self._is_bullish_engulfing(last_candles):
            patterns['bullish'] = True
            patterns['strength'] = 8
            patterns['pattern_name'] = 'Bullish Engulfing'

        elif self._is_morning_star(last_candles):
            patterns['bullish'] = True
            patterns['strength'] = 9
            patterns['pattern_name'] = 'Morning Star'

        # Kiểm tra mô hình nến giảm
        elif self._is_bearish_engulfing(last_candles):
            patterns['bearish'] = True
            patterns['strength'] = 8
            patterns['pattern_name'] = 'Bearish Engulfing'

        elif self._is_evening_star(last_candles):
            patterns['bearish'] = True
            patterns['strength'] = 9
            patterns['pattern_name'] = 'Evening Star'

        return patterns

    def _analyze_macd(self, df: pd.DataFrame) -> Dict:
        """Analyze MACD signals"""
        macd = df['macd'].iloc[-1]
        signal = df['macd_signal'].iloc[-1]
        hist = df['macd_hist'].iloc[-1]
        prev_hist = df['macd_hist'].iloc[-2]

        signals = {
            'buy_signal': False,
            'sell_signal': False,
            'macd_direction': 'NEUTRAL',
            'histogram_direction': 'NEUTRAL',
            'strength': 0
        }

        # Xác định hướng MACD
        if macd > signal:
            signals['macd_direction'] = 'BULLISH'
        elif macd < signal:
            signals['macd_direction'] = 'BEARISH'

        # Xác định hướng histogram
        if hist > prev_hist:
            signals['histogram_direction'] = 'BULLISH'
        elif hist < prev_hist:
            signals['histogram_direction'] = 'BEARISH'

        # Tín hiệu mua
        if (hist > 0 and prev_hist < 0) or (macd > signal and prev_hist < hist):
            signals['buy_signal'] = True
            signals['strength'] = min(abs(hist/signal) * 10, 10)

        # Tín hiệu bán
        elif (hist < 0 and prev_hist > 0) or (macd < signal and prev_hist > hist):
            signals['sell_signal'] = True
            signals['strength'] = min(abs(hist/signal) * 10, 10)

        return signals

    def _analyze_rsi(self, df: pd.DataFrame) -> Dict:
        """Analyze RSI signals"""
        current_rsi = df['rsi'].iloc[-1]
        prev_rsi = df['rsi'].iloc[-2]

        signals = {
            'current': current_rsi,
            'previous': prev_rsi,
            'overbought': current_rsi > self.rsi_overbought,
            'oversold': current_rsi < self.rsi_oversold,
            'strength': 0
        }

        # Tính độ mạnh của tín hiệu
        if signals['oversold']:
            signals['strength'] = (
                (self.rsi_oversold - current_rsi) / self.rsi_oversold) * 10
        elif signals['overbought']:
            signals['strength'] = (
                (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)) * 10

        return signals

    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze trading volume"""
        current_volume = df['volume'].iloc[-1]
        volume_ma = df['volume_ma'].iloc[-1]
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 0

        prev_close = df['close'].iloc[-2]
        current_close = df['close'].iloc[-1]
        price_change = (current_close - prev_close) / prev_close

        signals = {
            'volume_ratio': volume_ratio,
            'above_average': volume_ratio > self.min_volume_ratio,
            'bullish_volume': price_change > 0 and volume_ratio > self.min_volume_ratio,
            'bearish_volume': price_change < 0 and volume_ratio > self.min_volume_ratio,
            'strength': min(volume_ratio * 2, 10)  # Scale 0-10
        }

        return signals

    def _analyze_ema(self, df: pd.DataFrame) -> Dict:
        """Analyze EMA signals"""
        # Get current values
        current_close = df['close'].iloc[-1]
        ema_short = df['ema34'].iloc[-1]
        ema_long = df['ema89'].iloc[-1]

        # Get previous values for slope direction
        prev_ema_short = df['ema34'].iloc[-2] if len(df) > 2 else ema_short
        prev_ema_long = df['ema89'].iloc[-2] if len(df) > 2 else ema_long

        # Check for EMA crossover
        is_golden_cross = ema_short > ema_long and df['ema34'].iloc[-2] <= df['ema89'].iloc[-2]
        is_death_cross = ema_short < ema_long and df['ema34'].iloc[-2] >= df['ema89'].iloc[-2]

        # Calculate how recently the crossover happened (look back up to 5 candles)
        cross_candles_ago = 0
        if not is_golden_cross and not is_death_cross:
            for i in range(2, min(7, len(df))):
                if (df['ema34'].iloc[-i] > df['ema89'].iloc[-i] and df['ema34'].iloc[-i-1] <= df['ema89'].iloc[-i-1]) or \
                   (df['ema34'].iloc[-i] < df['ema89'].iloc[-i] and df['ema34'].iloc[-i-1] >= df['ema89'].iloc[-i-1]):
                    cross_candles_ago = i
                    break

        # Determine trend direction
        trend = "BULLISH" if ema_short > ema_long else "BEARISH" if ema_short < ema_long else "NEUTRAL"

        # Calculate price position relative to EMAs
        price_above_short = current_close > ema_short
        price_above_long = current_close > ema_long
        price_between_emas = (current_close > ema_short and current_close < ema_long) or \
            (current_close < ema_short and current_close > ema_long)

        # Check for bullish/bearish alignment
        bullish_alignment = price_above_short and price_above_long and ema_short > ema_long and ema_short > prev_ema_short
        bearish_alignment = current_close < ema_short and current_close < ema_long and ema_short < ema_long and ema_short < prev_ema_short

        # Check for pullback to EMA (potential entry points)
        pullback_to_ema34 = (trend == "BULLISH" and abs(
            current_close - ema_short) / ema_short < 0.005)
        pullback_to_ema89 = (trend == "BULLISH" and abs(
            current_close - ema_long) / ema_long < 0.005)

        # Calculate signal strength (0-10)
        strength = 0

        if is_golden_cross:
            strength += 10  # Fresh bullish crossover is very strong
        elif is_death_cross:
            strength += 8   # Fresh bearish crossover
        elif cross_candles_ago > 0 and cross_candles_ago <= 3:
            # Recent crossover (within 3 candles)
            strength += (5 - cross_candles_ago)

        if bullish_alignment:
            strength += 5
        elif bearish_alignment:
            strength += 5

        if pullback_to_ema34 or pullback_to_ema89:
            strength += 3  # Potential entry point

        signals = {
            'trend': trend,
            'ema_short': ema_short,
            'ema_long': ema_long,
            'ema_short_slope': "UP" if ema_short > prev_ema_short else "DOWN" if ema_short < prev_ema_short else "FLAT",
            'ema_long_slope': "UP" if ema_long > prev_ema_long else "DOWN" if ema_long < prev_ema_long else "FLAT",
            'price_above_short': price_above_short,
            'price_above_long': price_above_long,
            'price_between_emas': price_between_emas,
            'is_golden_cross': is_golden_cross,
            'is_death_cross': is_death_cross,
            'cross_candles_ago': cross_candles_ago,
            'bullish_alignment': bullish_alignment,
            'bearish_alignment': bearish_alignment,
            'pullback_to_ema34': pullback_to_ema34,
            'pullback_to_ema89': pullback_to_ema89,
            'buy_signal': is_golden_cross or (trend == "BULLISH" and (pullback_to_ema34 or pullback_to_ema89)),
            'sell_signal': is_death_cross or (trend == "BEARISH" and price_above_short),
            'strength': min(strength, 10)  # Cap at 10
        }

        return signals

    def _is_bullish_engulfing(self, candles: pd.DataFrame) -> bool:
        """Kiểm tra mô hình nến bao trùm tăng"""
        if len(candles) < 2:
            return False

        prev_candle = candles.iloc[-2]
        curr_candle = candles.iloc[-1]

        return (prev_candle['close'] < prev_candle['open'] and
                curr_candle['close'] > curr_candle['open'] and
                # Mở cửa thấp hơn
                curr_candle['open'] < prev_candle['close'] and
                # Đóng cửa cao hơn
                curr_candle['close'] > prev_candle['open'])

    def _is_bearish_engulfing(self, candles: pd.DataFrame) -> bool:
        """Kiểm tra mô hình nến bao trùm giảm"""
        if len(candles) < 2:
            return False

        prev_candle = candles.iloc[-2]
        curr_candle = candles.iloc[-1]

        return (prev_candle['close'] > prev_candle['open'] and
                curr_candle['close'] < curr_candle['open'] and
                # Mở cửa cao hơn
                curr_candle['open'] > prev_candle['close'] and
                # Đóng cửa thấp hơn
                curr_candle['close'] < prev_candle['open'])

    def _is_morning_star(self, candles: pd.DataFrame) -> bool:
        """Kiểm tra mô hình sao mai"""
        if len(candles) < 3:
            return False

        first = candles.iloc[-3]
        second = candles.iloc[-2]
        third = candles.iloc[-1]

        return (first['close'] < first['open'] and

                abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and

                third['close'] > third['open'] and
                # Tăng qua giữa nến 1
                third['close'] > (first['open'] + first['close']) / 2)

    def _is_evening_star(self, candles: pd.DataFrame) -> bool:
        """Kiểm tra mô hình sao hôm"""
        if len(candles) < 3:
            return False

        first = candles.iloc[-3]
        second = candles.iloc[-2]
        third = candles.iloc[-1]

        return (first['close'] > first['open'] and

                abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and

                third['close'] < third['open'] and
                # Giảm qua giữa nến 1
                third['close'] < (first['open'] + first['close']) / 2)

    def analyze_multi_timeframe(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze market across multiple timeframes to find high-probability entries.

        Parameters:
        -----------
        data_dict: Dict of DataFrames containing price data for different timeframes
                   Format: {'1h': df_1h, '4h': df_4h}

        Returns:
        --------
        Dict containing multi-timeframe analysis results
        """
        mtf_analysis = {
            'aligned_trend': False,
            'reversal_signals': [],
            'htf_support_resistance': {},
            'best_entries': []
        }

        # Get individual analyses for each timeframe
        analyses = {}
        for timeframe, df in data_dict.items():
            if len(df) >= 100:  # Ensure enough data
                analyses[timeframe] = self.analyze_market(df)

        # Return early if we don't have enough timeframes to analyze
        if len(analyses) < 2:
            return mtf_analysis

        # Check for trend alignment across timeframes
        ltf_trend = analyses.get(min(analyses.keys()), {}).get(
            'market_structure', {}).get('short_term_trend')
        htf_trend = analyses.get(max(analyses.keys()), {}).get(
            'market_structure', {}).get('medium_term_trend')

        if ltf_trend and htf_trend and ltf_trend == htf_trend:
            mtf_analysis['aligned_trend'] = True
            mtf_analysis['aligned_direction'] = ltf_trend

        # Detect reversals on higher timeframes
        htf_key = max(analyses.keys())
        htf_analysis = analyses.get(htf_key, {})
        htf_candle_patterns = htf_analysis.get(
            'indicators', {}).get('patterns', {})
        htf_divergence = htf_analysis.get(
            'indicators', {}).get('divergence', {})

        # Check for reversal patterns on higher timeframe
        if htf_candle_patterns.get('bullish') or htf_divergence.get('bullish_rsi_divergence'):
            mtf_analysis['reversal_signals'].append({
                'timeframe': htf_key,
                'type': 'BULLISH',
                'pattern': htf_candle_patterns.get('pattern_name') if htf_candle_patterns.get('bullish') else 'RSI Divergence',
                'price': htf_analysis.get('current_price', 0)
            })

        if htf_candle_patterns.get('bearish') or htf_divergence.get('bearish_rsi_divergence'):
            mtf_analysis['reversal_signals'].append({
                'timeframe': htf_key,
                'type': 'BEARISH',
                'pattern': htf_candle_patterns.get('pattern_name') if htf_candle_patterns.get('bearish') else 'RSI Divergence',
                'price': htf_analysis.get('current_price', 0)
            })

        # Extract support/resistance from higher timeframe
        htf_sr = htf_analysis.get('sr_levels', {})
        mtf_analysis['htf_support_resistance'] = htf_sr

        # Find high-probability entry points (confluence of factors)
        ltf_key = min(analyses.keys())
        ltf_analysis = analyses.get(ltf_key, {})
        ltf_signals = ltf_analysis.get('signals', [])

        for signal in ltf_signals:
            entry_quality = 0
            reasons = []

            # Base quality from the signal itself
            entry_quality += signal.get('strength', 0) * 0.5

            # Boost if trend is aligned across timeframes
            if mtf_analysis['aligned_trend']:
                if (signal['signal_type'] == 'BUY' and mtf_analysis['aligned_direction'] == 'BULLISH') or \
                   (signal['signal_type'] == 'SELL' and mtf_analysis['aligned_direction'] == 'BEARISH'):
                    entry_quality += 2
                    reasons.append(
                        f"Aligned {mtf_analysis['aligned_direction']} trend across timeframes")

            # Boost if signal is near HTF support/resistance
            if signal['signal_type'] == 'BUY' and htf_sr.get('near_support'):
                entry_quality += 3
                reasons.append(
                    f"Price near higher timeframe support ({htf_sr.get('closest_support', 0):.2f})")

            if signal['signal_type'] == 'SELL' and htf_sr.get('near_resistance'):
                entry_quality += 3
                reasons.append(
                    f"Price near higher timeframe resistance ({htf_sr.get('closest_resistance', 0):.2f})")

            # Boost if reversal on higher timeframe aligns with signal
            for reversal in mtf_analysis['reversal_signals']:
                if (signal['signal_type'] == 'BUY' and reversal['type'] == 'BULLISH') or \
                   (signal['signal_type'] == 'SELL' and reversal['type'] == 'BEARISH'):
                    entry_quality += 4
                    reasons.append(
                        f"Confirmed by {reversal['type']} reversal on {reversal['timeframe']} ({reversal['pattern']})")

            # Add to best entries if quality is sufficient
            if entry_quality >= 5:
                entry = signal.copy()
                entry['mtf_quality'] = entry_quality
                entry['mtf_reasons'] = reasons
                mtf_analysis['best_entries'].append(entry)

        # Sort entries by quality
        mtf_analysis['best_entries'].sort(
            key=lambda x: x.get('mtf_quality', 0), reverse=True)

        return mtf_analysis

    def detect_timeframe_reversals(self, df: pd.DataFrame) -> Dict:
        """
        Detect potential market reversals using multiple methods.

        Parameters:
        -----------
        df: DataFrame with price data

        Returns:
        --------
        Dict with reversal signals
        """
        reversals = {
            'bullish_reversals': [],
            'bearish_reversals': [],
            'strength': 0
        }

        # Get important indicators
        current_price = df['close'].iloc[-1]
        delta = df['close'].diff()
        delta = delta.fillna(0)  # Fill NaN values to avoid calculation errors

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Use simple calculation for first periods to avoid NaN values
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()

        # Avoid division by zero
        avg_loss = avg_loss.replace(0, 0.000001)

        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        rsi = df['rsi'].iloc[-1]

        # 1. Check for oversold/overbought conditions with RSI
        if rsi is not None:
            if rsi < 30:
                reversals['bullish_reversals'].append({
                    'type': 'RSI_OVERSOLD',
                    'value': rsi,
                    'price': current_price,
                    'strength': 3
                })
            elif rsi > 70:
                reversals['bearish_reversals'].append({
                    'type': 'RSI_OVERBOUGHT',
                    'value': rsi,
                    'price': current_price,
                    'strength': 3
                })

        # 2. Check for candlestick reversal patterns
        patterns = self._analyze_candle_patterns(df)
        if patterns['bullish']:
            reversals['bullish_reversals'].append({
                'type': 'CANDLESTICK_PATTERN',
                'pattern': patterns['pattern_name'],
                'price': current_price,
                'strength': patterns['strength']
            })
        elif patterns['bearish']:
            reversals['bearish_reversals'].append({
                'type': 'CANDLESTICK_PATTERN',
                'pattern': patterns['pattern_name'],
                'price': current_price,
                'strength': patterns['strength']
            })

        # 3. Check for divergences
        divergence = self._analyze_divergence(df)
        if divergence['bullish_rsi_divergence']:
            reversals['bullish_reversals'].append({
                'type': 'RSI_DIVERGENCE',
                'price': current_price,
                'strength': 5  # Divergences are strong signals
            })
        if divergence['bearish_rsi_divergence']:
            reversals['bearish_reversals'].append({
                'type': 'RSI_DIVERGENCE',
                'price': current_price,
                'strength': 5  # Divergences are strong signals
            })

        # 4. Check for EMA crossovers
        ema_signals = self._analyze_ema(df)
        if ema_signals['is_golden_cross']:
            reversals['bullish_reversals'].append({
                'type': 'EMA_CROSS',
                'price': current_price,
                'strength': 4,
                'description': 'Golden Cross (EMA34 above EMA89)'
            })
        elif ema_signals['is_death_cross']:
            reversals['bearish_reversals'].append({
                'type': 'EMA_CROSS',
                'price': current_price,
                'strength': 4,
                'description': 'Death Cross (EMA34 below EMA89)'
            })

        # 5. Check for price rejection at key levels
        sr_levels = self._analyze_support_resistance(df)
        if sr_levels['near_support'] and df['low'].iloc[-1] < sr_levels['closest_support'] and df['close'].iloc[-1] > sr_levels['closest_support']:
            # Price rejected from support (bullish)
            reversals['bullish_reversals'].append({
                'type': 'SUPPORT_REJECTION',
                'level': sr_levels['closest_support'],
                'price': current_price,
                'strength': 4
            })
        if sr_levels['near_resistance'] and df['high'].iloc[-1] > sr_levels['closest_resistance'] and df['close'].iloc[-1] < sr_levels['closest_resistance']:
            # Price rejected from resistance (bearish)
            reversals['bearish_reversals'].append({
                'type': 'RESISTANCE_REJECTION',
                'level': sr_levels['closest_resistance'],
                'price': current_price,
                'strength': 4
            })

        # Calculate overall reversal strength
        bull_strength = sum(r['strength']
                            for r in reversals['bullish_reversals'])
        bear_strength = sum(r['strength']
                            for r in reversals['bearish_reversals'])

        if bull_strength > bear_strength:
            reversals['direction'] = 'BULLISH'
            reversals['strength'] = bull_strength
        elif bear_strength > bull_strength:
            reversals['direction'] = 'BEARISH'
            reversals['strength'] = bear_strength
        else:
            reversals['direction'] = 'NEUTRAL'
            reversals['strength'] = 0

        return reversals
