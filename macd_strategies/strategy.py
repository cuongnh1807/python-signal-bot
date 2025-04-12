import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List
import os


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
        Analyze market and create trading signals
        """
        analysis = {}

        # Calculate indicators
        df = self._calculate_indicators(data)

        # Analyze candle patterns
        candle_patterns = self._analyze_candle_patterns(df)

        # Analyze MACD
        macd_signals = self._analyze_macd(df)

        # Analyze RSI
        rsi_signals = self._analyze_rsi(df)

        # Analyze volume
        volume_signals = self._analyze_volume(df)

        # Analyze EMA
        ema_signals = self._analyze_ema(df)

        # Combine signals
        signals = self._combine_signals(
            df,
            candle_patterns,
            macd_signals,
            rsi_signals,
            volume_signals,
            ema_signals
        )

        print("signals", signals)

        analysis['signals'] = signals
        analysis['indicators'] = {
            'macd': macd_signals,
            'rsi': rsi_signals,
            'volume': volume_signals,
            'patterns': candle_patterns,
            'ema': ema_signals
        }
        analysis['current_price'] = df['close'].iloc[-1]

        return analysis

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = df.copy()

        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(
            window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)
                ).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Calculate MACD
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(
            span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Calculate EMA34 and EMA89
        df['ema34'] = df['close'].ewm(span=self.ema_short, adjust=False).mean()
        df['ema89'] = df['close'].ewm(span=self.ema_long, adjust=False).mean()

        # Calculate volume MA
        df['volume_ma'] = df['volume'].rolling(
            window=self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        return df

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

    def _combine_signals(self, df: pd.DataFrame, candle_patterns: Dict,
                         macd_signals: Dict, rsi_signals: Dict,
                         volume_signals: Dict, ema_signals: Dict = None) -> List[Dict]:
        """Combine signals to make trading decisions"""
        signals = []
        current_price = df['close'].iloc[-1]

        # Calculate buy signal score
        buy_score = 0
        if macd_signals['buy_signal']:
            buy_score += macd_signals['strength'] * 0.25
        if rsi_signals['oversold']:
            buy_score += rsi_signals['strength'] * 0.2
        if candle_patterns['bullish']:
            buy_score += candle_patterns['strength'] * 0.15
        if volume_signals['bullish_volume']:
            buy_score += volume_signals['strength'] * 0.15

        # Add EMA signals to the score calculation
        if ema_signals:
            if ema_signals['buy_signal']:
                buy_score += ema_signals['strength'] * 0.25
            # Boost score if price is in ideal buy zone (pullback to EMA in bullish trend)
            if ema_signals['trend'] == 'BULLISH' and (ema_signals['pullback_to_ema34'] or ema_signals['pullback_to_ema89']):
                buy_score += 2.0

        # Calculate sell signal score
        sell_score = 0
        if macd_signals['sell_signal']:
            sell_score += macd_signals['strength'] * 0.25
        if rsi_signals['overbought']:
            sell_score += rsi_signals['strength'] * 0.2
        if candle_patterns['bearish']:
            sell_score += candle_patterns['strength'] * 0.15
        if volume_signals['bearish_volume']:
            sell_score += volume_signals['strength'] * 0.15

        # Add EMA signals to the sell score
        if ema_signals:
            if ema_signals['sell_signal']:
                sell_score += ema_signals['strength'] * 0.25
            # Boost sell score in strong bearish alignment
            if ema_signals['bearish_alignment']:
                sell_score += 2.0

        # Create signal if threshold is met
        threshold = 6.0  # Minimum score threshold

        if buy_score > threshold:
            signals.append({
                'signal_type': 'BUY',
                'price': current_price,
                'strength': buy_score,
                'macd': macd_signals,
                'rsi': rsi_signals['current'],
                'volume_ratio': volume_signals['volume_ratio'],
                'pattern': candle_patterns.get('pattern_name'),
                'ema': ema_signals['trend'] if ema_signals else None,
                'reason': self._generate_signal_reason(
                    'BUY',
                    macd_signals,
                    rsi_signals,
                    candle_patterns,
                    volume_signals,
                    ema_signals
                )
            })

        if sell_score > threshold:
            signals.append({
                'signal_type': 'SELL',
                'price': current_price,
                'strength': sell_score,
                'macd': macd_signals,
                'rsi': rsi_signals['current'],
                'volume_ratio': volume_signals['volume_ratio'],
                'pattern': candle_patterns.get('pattern_name'),
                'ema': ema_signals['trend'] if ema_signals else None,
                'reason': self._generate_signal_reason(
                    'SELL',
                    macd_signals,
                    rsi_signals,
                    candle_patterns,
                    volume_signals,
                    ema_signals
                )
            })

        return signals

    def _generate_signal_reason(self, signal_type: str, macd: Dict,
                                rsi: Dict, patterns: Dict, volume: Dict,
                                ema: Dict = None) -> str:
        """Generate trading signal reason"""
        reasons = []

        if signal_type == 'BUY':
            if macd['buy_signal']:
                reasons.append(
                    f"MACD cross up (strength: {macd['strength']:.1f})")
            if rsi['oversold']:
                reasons.append(f"RSI oversold at {rsi['current']:.1f}")
            if patterns['bullish']:
                reasons.append(f"Bullish {patterns['pattern_name']}")
            if volume['bullish_volume']:
                reasons.append(
                    f"High volume ({volume['volume_ratio']:.1f}x avg)")
            # Add EMA reasons
            if ema and ema['buy_signal']:
                if ema['is_golden_cross']:
                    reasons.append(f"Golden Cross (EMA34 above EMA89)")
                if ema['pullback_to_ema34']:
                    reasons.append(f"Price pullback to EMA34 support")
                if ema['pullback_to_ema89']:
                    reasons.append(f"Price pullback to EMA89 support")
                if ema['bullish_alignment']:
                    reasons.append(f"Strong bullish trend alignment")

        else:  # SELL
            if macd['sell_signal']:
                reasons.append(
                    f"MACD cross down (strength: {macd['strength']:.1f})")
            if rsi['overbought']:
                reasons.append(f"RSI overbought at {rsi['current']:.1f}")
            if patterns['bearish']:
                reasons.append(f"Bearish {patterns['pattern_name']}")
            if volume['bearish_volume']:
                reasons.append(
                    f"High volume ({volume['volume_ratio']:.1f}x avg)")
            # Add EMA reasons
            if ema and ema['sell_signal']:
                if ema['is_death_cross']:
                    reasons.append(f"Death Cross (EMA34 below EMA89)")
                if not ema['price_above_short'] and ema['trend'] == 'BEARISH':
                    reasons.append(f"Price below EMA34 in downtrend")
                if ema['bearish_alignment']:
                    reasons.append(f"Strong bearish trend alignment")

        return " | ".join(reasons)

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
