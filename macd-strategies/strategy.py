import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from binance.client import Client
from binance.exceptions import BinanceAPIException

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
                 min_volume_ratio: float = 2.0):
        """
        Khởi tạo chiến lược giao dịch dựa trên MACD và RSI

        Parameters:
        -----------
        rsi_period: Chu kỳ tính RSI
        rsi_overbought: Ngưỡng quá mua của RSI
        rsi_oversold: Ngưỡng quá bán của RSI
        macd_fast: EMA nhanh cho MACD
        macd_slow: EMA chậm cho MACD
        macd_signal: Chu kỳ đường tín hiệu MACD
        volume_ma_period: Chu kỳ MA của khối lượng
        min_volume_ratio: Tỷ lệ khối lượng tối thiểu so với trung bình
        """
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.volume_ma_period = volume_ma_period
        self.min_volume_ratio = min_volume_ratio

    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """
        Phân tích thị trường và tạo tín hiệu giao dịch
        """
        analysis = {}

        # Tính toán các chỉ báo
        df = self._calculate_indicators(data)

        # Phân tích mô hình nến
        candle_patterns = self._analyze_candle_patterns(df)

        # Phân tích MACD
        macd_signals = self._analyze_macd(df)

        # Phân tích RSI
        rsi_signals = self._analyze_rsi(df)

        # Phân tích khối lượng
        volume_signals = self._analyze_volume(df)

        # Kết hợp tín hiệu
        signals = self._combine_signals(
            df,
            candle_patterns,
            macd_signals,
            rsi_signals,
            volume_signals
        )

        analysis['signals'] = signals
        analysis['indicators'] = {
            'macd': macd_signals,
            'rsi': rsi_signals,
            'volume': volume_signals,
            'patterns': candle_patterns
        }
        analysis['current_price'] = df['close'].iloc[-1]

        return analysis

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính toán các chỉ báo kỹ thuật"""
        df = df.copy()

        # Tính RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(
            window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)
                ).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Tính MACD
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(
            span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Tính khối lượng trung bình
        df['volume_ma'] = df['volume'].rolling(
            window=self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # Tính Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        return df

    def _analyze_candle_patterns(self, df: pd.DataFrame) -> Dict:
        """Phân tích mô hình nến"""
        patterns = {
            'bullish': False,
            'bearish': False,
            'strength': 0,
            'pattern_name': None
        }

        # Lấy 3 cây nến gần nhất
        last_candles = df.iloc[-3:]

        # Kiểm tra mô hình nến tăng
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
        """Phân tích tín hiệu MACD"""
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
        """Phân tích tín hiệu RSI"""
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
        """Phân tích khối lượng giao dịch"""
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
                         volume_signals: Dict) -> List[Dict]:
        """Kết hợp các tín hiệu để đưa ra quyết định giao dịch"""
        signals = []
        current_price = df['close'].iloc[-1]

        # Tính điểm cho tín hiệu mua
        buy_score = 0
        if macd_signals['buy_signal']:
            buy_score += macd_signals['strength'] * 0.3
        if rsi_signals['oversold']:
            buy_score += rsi_signals['strength'] * 0.3
        if candle_patterns['bullish']:
            buy_score += candle_patterns['strength'] * 0.2
        if volume_signals['bullish_volume']:
            buy_score += volume_signals['strength'] * 0.2

        # Tính điểm cho tín hiệu bán
        sell_score = 0
        if macd_signals['sell_signal']:
            sell_score += macd_signals['strength'] * 0.3
        if rsi_signals['overbought']:
            sell_score += rsi_signals['strength'] * 0.3
        if candle_patterns['bearish']:
            sell_score += candle_patterns['strength'] * 0.2
        if volume_signals['bearish_volume']:
            sell_score += volume_signals['strength'] * 0.2

        # Tạo tín hiệu nếu đạt ngưỡng
        threshold = 6.0  # Ngưỡng điểm tối thiểu

        if buy_score > threshold:
            signals.append({
                'signal_type': 'BUY',
                'price': current_price,
                'strength': buy_score,
                'macd': macd_signals,
                'rsi': rsi_signals['current'],
                'volume_ratio': volume_signals['volume_ratio'],
                'pattern': candle_patterns.get('pattern_name'),
                'reason': self._generate_signal_reason(
                    'BUY',
                    macd_signals,
                    rsi_signals,
                    candle_patterns,
                    volume_signals
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
                'reason': self._generate_signal_reason(
                    'SELL',
                    macd_signals,
                    rsi_signals,
                    candle_patterns,
                    volume_signals
                )
            })

        return signals

    def _generate_signal_reason(self, signal_type: str, macd: Dict,
                                rsi: Dict, patterns: Dict, volume: Dict) -> str:
        """Tạo lý do cho tín hiệu giao dịch"""
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

        return " | ".join(reasons)

    def _is_bullish_engulfing(self, candles: pd.DataFrame) -> bool:
        """Kiểm tra mô hình nến bao trùm tăng"""
        if len(candles) < 2:
            return False

        prev_candle = candles.iloc[-2]
        curr_candle = candles.iloc[-1]

        return (prev_candle['close'] < prev_candle['open'] and  # Nến giảm
                curr_candle['close'] > curr_candle['open'] and  # Nến tăng
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

        return (prev_candle['close'] > prev_candle['open'] and  # Nến tăng
                curr_candle['close'] < curr_candle['open'] and  # Nến giảm
                # Mở cửa cao hơn
                curr_candle['open'] > prev_candle['close'] and
                # Đóng cửa thấp hơn
                curr_candle['close'] < prev_candle['open'])

    def _is_morning_star(self, candles: pd.DataFrame) -> bool:
        """Kiểm tra mô hình sao mai"""
        if len(candles) < 3:
            return False

        first = candles.iloc[-3]   # Nến giảm
        second = candles.iloc[-2]  # Nến nhỏ
        third = candles.iloc[-1]   # Nến tăng

        return (first['close'] < first['open'] and                     # Nến giảm
                # Nến nhỏ
                abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and
                # Nến tăng
                third['close'] > third['open'] and
                # Tăng qua giữa nến 1
                third['close'] > (first['open'] + first['close']) / 2)

    def _is_evening_star(self, candles: pd.DataFrame) -> bool:
        """Kiểm tra mô hình sao hôm"""
        if len(candles) < 3:
            return False

        first = candles.iloc[-3]   # Nến tăng
        second = candles.iloc[-2]  # Nến nhỏ
        third = candles.iloc[-1]   # Nến giảm

        return (first['close'] > first['open'] and                     # Nến tăng
                # Nến nhỏ
                abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and
                # Nến giảm
                third['close'] < third['open'] and
                # Giảm qua giữa nến 1
                third['close'] < (first['open'] + first['close']) / 2)
