from smartmoneyconcepts.smc import smc
from datetime import datetime, timedelta
import pandas as pd
from typing import Union
from helpers.price import calculate_price_momentum


setup_classification = {
    'LONG_BOS': 'Break of Structure Long - Strong counter-trend reversal with high volume climax',
    'LONG_CHoCH': 'Change of Character Long - Potential reversal with accumulation phase',
    'LONG_CONTINUATION': 'Strong Continuation Long - Trend continuation with high volume momentum',
    'LONG_PULLBACK': 'Pullback Long - Healthy retracement in uptrend with moderate volume',
    'LONG_POTENTIAL': 'Potential Long - Setup needs volume confirmation',
    'LONG_WEAK': 'Weak Long Setup - Low volume or unclear momentum',

    'SHORT_BOS': 'Break of Structure Short - Strong counter-trend reversal with high volume climax',
    'SHORT_CHoCH': 'Change of Character Short - Potential reversal with distribution phase',
    'SHORT_CONTINUATION': 'Strong Continuation Short - Trend continuation with high volume momentum',
    'SHORT_PULLBACK': 'Pullback Short - Healthy retracement in downtrend with moderate volume',
    'SHORT_POTENTIAL': 'Potential Short - Setup needs volume confirmation',
    'SHORT_WEAK': 'Weak Short Setup - Low volume or unclear momentum'
}


def calculate_rsi(data: Union[pd.DataFrame, pd.Series], rsi_length: int = 14, ma_type: str = "SMA") -> pd.Series:
    """
    Calculate RSI (Relative Strength Index)

    Parameters:
    - data: DataFrame with OHLCV data or Series of values
    - periods: RSI period (default 14)

    Returns:
    - RSI values as Series
    """
    # Handle both DataFrame and Series inputs
    if isinstance(data, pd.DataFrame):
        series = data['close']
    else:
        series = data

    delta = series.diff()
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the smoothed average gains and losses
    if ma_type == "SMA":
        avg_gain = gain.rolling(window=rsi_length).mean()
        avg_loss = loss.rolling(window=rsi_length).mean()
    elif ma_type == "EMA":
        avg_gain = gain.ewm(span=rsi_length, adjust=False).mean()
        avg_loss = loss.ewm(span=rsi_length, adjust=False).mean()
    else:
        raise ValueError("Invalid moving average type. Use 'SMA' or 'EMA'.")
    for i in range(rsi_length, len(data)):

        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] *
                            (rsi_length - 1) + gain.iloc[i]) / rsi_length

        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] *
                            (rsi_length - 1) + loss.iloc[i]) / rsi_length

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
    """Calculate the MACD and Signal line."""
    exp1 = data['close'].ewm(span=short_window, adjust=False).mean()
    exp2 = data['close'].ewm(span=long_window, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    histogram = macd - signal

    return pd.DataFrame({'macd': macd, 'signal': signal, 'histogram': histogram})


def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals based on RSI and MACD crossovers"""
    # Calculate RSI
    data['rsi'] = calculate_rsi(data)

    # Calculate MACD
    macd_df = calculate_macd(data)
    data['macd'] = macd_df['macd']
    data['macd_signal'] = macd_df['signal']

    # Calculate previous values for crossover detection
    data['prev_macd'] = data['macd'].shift(1)
    data['prev_signal'] = data['macd_signal'].shift(1)

    # Initialize signal column
    data['signal'] = 0

    # Generate signals based on your TradingView logic
    for i in range(1, len(data)):
        # MACD Crossover detection
        macd_crossover = (data['prev_macd'].iloc[i] <= data['prev_signal'].iloc[i] and
                          data['macd'].iloc[i] > data['macd_signal'].iloc[i])

        macd_crossunder = (data['prev_macd'].iloc[i] >= data['prev_signal'].iloc[i] and
                           data['macd'].iloc[i] < data['macd_signal'].iloc[i])

        # Buy conditions
        if macd_crossover and data['rsi'].iloc[i] < 40:
            data.loc[data.index[i], 'signal'] = 1  # Buy signal

        # Sell conditions
        elif macd_crossunder and data['rsi'].iloc[i] > 60:
            data.loc[data.index[i], 'signal'] = -1  # Sell signal

        # Short entry conditions
        elif macd_crossunder and data['rsi'].iloc[i] < 40:
            data.loc[data.index[i], 'signal'] = -2  # Short entry signal

        # Short exit conditions
        elif macd_crossover and data['rsi'].iloc[i] > 75:
            data.loc[data.index[i], 'signal'] = 2  # Short exit signal

    # Clean up temporary columns
    data = data.drop(['prev_macd', 'prev_signal'], axis=1)

    return data


def find_closest_signal(data: pd.DataFrame, current_price: float, limit: int = 5, loopback: int = 10) -> dict:
    """Find the 5 closest signals and determine safety rating based on RSI and momentum"""
    # Filter signals
    # m = generate_signals(data)
    signals = data[data['signal'] != 0].copy()

    if signals.empty:
        return {
            "signals": [],
            "trend": "No signals available"
        }

    # Calculate price distance and momentum
    signals.loc[:, 'price_distance'] = (
        signals['close'] - current_price).abs() / current_price * 100
    signals['momentum'] = data['close'].pct_change(
        periods=loopback) * 100  # 3-period momentum

    # Get 5 closest signals
    closest_signals = signals.nsmallest(limit, 'price_distance')

    signal_details = []
    for _, signal in closest_signals.iterrows():
        # Calculate MACD histogram
        macd_hist = signal['macd'] - signal['macd_signal']

        # Calculate safety factors
        safety_factors = {
            'rsi_safety': {
                'weight': 0.4,
                'score': calculate_rsi_safety_score(signal['rsi'], signal['signal'])
            },
            'momentum_safety': {
                'weight': 0.4,
                'score': calculate_momentum_safety_score(
                    signal['momentum'],
                    signal['signal'],
                    signal['macd'],
                    signal['macd_signal'],
                    macd_hist
                )
            },
            'price_distance': {
                'weight': 0.2,
                'score': max(0, 100 - (signal['price_distance'] * 10))
            }
        }

        # Calculate total safety score
        safety_score = sum(factor['weight'] * factor['score']
                           for factor in safety_factors.values())

        # Determine signal type
        signal_type = {
            1: "LONG_ENTRY",
            -1: "LONG_EXIT",
            -2: "SHORT_ENTRY",
            2: "SHORT_EXIT"
        }.get(signal['signal'], "UNKNOWN")

        # Determine safety rating
        if safety_score >= 80:
            safety_rating = "VERY_SAFE"
            safety_emoji = "🟢"
        elif safety_score >= 60:
            safety_rating = "SAFE"
            safety_emoji = "🟡"
        elif safety_score >= 40:
            safety_rating = "MODERATE"
            safety_emoji = "🟠"
        elif safety_score >= 20:
            safety_rating = "RISKY"
            safety_emoji = "🔴"
        else:
            safety_rating = "DANGEROUS"
            safety_emoji = "⛔"

        signal_details.append({
            "signal_type": signal_type,
            "price": signal['close'],
            "entry_price": calculate_entry_price(signal, safety_score),
            "momentum": signal['momentum'],
            "rsi": signal['rsi'],
            "macd": signal['macd'],
            "macd_signal": signal['macd_signal'],
            "macd_hist": macd_hist,
            "price_distance": signal['price_distance'],
            "safety_score": round(safety_score, 1),
            "safety_rating": safety_rating,
            "safety_emoji": safety_emoji,
            "safety_factors": {
                name: {
                    'score': factor['score'],
                    'contribution': round(factor['weight'] * factor['score'], 1)
                }
                for name, factor in safety_factors.items()
            }
        })

    # Sort signals by safety score
    signal_details.sort(key=lambda x: x['safety_score'], reverse=True)

    # Calculate overall trend
    recent_prices = data['close'].tail(5)
    trend = "UPTREND" if recent_prices.is_monotonic_increasing else "DOWNTREND"
    return {
        "signals": signal_details,
        "trend": trend,
        "current_price": current_price,
        "total_signals_found": len(signal_details),
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def calculate_rsi_safety_score(rsi: float, signal: int) -> float:
    """Calculate safety score based on RSI value and signal direction"""
    if signal == 1:  # Buy signal
        if rsi <= 30:
            return 100  # Very safe for buying (oversold)
        elif rsi <= 40:
            return 80
        elif rsi <= 50:
            return 60
        elif rsi <= 60:
            return 40
        elif rsi <= 70:
            return 20
        else:
            return 0  # Dangerous (overbought)
    else:  # Sell signal
        if rsi >= 70:
            return 100  # Very safe for selling (overbought)
        elif rsi >= 60:
            return 80
        elif rsi >= 50:
            return 60
        elif rsi >= 40:
            return 40
        elif rsi >= 30:
            return 20
        else:
            return 0  # Dangerous (oversold)


def calculate_momentum_safety_score(momentum: float, signal: int, macd: float, macd_signal: float, macd_hist: float) -> float:
    """
    Calculate safety score based on price momentum and MACD

    Parameters:
    - momentum: Price momentum (rate of change)
    - signal: Trade direction (1 for buy, -1 for sell)
    - macd: MACD line value
    - macd_signal: Signal line value
    - macd_hist: MACD histogram value (macd - signal)
    """
    abs_momentum = abs(momentum)

    # MACD Analysis (0-100 score)
    macd_score = 0
    if signal == 1:  # Buy signal
        if macd > macd_signal:  # Bullish MACD crossover
            macd_score = min(100, max(0, 50 + (macd_hist * 100)))
        else:
            macd_score = max(0, 50 - abs(macd_hist * 100))
    else:  # Sell signal
        if macd < macd_signal:  # Bearish MACD crossover
            macd_score = min(100, max(0, 50 + abs(macd_hist * 100)))
        else:
            macd_score = max(0, 50 - (macd_hist * 100))

    # Momentum Score (0-100)
    momentum_score = 0
    if signal == 1:  # Buy signal
        if momentum > 0:  # Positive momentum for buy
            if abs_momentum <= 1:
                momentum_score = 100  # Steady upward momentum
            elif abs_momentum <= 2:
                momentum_score = 80
            elif abs_momentum <= 3:
                momentum_score = 60
            else:
                momentum_score = 40  # Too volatile
        else:  # Negative momentum for buy
            momentum_score = max(0, 50 - abs_momentum * 10)

    else:  # Sell signal
        if momentum < 0:  # Negative momentum for sell
            if abs_momentum <= 1:
                momentum_score = 100  # Steady downward momentum
            elif abs_momentum <= 2:
                momentum_score = 80
            elif abs_momentum <= 3:
                momentum_score = 60
            else:
                momentum_score = 40  # Too volatile
        else:  # Positive momentum for sell
            momentum_score = max(0, 50 - abs_momentum * 10)

    # Combined score (60% MACD, 40% Momentum)
    final_score = (macd_score * 0.6) + (momentum_score * 0.4)

    return round(final_score, 1)


def calculate_entry_price(signal: pd.Series, safety_score: float) -> float:
    """Calculate entry price based on signal and safety score"""
    base_price = signal['close']

    # Adjust entry based on safety score
    if signal['signal'] == 1:  # Buy signal
        # More conservative for lower safety
        adjustment = (100 - safety_score) / 1000
        return base_price * (1 - adjustment)
    else:  # Sell signal
        # More conservative for lower safety
        adjustment = (100 - safety_score) / 1000
        return base_price * (1 + adjustment)


def calculate_entry_percentage(ob_direction: int, volume_score: float, ob_height_percent: float) -> dict:
    """
    Calculate entry percentages within the order block for limit orders
    Returns dictionary with aggressive, moderate, and conservative entry percentages

    Parameters:
    - ob_direction: 1 for bullish OB, -1 for bearish OB
    - volume_score: 0-100 score based on volume analysis
    - ob_height_percent: OB height as percentage of current price

    Returns:
    - Dictionary with entry percentages and zones
    """
    # Base percentages for different entry styles
    base_percentages = {
        'aggressive': 15.0,    # Enter closer to edge of OB
        'moderate': 30.0,      # Enter middle of OB
        'conservative': 45.0   # Enter deeper into OB
    }

    # Volume factor (0-10% adjustment)
    # Higher volume = entries closer to edge
    volume_factor = (volume_score / 100) * 10.0

    # Height factor (0-5% adjustment)
    # Larger OB = entries deeper into block
    height_factor = min(5.0, ob_height_percent * 0.1)

    # Calculate entry percentages for each style
    entry_zones = {}

    if ob_direction == 1:  # Bullish OB
        # For bullish OB, percentage is from bottom
        for style, base_pct in base_percentages.items():
            if style == 'aggressive':
                # Aggressive entries closer to bottom, reduced by factors
                pct = max(10.0, base_pct - volume_factor - height_factor)
            elif style == 'moderate':
                # Moderate entries in middle zone
                pct = base_pct
            else:  # conservative
                # Conservative entries deeper in OB, increased by factors
                pct = min(50.0, base_pct + volume_factor + height_factor)

            entry_zones[style] = {
                'percentage': pct,
                'description': f"{pct:.1f}% above OB bottom"
            }

    else:  # Bearish OB
        # For bearish OB, percentage is from top
        for style, base_pct in base_percentages.items():
            if style == 'aggressive':
                # Aggressive entries closer to top, reduced by factors
                pct = max(10.0, base_pct - volume_factor - height_factor)
            elif style == 'moderate':
                # Moderate entries in middle zone
                pct = base_pct
            else:  # conservative
                # Conservative entries deeper in OB, increased by factors
                pct = min(50.0, base_pct + volume_factor + height_factor)

            entry_zones[style] = {
                'percentage': pct,
                'description': f"{pct:.1f}% below OB top"
            }

    # Add additional information
    entry_zones['volume_score'] = volume_score
    entry_zones['height_factor'] = height_factor
    entry_zones['volume_factor'] = volume_factor

    return entry_zones


def calculate_stop_loss_percentage(ob_direction: int, volume_score: float, ob_height_percent: float) -> float:
    """
    Calculate stop loss percentage beyond the order block
    For longs: percentage below OB bottom
    For shorts: percentage above OB top
    """
    # Base percentage (0.5% beyond OB)
    base_percentage = 0.5

    # Volume factor (0-0.3% beyond OB)
    # Higher volume = tighter stop loss
    volume_factor = (1 - volume_score / 100) * 0.3

    # OB height factor (0-0.2% beyond OB)
    # Larger OB = wider stop loss
    height_factor = min(0.2, ob_height_percent * 0.05)

    # Calculate total percentage beyond OB
    total_percentage = base_percentage + volume_factor + height_factor

    # Cap at 1.5% beyond OB
    return min(1.5, total_percentage)


def calculate_pivot_points(data: pd.DataFrame) -> dict:
    """
    Calculate Pivot Points including support and resistance levels

    Parameters:
    - data: DataFrame with OHLCV data

    Returns:
    - Dictionary containing pivot points and levels
    """
    pivot = (data['high'].iloc[-1] + data['low'].iloc[-1] +
             data['close'].iloc[-1]) / 3
    r1 = 2 * pivot - data['low'].iloc[-1]
    s1 = 2 * pivot - data['high'].iloc[-1]
    r2 = pivot + (data['high'].iloc[-1] - data['low'].iloc[-1])
    s2 = pivot - (data['high'].iloc[-1] - data['low'].iloc[-1])

    return {
        'pivot': pivot,
        'r1': r1,
        'r2': r2,
        's1': s1,
        's2': s2
    }


def calculate_rsi_quality(rsi_value: float) -> float:
    """
    Calculate quality score based on RSI value

    Parameters:
    - rsi_value: Current RSI value

    Returns:
    - Quality score (0-20)
    """
    if rsi_value > 70:  # Overbought
        return 20  # High score for potential short setups
    elif rsi_value < 30:  # Oversold
        return 20  # High score for potential long setups
    elif 45 <= rsi_value <= 55:  # Neutral zone
        return 10  # Moderate score
    else:  # Trending conditions
        return 15  # Good score for trend following


def calculate_pivot_quality(price: float, pivots: dict) -> float:
    """
    Calculate quality score based on proximity to pivot points

    Parameters:
    - price: Current price
    - pivots: Dictionary of pivot points

    Returns:
    - Quality score (0-15)
    """
    # Calculate distance to each pivot level
    distances = {
        level: abs(price - value) / price * 100
        for level, value in pivots.items()
    }

    min_distance = min(distances.values())

    # Score based on closest distance
    if min_distance < 0.5:  # Very close to pivot (< 0.5%)
        return 15
    elif min_distance < 1.0:  # Close to pivot (< 1%)
        return 10
    elif min_distance < 2.0:  # Moderate distance (< 2%)
        return 5
    else:
        return 0


def calculate_macd_quality(data: pd.DataFrame, current_price: float) -> float:
    """
    Calculate quality score based on MACD, Signal line, and current price relationship
    Returns quality score (0-15)
    """
    # Get the latest values
    data = generate_signals(data)
    current_macd = data['macd'].iloc[-1]
    current_signal = data['macd_signal'].iloc[-1]
    last_close = data['close'].iloc[-1]

    # Calculate MACD histogram
    histogram = current_macd - current_signal

    # Calculate price momentum
    price_change_percent = ((current_price - last_close) / last_close) * 100

    # Calculate recent MACD momentum (last 10 periods)
    macd_change = data['macd'].diff().tail(10).mean()

    # Initialize quality score
    quality_score = 0

    # Score based on MACD vs Signal position (0-6 points)
    if current_macd > current_signal:  # Bullish MACD
        if price_change_percent > 0:  # Price confirms bullish signal
            quality_score += 6
        else:  # Price diverges from signal
            quality_score += 3
    else:  # Bearish MACD
        if price_change_percent < 0:  # Price confirms bearish signal
            quality_score += 6
        else:  # Price diverges from signal
            quality_score += 3

     # Score based on histogram strength (0-5 points)
    hist_strength = abs(histogram)
    if hist_strength > 0.5:
        quality_score += 5
    elif hist_strength > 0.2:
        quality_score += 3
    else:
        quality_score += 1

    # Score based on price-MACD alignment (0-4 points)
    if (price_change_percent > 0 and macd_change > 0) or \
       (price_change_percent < 0 and macd_change < 0):
        quality_score += 4  # Price and MACD moving in same direction
    elif abs(price_change_percent) < 0.1:
        quality_score += 2  # Price relatively stable

    # Cap the total score at 15
    return min(15, quality_score)


def calculate_setup_quality(
    data: pd.DataFrame,
    volume_score: float,
    price_distance: float,
    ob_height_percent: float,
    stop_distance: float,
    lookback: int = 10,
    macd_quality: float = 0,
) -> float:
    """
    Calculate overall setup quality score with enhanced analysis
    """
    # Get recent volume data
    recent_volumes = data['volume'].tail(lookback)
    current_volume = recent_volumes.iloc[-1]
    avg_recent_volume = recent_volumes.mean()

    # Calculate RSI
    rsi = calculate_rsi(data)
    current_rsi = rsi.iloc[-1]

    # Calculate Pivot Point
    # Enhanced volume quality (0-30 points)
    volume_ratio = current_volume / avg_recent_volume
    volume_trend = recent_volumes.pct_change().mean() * 100
    volume_trend_score = max(0, min(100, 50 + volume_trend * 10))

    volume_quality = (
        (volume_score * 0.4) +
        (volume_ratio * 30 * 0.4) +
        (volume_trend_score * 0.2)
    ) * 0.2  # 20 points max

    # Get RSI quality (0-20 points)
    rsi_quality = calculate_rsi_quality(current_rsi)

    # Get MACD point quality (0-15 points)

    # Price distance quality (0-15 points)
    distance_quality = max(0, (1 - price_distance * 10) * 10)

    # Order block height quality (0-10 points)
    height_quality = max(0, (1 - ob_height_percent / 5) * 25)

    # Stop loss quality (0-10 points)
    stop_quality = max(0, (1 - stop_distance * 20) * 5)

    # Calculate total quality score (0-100)
    total_quality = (
        volume_quality +      # 25 points
        rsi_quality +         # 20 points
        macd_quality +  # 10 points
        distance_quality +    # 10 points
        height_quality +      # 30 points
        stop_quality         # 5 points
    )

    return round(total_quality, 1)

    # Return quality metrics
    # return {
    #     'total_quality': round(total_quality, 1),
    #     'components': {
    #         'volume_quality': round(volume_quality, 1),
    #         'rsi_quality': round(rsi_quality, 1),
    #         'pivot_quality': round(pivot_quality, 1),
    #         'distance_quality': round(distance_quality, 1),
    #         'height_quality': round(height_quality, 1),
    #         'stop_quality': round(stop_quality, 1)
    #     },
    #     'technical_levels': {
    #         'rsi': round(current_rsi, 1),
    #         'pivot_points': {k: round(v, 2) for k, v in pivots.items()}
    #     }
    # }


def analyze_volume_patterns(data: pd.DataFrame, lookback: int = 20) -> dict:
    """Analyze volume patterns with RSI integration"""
    recent_data = data.tail(lookback).copy()

    # Calculate Volume RSI
    volume_rsi = calculate_rsi(recent_data['volume'], rsi_length=14)
    current_volume_rsi = volume_rsi.iloc[-1]

    # Analyze last candle
    last_candle = recent_data.iloc[-1]
    prev_candle = recent_data.iloc[-2]

    # Determine last candle type
    is_bullish = last_candle['close'] > last_candle['open']
    volume_increase = last_candle['volume'] > prev_candle['volume']

    # Calculate candle body and wicks
    body_size = abs(last_candle['close'] - last_candle['open'])
    upper_wick = last_candle['high'] - \
        max(last_candle['open'], last_candle['close'])
    lower_wick = min(last_candle['open'],
                     last_candle['close']) - last_candle['low']

    # Classify last candle
    if is_bullish:
        if body_size > upper_wick and volume_increase:
            candle_type = "STRONG_BUY"
            candle_score = 90
        elif body_size > upper_wick:
            candle_type = "BUY"
            candle_score = 70
        elif upper_wick > body_size:
            candle_type = "WEAK_BUY"
            candle_score = 55
        else:
            candle_type = "NEUTRAL"
            candle_score = 50
    else:
        if body_size > lower_wick and volume_increase:
            candle_type = "STRONG_SELL"
            candle_score = 10
        elif body_size > lower_wick:
            candle_type = "SELL"
            candle_score = 30
        elif lower_wick > body_size:
            candle_type = "WEAK_SELL"
            candle_score = 45
        else:
            candle_type = "NEUTRAL"
            candle_score = 50

    # Calculate recent candle patterns (last 3 candles)
    recent_candles = recent_data.tail(5)
    bullish_candles = recent_candles[recent_candles['close']
                                     > recent_candles['open']]
    bearish_candles = recent_candles[recent_candles['close']
                                     <= recent_candles['open']]

    buy_volume = bullish_candles['volume'].sum()
    sell_volume = bearish_candles['volume'].sum()
    total_volume = buy_volume + sell_volume

    # Calculate volume metrics
    avg_volume = recent_data['volume'].mean()
    current_volume = last_candle['volume']
    volume_ratio = current_volume / avg_volume

    # Calculate buy/sell ratios
    buy_ratio = (buy_volume / total_volume * 100) if total_volume > 0 else 0
    sell_ratio = (sell_volume / total_volume * 100) if total_volume > 0 else 0

    # Volume trend calculation
    volume_trend = recent_data['volume'].pct_change().mean() * 100
    volume_score = 0
    if volume_trend > 0:
        volume_score += 50  # Base score for positive trend
    if buy_ratio > 60:
        volume_score += 30  # Additional score for strong buying
    elif sell_ratio > 60:
        volume_score -= 30  # Penal
    # Determine volume pressure
    if candle_type in ["STRONG_BUY", "BUY"]:
        if volume_ratio > 1.5:
            pressure = "Strong Buying Pressure"
            pressure_score = 90
        else:
            pressure = "Moderate Buying Pressure"
            pressure_score = 70
    elif candle_type in ["STRONG_SELL", "SELL"]:
        if volume_ratio > 1.5:
            pressure = "Strong Selling Pressure"
            pressure_score = 10
        else:
            pressure = "Moderate Selling Pressure"
            pressure_score = 30
    else:
        if buy_ratio > 60:
            pressure = "Weak Buying Pressure"
            pressure_score = 60
        elif sell_ratio > 60:
            pressure = "Weak Selling Pressure"
            pressure_score = 40
        else:
            pressure = "Neutral Pressure"
            pressure_score = 50

    return {
        'volume_rsi': current_volume_rsi,
        'buy_ratio': buy_ratio,
        'volume_score': volume_score,
        'pressure_ratio': pressure_score / 100,
        'sell_ratio': sell_ratio,
        'volume_trend': volume_trend,
        'analysis': {
            'pressure': pressure,
            'score': pressure_score,
        },
        'last_candle': {
            'type': candle_type,
            'score': candle_score,
            'volume': current_volume,
            'volume_ratio': volume_ratio,
            'body_size': body_size,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick
        },
        'recent_pattern': {
            'bullish_count': len(bullish_candles),
            'bearish_count': len(bearish_candles),
            'dominant_side': 'BULLISH' if len(bullish_candles) > len(bearish_candles) else 'BEARISH'
        }
    }


def get_trade_recommendation(setup_quality: float, volume_analysis: dict,
                             price_distance: float, total_risk_score: int) -> str:
    """Generate detailed trade recommendation based on analysis"""

    if setup_quality < 50:
        return "DO NOT TRADE - Setup quality too low"

    if volume_analysis['volume_trend'] < -15:
        return "DO NOT TRADE - Significant volume decline"

    if price_distance > 0.05:  # 5% away from price
        return "MONITOR ONLY - Wait for price to come closer to entry"

    # High-quality setups
    if setup_quality >= 80:
        if volume_analysis['volume_trend'] > 10 and total_risk_score >= 0.8:
            return "STRONG ENTRY - High-quality setup with increasing volume"
        elif volume_analysis['volume_trend'] > 0:
            return "ENTER - High-quality setup with stable volume"
        else:
            return "PARTIAL ENTRY - High quality but declining volume"

    # Good setups
    if setup_quality >= 65:
        if volume_analysis['volume_trend'] > 5 and total_risk_score >= 0.7:
            return "ENTER - Good setup with improving volume"
        elif volume_analysis['volume_trend'] > -5:
            return "PARTIAL ENTRY - Good setup with stable volume"
        else:
            return "MONITOR - Good setup but volume concerns"

    # Moderate setups
    if setup_quality >= 50:
        if volume_analysis['volume_trend'] > 10 and total_risk_score >= 0.6:
            return "PARTIAL ENTRY - Moderate setup with strong volume"
        else:
            return "MONITOR - Moderate setup needs better conditions"

    return "DO NOT TRADE - Conditions not favorable"


def calculate_dynamic_risk_percentage(data: pd.DataFrame,
                                      entry_price: float,
                                      stop_loss: float,
                                      volume_score: float,
                                      ob_height_percent: float,
                                      current_price: float,
                                      ob_direction: int,
                                      vol_analysis: dict,
                                      macd_quality: dict) -> dict:
    """
    Calculate dynamic risk percentage with momentum analysis
    """

    # Calculate momentum indicators
    momentum_data = calculate_price_momentum(data, lookback=20)

    # Initialize risk factors dictionary
    risk_factors = {
        'momentum': {'score': 0, 'weight': 0.3, 'contribution': 0},
        'volume': {'score': 0, 'weight': 0.2, 'contribution': 0},
        'ob_quality': {'score': 0, 'weight': 0.3, 'contribution': 0},
        'price_action': {'score': 0, 'weight': 0.2, 'contribution': 0}
    }

    warning_messages = []

    # 1. Evaluate Momentum
    short_term_change = momentum_data['momentum']['short_term']['pct_change']
    momentum_direction = momentum_data['momentum']['short_term']['direction']
    candle_momentum = momentum_data['candle_momentum']['score']

    # Calculate momentum score (0-100)
    momentum_score = 50  # Base score
    momentum_score += short_term_change * 2  # Adjust based on price change
    momentum_score += candle_momentum * 0.5  # Add candle momentum influence
    momentum_score = max(0, min(100, momentum_score))  # Cap between 0-100

    risk_factors['momentum']['score'] = momentum_score

    if ob_direction == 1:  # Bullish OB
        if momentum_direction == -1 and abs(short_term_change) > 1:
            warning_messages.append(
                "⚠️ Strong bearish momentum against bullish setup")
        elif momentum_direction == -1:
            warning_messages.append("⚡ Moderate bearish pressure present")
    else:  # Bearish OB
        if momentum_direction == 1 and abs(short_term_change) > 1:
            warning_messages.append(
                "⚠️ Strong bullish momentum against bearish setup")
        elif momentum_direction == 1:
            warning_messages.append("⚡ Moderate bullish pressure present")

    # 2. Evaluate Volume
    risk_factors['volume']['score'] = volume_score
    if volume_score < 40:
        warning_messages.append("📊 Low volume confidence")

    # 3. Evaluate Order Block Quality
    ob_quality = 100 - min(100, ob_height_percent * 2)
    risk_factors['ob_quality']['score'] = ob_quality

    if ob_quality < 50:
        warning_messages.append(
            "📐 Large order block height - reduced precision")

    # 4. Evaluate Price Action
    trend_strength = momentum_data['trend_strength']['score']
    risk_factors['price_action']['score'] = (
        candle_momentum + trend_strength) / 2

    # Calculate weighted setup quality
    setup_quality = 0
    for factor, values in risk_factors.items():
        contribution = values['score'] * values['weight']
        values['contribution'] = contribution
        setup_quality += contribution

    # Determine trade recommendation
    if setup_quality >= 80:
        trade_recommendation = "Strong setup - Consider full position size"
    elif setup_quality >= 65:
        trade_recommendation = "Good setup - Consider moderate position size"
    elif setup_quality >= 50:
        trade_recommendation = "Moderate setup - Consider reduced position size"
    else:
        trade_recommendation = "Weak setup - Consider avoiding or minimal position"

    # Calculate final risk percentage based on setup quality
    base_risk = 1.0
    risk_multiplier = setup_quality / 100
    risk_percentage = base_risk * risk_multiplier

    return {
        'risk_percentage': risk_percentage,
        'setup_quality': setup_quality,
        'risk_factors': risk_factors,
        'warning_messages': warning_messages,
        'trade_recommendation': trade_recommendation
    }


def calculate_order_percentage(ob_data: dict, current_price: float, volume_score: int, trend: str) -> float:
    """
    Calculate optimal percentage for limit orders based on OB analysis

    Parameters:
    - ob_data: Order Block data including volume, price levels
    - current_price: Current market price
    - volume_score: Volume analysis score (0-100)
    - trend: Current market trend

    Returns:
    - percentage: Optimal percentage for limit order
    """
    # Base percentage (0.5% - 2%)
    base_percentage = 0.5

    # Volume factor (0-0.5%)
    volume_factor = (volume_score / 100) * 0.5

    # Price distance factor (0-0.5%)
    if ob_data['OB'] == 1:  # Bullish OB
        price_distance = abs(
            current_price - ob_data['Bottom']) / ob_data['Bottom']
    else:  # Bearish OB
        price_distance = abs(current_price - ob_data['Top']) / ob_data['Top']

    distance_factor = min(0.5, price_distance * 100)

    # Trend alignment factor (0-0.5%)
    trend_factor = 0.5 if (
        (ob_data['OB'] == 1 and trend == 'UPTREND') or
        (ob_data['OB'] == -1 and trend == 'DOWNTREND')
    ) else 0.25

    # Calculate final percentage
    total_percentage = base_percentage + \
        volume_factor + distance_factor + trend_factor

    # Cap at 3%
    return min(3.0, total_percentage)


def detect_ob_trend_reversal(data: pd.DataFrame, ob_results: dict, current_trend: str) -> dict:
    """
    Enhanced version with limit order percentage calculation
    """
    current_price = float(data['close'].iloc[-1])
    current_volume = float(data['volume'].iloc[-1])

    # Enhanced volume analysis
    lookback = 20  # Look back period for volume analysis
    recent_data = data.tail(lookback)
    avg_volume = recent_data['volume'].mean()
    volume_std = recent_data['volume'].std()
    volume_percentile = (recent_data['volume'] <= current_volume).mean() * 100

    # Volume conditions
    volume_conditions = {
        'above_average': current_volume > avg_volume * 1.2,
        'above_percentile': volume_percentile > 80,  # Above 80th percentile
        # Statistical outlier
        'surge': current_volume > (avg_volume + 2 * volume_std),
        # Increasing volume
        'increasing': current_volume > recent_data['volume'].iloc[-2]
    }

    # Calculate volume strength score (0-100)
    volume_score = (
        (volume_conditions['above_average'] * 25) +
        (volume_conditions['above_percentile'] * 25) +
        (volume_conditions['surge'] * 25) +
        (volume_conditions['increasing'] * 25)
    )

    reversal = {
        'detected': False,
        'type': None,
        'ob_level': None,
        'volume_analysis': {
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'volume_ratio': current_volume/avg_volume,
            'percentile': volume_percentile,
            'volume_score': volume_score,
            'conditions': volume_conditions
        },
        'price': current_price,
        'limit_order_percentage': None  # New field
    }

    # Check for bearish reversal in uptrend
    if current_trend == 'UPTREND':
        for i in range(len(ob_results["OB"])):
            if ob_results["OB"][i] == -1:  # Bearish OB
                ob_bottom = ob_results["Bottom"][i]
                ob_top = ob_results["Top"][i]
                ob_volume = ob_results["OBVolume"][i]

                if (current_price >= ob_bottom and
                    current_price <= ob_top and
                        volume_score >= 50):

                    ob_data = {
                        'OB': -1,
                        'Top': ob_top,
                        'Bottom': ob_bottom,
                        'Volume': ob_volume
                    }

                    limit_percentage = calculate_order_percentage(
                        ob_data,
                        current_price,
                        volume_score,
                        current_trend
                    )

                    reversal['detected'] = True
                    reversal['type'] = 'BEARISH'
                    reversal['ob_level'] = f"{ob_bottom:.2f}-{ob_top:.2f}"
                    reversal['ob_volume'] = ob_volume
                    reversal['strength'] = min(100, int(volume_score * 0.7 +
                                                        ((ob_top - current_price)/(ob_top - ob_bottom)) * 30))
                    reversal['limit_order_percentage'] = limit_percentage
                    break

    # Check for bullish reversal in downtrend
    elif current_trend == 'DOWNTREND':
        for i in range(len(ob_results["OB"])):
            if ob_results["OB"][i] == 1:  # Bullish OB
                ob_bottom = ob_results["Bottom"][i]
                ob_top = ob_results["Top"][i]
                ob_volume = ob_results["OBVolume"][i]

                if (current_price >= ob_bottom and
                    current_price <= ob_top and
                        volume_score >= 50):

                    ob_data = {
                        'OB': 1,
                        'Top': ob_top,
                        'Bottom': ob_bottom,
                        'Volume': ob_volume
                    }

                    limit_percentage = calculate_order_percentage(
                        ob_data,
                        current_price,
                        volume_score,
                        current_trend
                    )

                    reversal['detected'] = True
                    reversal['type'] = 'BULLISH'
                    reversal['ob_level'] = f"{ob_bottom:.2f}-{ob_top:.2f}"
                    reversal['ob_volume'] = ob_volume
                    reversal['strength'] = min(100, int(volume_score * 0.7 +
                                                        ((current_price - ob_bottom)/(ob_top - ob_bottom)) * 30))
                    reversal['limit_order_percentage'] = limit_percentage
                    break

    if reversal['detected']:
        print("\nVolume Analysis for Reversal:")
        print(f"Current Volume: {current_volume:.2f}")
        print(f"Average Volume: {avg_volume:.2f}")
        print(f"Volume Ratio: {current_volume/avg_volume:.2f}x")
        print(f"Volume Percentile: {volume_percentile:.1f}%")
        print(f"Volume Score: {volume_score}/100")
        print("\nVolume Conditions Met:")
        for condition, met in volume_conditions.items():
            print(f"- {condition}: {'✓' if met else '✗'}")
        print(f"Overall Reversal Strength: {reversal['strength']}/100")
        print(f"\nLimit Order Details:")
        print(
            f"Suggested Limit Order Percentage: {reversal['limit_order_percentage']:.2f}%")
        print(f"Limit Price for {reversal['type']} position: "
              f"{current_price * (1 - reversal['limit_order_percentage']/100):.2f}" if reversal['type'] == 'BEARISH'
              else f"{current_price * (1 + reversal['limit_order_percentage']/100):.2f}")

    return reversal


def calculate_velocity(data: pd.DataFrame, lookback: int = 3) -> dict:
    """
    Calculate price and volume velocity with MA and RSI confirmations
    """
    # Calculate MAs
    data['SMA50'] = data['close'].rolling(window=50).mean()
    data['EMA9'] = data['close'].ewm(span=5, adjust=False).mean()
    data['EMA21'] = data['close'].ewm(span=12, adjust=False).mean()

    # Calculate RSI
    rsi = calculate_rsi(data)
    current_rsi = rsi.iloc[-1]

    # macd = calculate_macd(data)
    # macd_hist = macd['histogram'].iloc[-1]
    # prev_macd_hist = macd['histogram'].iloc[-2]
    # macd_increasing = macd_hist > prev_macd_hist

    # Get current values
    current_price = data['close'].iloc[-1]
    current_sma50 = data['SMA50'].iloc[-1]
    current_ema9 = data['EMA9'].iloc[-1]
    current_ema21 = data['EMA21'].iloc[-1]

    # Price velocity calculation
    price_changes = data['close'].pct_change(periods=1).tail(lookback)
    current_price_velocity = price_changes.iloc[-1] * 100
    avg_price_velocity = price_changes.mean() * 100

    # Volume velocity calculation
    volume_changes = data['volume'].pct_change(periods=1).tail(lookback)
    current_volume_velocity = volume_changes.iloc[-1] * 100
    avg_volume_velocity = volume_changes.mean() * 100

    # MA Crossover detection
    ema_crossover = (
        data['EMA9'].iloc[-2] <= data['EMA21'].iloc[-2] and
        current_ema9 > current_ema21
    )

    # Market conditions
    conditions = {
        'above_sma50': current_price > current_sma50,
        'ema_crossover': ema_crossover,
        'rsi_above_50': current_rsi > 50,
        'rsi_oversold': current_rsi < 30,
        'rsi_overbought': current_rsi > 70
    }

    # Generate signals based on conditions
    signals = []
    if conditions['above_sma50'] and conditions['rsi_above_50']:
        signals.append("Price above SMA50 with bullish RSI")
    if conditions['ema_crossover']:
        signals.append("EMA5 crossed above EMA12")
    if conditions['rsi_oversold']:
        signals.append("RSI indicates oversold")
    if conditions['rsi_overbought']:
        signals.append("RSI indicates overbought")

    return {
        'price': {
            'current': current_price_velocity,
            'average': avg_price_velocity,
            'acceleration': current_price_velocity - avg_price_velocity,
            'condition': 'INCREASING' if current_price_velocity > avg_price_velocity else 'DECREASING'
        },
        'volume': {
            'current': current_volume_velocity,
            'average': avg_volume_velocity,
            'acceleration': current_volume_velocity - avg_volume_velocity,
            'condition': 'INCREASING' if current_volume_velocity > avg_volume_velocity else 'DECREASING'
        },
        'ma_analysis': {
            'above_sma50': conditions['above_sma50'],
            'ema_crossover': conditions['ema_crossover'],
            'current_price': current_price,
            'sma50': current_sma50,
            'ema9': current_ema9,
            'ema21': current_ema21
        },
        'rsi_analysis': {
            'current': current_rsi,
            'above_50': conditions['rsi_above_50'],
            'oversold': conditions['rsi_oversold'],
            'overbought': conditions['rsi_overbought']
        },
        'signals': signals
    }


def should_place_limit_order(velocity: dict, setup_type: str) -> dict:
    """
    Determine if current conditions are suitable for limit orders
    """
    price_vel = velocity['price']
    volume_vel = velocity['volume']

    # For long setups
    if setup_type.startswith('LONG'):
        if price_vel['condition'] == 'DECREASING' and volume_vel['condition'] == 'INCREASING':
            return {
                'place_order': True,
                'reason': "Price pullback with increasing volume - Good for long entry",
                'urgency': 'HIGH' if abs(price_vel['acceleration']) > 1 else 'MODERATE'
            }
    # For short setups
    elif setup_type.startswith('SHORT'):
        if price_vel['condition'] == 'INCREASING' and volume_vel['condition'] == 'INCREASING':
            return {
                'place_order': True,
                'reason': "Price rally with increasing volume - Good for short entry",
                'urgency': 'HIGH' if abs(price_vel['acceleration']) > 1 else 'MODERATE'
            }

    return {
        'place_order': False,
        'reason': "Current momentum not ideal for entry",
        'urgency': 'LOW'
    }

# Update analyze_trading_setup to include velocity analysis


def analyze_trading_setup(data, swing_hl):
    """
    Analyze trading setups and calculate order percentages for active order blocks only
    """
    trade_setups = []

    # Calculate EMAs for trend
    ema_34 = data['close'].ewm(span=34, adjust=False).mean()
    ema_89 = data['close'].ewm(span=89, adjust=False).mean()

    # Get current price and volume metrics
    current_price = float(data['close'].iloc[-1])
    avg_volume = data['volume'].mean()
    current_time = datetime.now().astimezone().astimezone(tz=None)

    try:
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        elif str(data.index.tz) != 'UTC':
            data.index = data.index.tz_convert('UTC')
    except Exception as e:
        print(f"Timezone conversion warning: {str(e)}")
        pass

    print(f"Current Time (UTC): {current_time}")

    # Determine current trend
    current_trend = 'UPTREND' if ema_34.iloc[-1] > ema_89.iloc[-1] else 'DOWNTREND'
    # Get order blocks
    ob_results = smc.ob(data, swing_hl)

    # Add velocity analysis
    velocity = calculate_velocity(data, 50)

    volume_analysis = analyze_volume_patterns(data, lookback=50)
    macd_quality = calculate_macd_quality(data, current_price)

    # Analyze each order block
    for i in range(len(ob_results)):
        if pd.notna(ob_results["OB"][i]):
            # Check if order block is still active
            x1 = int(ob_results["MitigatedIndex"][i]
                     if ob_results["MitigatedIndex"][i] != 0 else len(data) - 1)
            # Skip if order block has been mitigated
            if not (current_time < data.index[x1] + timedelta(minutes=30)):
                continue

            ob_volume = ob_results["OBVolume"][i]
            ob_direction = ob_results["OB"][i]
            ob_top = ob_results["Top"][i]
            ob_bottom = ob_results["Bottom"][i]
            ob_height = ob_top - ob_bottom
            ob_height_percent = (ob_height/current_price) * 100

            # Calculate volume metrics
            volume_ratio = ob_volume / avg_volume
            volume_score = min(100, int((volume_ratio - 1) * 50))

            # Calculate entry zones
            entry_zones = calculate_entry_percentage(
                ob_direction, volume_score, ob_height_percent)

            # Calculate stop loss percentage
            stop_loss_percent = calculate_stop_loss_percentage(
                ob_direction, volume_score, ob_height_percent)

            # Calculate actual price levels
            if ob_direction == 1:  # Bullish OB
                entry_prices = {
                    style: ob_bottom + (ob_height * zone['percentage']/100)
                    for style, zone in entry_zones.items()
                    if style in ['aggressive', 'moderate', 'conservative']
                }
                stop_loss = ob_bottom * (1 - stop_loss_percent/100)
            else:  # Bearish OB
                entry_prices = {
                    style: ob_top - (ob_height * zone['percentage']/100)
                    for style, zone in entry_zones.items()
                    if style in ['aggressive', 'moderate', 'conservative']
                }
                stop_loss = ob_top * (1 + stop_loss_percent/100)

            # Get risk assessment
            risk_assessment = calculate_dynamic_risk_percentage(
                data=data,
                entry_price=entry_prices['moderate'],
                stop_loss=stop_loss,
                volume_score=volume_score,
                ob_height_percent=ob_height_percent,
                current_price=current_price,
                ob_direction=ob_direction,
                vol_analysis=volume_analysis,
                macd_quality=macd_quality
            )

            # Determine setup type based on OB direction, trend, and volume pressure
            if ob_direction == 1:  # Bullish OB
                if current_trend == 'DOWNTREND':
                    if volume_analysis['analysis']['pressure'] in ['Strong Buying', 'Moderate Buying']:
                        # Break of Structure Long (Counter-trend)
                        setup_type = 'LONG_BOS'
                        setup_strength = 'Strong'
                    else:
                        setup_type = 'LONG_CHoCH'  # Change of Character Long
                        setup_strength = 'Moderate'
                else:  # UPTREND
                    if volume_analysis['analysis']['pressure'] in ['Strong Buying Climax', 'Moderate Buying Climax']:
                        setup_type = 'LONG_CONTINUATION'  # Continuation Long
                        setup_strength = 'Strong'
                    else:
                        setup_type = 'LONG_PULLBACK'  # Pullback in Uptrend
                        setup_strength = 'Moderate'

            else:  # Bearish OB
                if current_trend == 'UPTREND':
                    if volume_analysis['analysis']['pressure'] in ['Strong Selling Climax', 'Moderate Selling Climax']:
                        # Break of Structure Short (Counter-trend)
                        setup_type = 'SHORT_BOS'
                        setup_strength = 'Strong'
                    else:
                        setup_type = 'SHORT_CHoCH'  # Change of Character Short
                        setup_strength = 'Moderate'
                else:  # DOWNTREND
                    if volume_analysis['analysis']['pressure'] in ['Strong Selling Climax', 'Moderate Selling Climax']:
                        setup_type = 'SHORT_CONTINUATION'  # Continuation Short
                        setup_strength = 'Strong'
                    else:
                        setup_type = 'SHORT_PULLBACK'  # Pullback in Downtrend
                        setup_strength = 'Moderate'

            # Get the risk percentage from the assessment
            risk_percentage = risk_assessment['risk_percentage']

            # Adjust leverage based on risk
            if risk_percentage <= 0.5:
                max_leverage = 30  # More conservative setups allow higher leverage
            elif risk_percentage <= 0.75:
                max_leverage = 50  # Moderate risk setups
            else:
                max_leverage = 70  # Higher risk setups get limited leverage

            if risk_percentage <= 0:
                suggested_leverage = 10
            else:
                suggested_leverage = min(
                    max_leverage, int(1 / risk_percentage * 50))

            # Get entry quality based on setup quality
            if risk_assessment['setup_quality'] >= 80:
                entry_quality = 'Excellent'
            elif risk_assessment['setup_quality'] >= 65:
                entry_quality = 'Good'
            elif risk_assessment['setup_quality'] >= 50:
                entry_quality = 'Moderate'
            else:
                entry_quality = 'Poor'

            # Create setup dictionary
            setup = {
                'current_price': current_price,
                'type': setup_type,
                'ob_direction': 'Bullish' if ob_direction == 1 else 'Bearish',
                'current_trend': current_trend,
                'ob_level': f"{ob_bottom:.0f}-{ob_top:.0f}",
                'entry_prices': entry_prices,
                'entry_zones': entry_zones,
                'stop_loss': stop_loss,
                'stop_loss_percentage': stop_loss_percent,
                'volume_score': volume_score,
                'volume_ratio': volume_ratio,
                'ob_volume': ob_volume,
                'ob_volume_ratio': ob_volume / avg_volume,
                'risk_percentage': risk_percentage,
                'suggested_leverage': suggested_leverage,
                'setup_quality': risk_assessment['setup_quality'],
                'entry_quality': entry_quality,
                'trade_recommendation': risk_assessment['trade_recommendation'],
                'warning_messages': risk_assessment['warning_messages'],
                'risk_factors': risk_assessment['risk_factors'],
                'risk_rating': 'Low' if risk_percentage <= 0.5 else
                'Moderate' if risk_percentage <= 0.75 else 'High',
                'effective_risk': risk_percentage * suggested_leverage
            }
            ob_height = ob_top - ob_bottom
            if ob_direction == 1:  # Bullish OB
                tp1 = ob_top + (ob_height * 1.0)  # 100% extension
                tp2 = ob_top + (ob_height * 1.5)  # 150% extension
                tp3 = ob_top + (ob_height * 2.0)  # 200% extension
            else:  # Bearish OB
                tp1 = ob_bottom - (ob_height * 1.0)
                tp2 = ob_bottom - (ob_height * 1.5)
                tp3 = ob_bottom - (ob_height * 2.0)
            setup.update({
                'setup_type': setup_type,
                'setup_strength': setup_strength,
                'setup_description': setup_classification[setup_type],
                'position_type': 'LONG' if setup_type.startswith('LONG') else 'SHORT',

                'take_profit_levels': {
                    'tp1': tp1,
                    'tp2': tp2,
                    'tp3': tp3
                }
            })

            # Add velocity analysis to each setup
            setup['limit_order_recommendation'] = should_place_limit_order(
                velocity,
                setup['setup_type']
            )

            trade_setups.append(setup)

    # Before returning trade_setups, add price distance and sort:
    for setup in trade_setups:
        # Calculate price distance percentage for moderate entry
        moderate_entry = setup['entry_prices']['moderate']
        price_distance = abs(
            current_price - moderate_entry) / current_price * 100
        setup['price_distance'] = price_distance

    # Sort setups by:
    # 1. Price distance (closest first)
    # 2. Setup quality (highest first)
    # 3. Volume score (highest first)
    trade_setups.sort(key=lambda x: (
        x['price_distance'],          # Primary: closest to price
        -x['setup_quality'],          # Secondary: highest quality
        -x['volume_score']            # Tertiary: highest volume
    ))

    # if not trade_setups:
    #     print("\nNo valid active setups found")
    # else:
    #     print(f"\nFound {len(trade_setups)} potential active setups")
    #     print("\nSetups Summary (Sorted by proximity to current price):")
    #     for i, setup in enumerate(trade_setups, 1):
    #         print(f"\n{'='*50}")
    #         print(f"🎯 Trade Setup Analysis {i}:")
    #         print(f"Type: {setup['type']}")
    #         print(f"Setup Quality: {setup['setup_quality']}/100")
    #         print(f"Entry Quality: {setup['entry_quality']}")
    #         print(f"Risk Percentage: {setup['risk_percentage']:.3f}%")
    #         print(f"Recommended Leverage: {setup['suggested_leverage']}x")
    #         print("\n📊 Risk Factors:")
    #         for name, factor in setup['risk_factors'].items():
    #             print(f"{name.title()}: {factor['score']:.1f}/100 "
    #                   f"(Contributing {factor['contribution']:.3f}%)")
    #         print("\n⚠️ Warnings:")
    #         for warning in setup['warning_messages']:
    #             print(warning)
    #         print(f"\n📝 Recommendation: {setup['trade_recommendation']}")
    #         print(
    #             f"Distance from current price: {setup['price_distance']:.2f}%")
    #         print(f"Entry price: {setup['entry_prices']['moderate']:.2f}")
    #         print(f"Current price: {current_price:.2f}")
    #         print(f"{'='*50}")
    return {
        "trade_setups": trade_setups,
        "current_price": current_price,
        "velocity": velocity,
        "volume_analysis": volume_analysis,
        "current_trend": current_trend,
        "current_volume": data['volume'].iloc[-1],
        "current_volume_ratio": data['volume'].iloc[-1] / data['volume'].tail(20).mean(),
        'last_candle_signal': "SELL" if data['close'].iloc[-1] <= data['open'].iloc[-1] else "BUY"
    }


def detect_trend(data: pd.DataFrame) -> str:
    """
    Detect market trend including sideways condition

    Parameters:
    - data: DataFrame with OHLCV data

    Returns:
    - str: 'UPTREND', 'DOWNTREND', or 'SIDEWAYS'
    """
    # Calculate EMAs
    ema_34 = data['close'].ewm(span=34, adjust=False).mean()
    ema_89 = data['close'].ewm(span=89, adjust=False).mean()

    # Calculate percentage difference between EMAs
    ema_diff_percent = abs(
        (ema_34.iloc[-1] - ema_89.iloc[-1]) / ema_89.iloc[-1] * 100)

    # Define sideways threshold (e.g., 0.5% difference between EMAs)
    SIDEWAYS_THRESHOLD = 0.5

    if ema_diff_percent <= SIDEWAYS_THRESHOLD:
        return 'SIDEWAYS'
    elif ema_34.iloc[-1] > ema_89.iloc[-1]:
        return 'UPTREND'
    else:
        return 'DOWNTREND'


def find_pivots(series, left, right):
    """Find pivot points in a series."""
    pivots_low = []
    pivots_high = []

    for i in range(left, len(series) - right):
        if series[i] == min(series[i - left:i + right + 1]):
            pivots_low.append(i)
        elif series[i] == max(series[i - left:i + right + 1]):
            pivots_high.append(i)

    return pivots_low, pivots_high


def detect_rsi_divergence(data: pd.DataFrame, lookback: int = 20, lbL: int = 5, lbR: int = 5, rangeUpper: int = 60, rangeLower: int = 5) -> dict:
    """
    Detect both bullish and bearish RSI divergences without using ta functions.

    Parameters:
    -----------
    data: DataFrame with OHLCV data
    lookback: Number of candles to look back
    lbL: Pivot Lookback Left
    lbR: Pivot Lookback Right
    rangeUpper: Max of Lookback Range
    rangeLower: Min of Lookback Range

    Returns:
    --------
    dict: Contains both bullish and bearish divergence analysis
    """
    # Calculate RSI
    rsi = calculate_rsi(data)
    n = len(rsi)

    # Find pivots
    pivots_low, pivots_high = find_pivots(rsi, lbL, lbR)

    results = {
        'bullish': [],
        'bearish': []
    }

    for i in pivots_low:
        if i >= lbR and i < n - lbR:
            oscHL = rsi[i] > rsi[i - lbR]  # RSI Higher Low
            priceLL = data['low'][i] < data['low'][i - lbR]  # Price Lower Low
            if priceLL and oscHL:
                results['bullish'].append({
                    'type': 'Regular Bullish',
                    'strength': 1,  # Placeholder for strength calculation
                    'points': {
                        'start': {'time': data.index[i - lbR], 'price': data['low'][i - lbR], 'rsi': rsi[i - lbR]},
                        'end': {'time': data.index[i], 'price': data['low'][i], 'rsi': rsi[i]}
                    }
                })

    # Regular Bearish Divergence
    for i in pivots_high:
        if i >= lbR and i < n - lbR:
            oscLH = rsi[i] < rsi[i - lbR]  # RSI Lower High
            # Price Higher High
            priceHH = data['high'][i] > data['high'][i - lbR]
            if priceHH and oscLH:
                results['bearish'].append({
                    'type': 'Regular Bearish',
                    'strength': 1,  # Placeholder for strength calculation
                    'points': {
                        'start': {'time': data.index[i - lbR], 'price': data['high'][i - lbR], 'rsi': rsi[i - lbR]},
                        'end': {'time': data.index[i], 'price': data['high'][i], 'rsi': rsi[i]}
                    }
                })

    return results


# def detect_rsi_divergence(data: pd.DataFrame, lookback: int = 20) -> dict:
#     """
#     Detect both bullish and bearish RSI divergences with improved algorithm

#     Parameters:
#     -----------
#     data: DataFrame with OHLCV and RSI data
#     lookback: Number of candles to look back

#     Returns:
#     --------
#     dict: Contains both bullish and bearish divergence analysis
#     """
#     # Get recent data
#     recent_data = data.tail(lookback).copy()

#     def find_extremes(df: pd.DataFrame, window: int = 20) -> tuple:
#         """Find local highs and lows with improved accuracy"""
#         highs = []
#         lows = []

#         for i in range(window, len(df) - window):
#             price_window = df['close'].iloc[i-window:i+window+1]
#             rsi_window = df['rsi'].iloc[i-window:i+window+1]

#             # Price extremes
#             if df['close'].iloc[i] == price_window.max():
#                 highs.append({
#                     'index': i,
#                     'price': df['close'].iloc[i],
#                     'rsi': df['rsi'].iloc[i],
#                     'time': df.index[i]
#                 })

#             if df['close'].iloc[i] == price_window.min():
#                 lows.append({
#                     'index': i,
#                     'price': df['close'].iloc[i],
#                     'rsi': df['rsi'].iloc[i],
#                     'time': df.index[i]
#                 })

#         return highs, lows

#     def analyze_divergence(points: list, point_type: str) -> list:
#         """Analyze sequence of points for divergence with stricter conditions"""
#         if len(points) < 2:
#             return []

#         divergences = []
#         for i in range(1, len(points)):
#             current = points[i]
#             previous = points[i-1]

#             # Calculate time difference to avoid false signals
#             time_diff = (current['time'] - previous['time']
#                          ).total_seconds() / 3600
#             if time_diff < 4:  # Minimum 4 hours between points
#                 continue

#             if point_type == 'low':
#                 # Bullish divergence
#                 price_lower = current['price'] < previous['price']
#                 rsi_higher = current['rsi'] > previous['rsi']

#                 if price_lower and rsi_higher:
#                     # Calculate divergence strength
#                     price_change = abs(
#                         (current['price'] - previous['price']) / previous['price'] * 100)
#                     rsi_change = current['rsi'] - previous['rsi']

#                     if price_change > 0.5 and rsi_change > 2:  # Minimum thresholds
#                         divergences.append({
#                             'type': 'bullish',
#                             'strength': min(price_change * rsi_change / 10, 100),
#                             'points': {
#                                 'start': previous,
#                                 'end': current
#                             }
#                         })

#             elif point_type == 'high':
#                 # Bearish divergence
#                 price_higher = current['price'] > previous['price']
#                 rsi_lower = current['rsi'] < previous['rsi']

#                 if price_higher and rsi_lower:
#                     price_change = abs(
#                         (current['price'] - previous['price']) / previous['price'] * 100)
#                     rsi_change = previous['rsi'] - current['rsi']

#                     if price_change > 0.5 and rsi_change > 2:
#                         divergences.append({
#                             'type': 'bearish',
#                             'strength': min(price_change * rsi_change / 10, 100),
#                             'points': {
#                                 'start': previous,
#                                 'end': current
#                             }
#                         })

#         return divergences

#     # Find extremes
#     highs, lows = find_extremes(recent_data)

#     # Detect divergences
#     bullish_divs = analyze_divergence(lows, 'low')
#     bearish_divs = analyze_divergence(highs, 'high')

#     return {
#         'bullish': bullish_divs,
#         'bearish': bearish_divs
#     }
