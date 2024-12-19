from smartmoneyconcepts.smc import smc
from datetime import datetime, timedelta
import pandas as pd
from typing import Union

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
    return pd.DataFrame({'macd': macd, 'signal': signal})


def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals based on RSI and MACD."""
    # Calculate RSI
    data['rsi'] = calculate_rsi(data)

    # Calculate MACD
    macd_df = calculate_macd(data)
    data['macd'] = macd_df['macd']
    data['macd_signal'] = macd_df['signal']

    # Generate signals
    data['signal'] = 0  # Default to no signal
    data['signal'] = data.apply(lambda x: 1 if (
        x['rsi'] < 30 and x['macd'] > x['macd_signal']) else x['signal'], axis=1)  # Buy signal
    data['signal'] = data.apply(lambda x: -1 if (x['rsi'] > 70 and x['macd']
                                < x['macd_signal']) else x['signal'], axis=1)  # Sell signal

    return data


def find_closest_signal(data: pd.DataFrame, current_price: float) -> dict:
    """Find the closest signal to the current price and determine entry price."""
    # Filter signals
    # Create a copy to avoid SettingWithCopyWarning
    signals = data[data['signal'] != 0].copy()

    if signals.empty:
        return {"signal": None, "trend": "No signals available", "entry_price": None}

    # Find the closest signal
    signals.loc[:, 'price_distance'] = (
        signals['close'] - current_price).abs()  # Use .loc to set values
    closest_signal = signals.loc[signals['price_distance'].idxmin()]

    # Determine short trend based on last few closing prices
    recent_prices = data['close'].tail(5)
    trend = "UPTREND" if recent_prices.is_monotonic_increasing else "DOWNTREND"

    # Calculate entry price based on signal price
    if closest_signal['signal'] == 1:  # Buy signal
        entry_price = closest_signal['close'] * \
            0.99  # 1% below the signal price
    elif closest_signal['signal'] == -1:  # Sell signal
        entry_price = closest_signal['close'] * \
            1.01  # 1% above the signal price
    else:
        # Default to signal price if no action
        entry_price = closest_signal['close']

    return {
        "signal": closest_signal['signal'],
        "price": closest_signal['close'],
        "entry_price": entry_price,
        "trend": trend,
        "rsi": closest_signal['rsi'],
        "macd": closest_signal['macd'],
        "macd_signal": closest_signal['macd_signal']
    }


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

    # Calculate recent MACD momentum (last 5 periods)
    macd_change = data['macd'].diff().tail(5).mean()

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
    lookback: int = 10
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

    # Calculate Pivot Points
    current_price = data['close'].iloc[-1]

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
    macd_quality = calculate_macd_quality(data, current_price)

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

    # Calculate volume score with RSI integration
    avg_volume = recent_data['volume'].mean()
    current_volume = recent_data['volume'].iloc[-1]
    volume_ratio = current_volume / avg_volume

    # Volume trend calculation
    volume_trend = recent_data['volume'].pct_change().mean() * 100

    # RSI-based volume conditions
    volume_conditions = {
        'overbought': current_volume_rsi > 70,
        'oversold': current_volume_rsi < 30,
        'rising': current_volume_rsi > 50 and volume_trend > 0,
        'falling': current_volume_rsi < 50 and volume_trend < 0
    }

    # Enhanced volume score calculation (0-100)
    volume_score = min(100, max(0, int((
        # Volume ratio contribution (30%)
        (volume_ratio - 0.5) * 30 +
        # RSI contribution (40%)
        (current_volume_rsi * 0.4) +
        # Trend contribution (30%)
        (volume_trend * 3) +
        # Base score adjustment
        20
    ))))

    # Calculate pressure (buy/sell ratio with RSI context)
    buy_candles = recent_data[recent_data['close'] > recent_data['open']]
    sell_candles = recent_data[recent_data['close'] <= recent_data['open']]

    buy_volume = buy_candles['volume'].sum()
    sell_volume = sell_candles['volume'].sum()
    total_volume = buy_volume + sell_volume

    # Calculate ratios
    buy_ratio = (buy_volume / total_volume * 100) if total_volume > 0 else 0
    sell_ratio = (sell_volume / total_volume * 100) if total_volume > 0 else 0
    pressure_ratio = (
        buy_volume / sell_volume) if sell_volume > 0 else float('inf')

    # Determine volume dominance with RSI context
    if current_volume_rsi > 70:
        if buy_ratio > 60:
            dominance = "Strong Buyers Dominant"
        elif sell_ratio > 60:
            dominance = "Strong Sellers Dominant"
        else:
            dominance = "High Volume Neutral"
    elif current_volume_rsi < 30:
        if buy_ratio > 60:
            dominance = "Weak Buyers Present"
        elif sell_ratio > 60:
            dominance = "Weak Sellers Present"
        else:
            dominance = "Low Volume Neutral"
    else:
        if buy_ratio > 60:
            dominance = "Buyers Dominant"
        elif sell_ratio > 60:
            dominance = "Sellers Dominant"
        else:
            dominance = "Neutral"

    # Determine pressure with RSI context
    if current_volume_rsi > 70:
        if pressure_ratio > 1.2:
            pressure = "Strong Buying Climax"
        elif pressure_ratio < 0.8:
            pressure = "Strong Selling Climax"
        else:
            pressure = "High Volume Churn"
    elif current_volume_rsi < 30:
        if pressure_ratio > 1.2:
            pressure = "Weak Buying"
        elif pressure_ratio < 0.8:
            pressure = "Weak Selling"
        else:
            pressure = "Low Volume Consolidation"
    else:
        if pressure_ratio > 1.2:
            pressure = "Moderate Buying"
        elif pressure_ratio < 0.8:
            pressure = "Moderate Selling"
        else:
            pressure = "Neutral"

    return {
        'volume_score': volume_score,
        'volume_rsi': current_volume_rsi,
        'buy_ratio': buy_ratio,
        'sell_ratio': sell_ratio,
        'pressure_ratio': pressure_ratio,
        'volume_trend': volume_trend,
        'volume_conditions': volume_conditions,
        'analysis': {
            'pressure': pressure,
            'dominance': dominance
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


def calculate_dynamic_risk_percentage(
    data: pd.DataFrame,
    entry_price: float,
    stop_loss: float,
    volume_score: float,
    ob_height_percent: float,
    current_price: float,
    ob_direction: int
) -> dict:
    """
    Calculate risk percentage based on market conditions and volume patterns

    Parameters:
    - data: DataFrame containing price and volume data
    - entry_price: Planned entry price
    - stop_loss: Stop loss level
    - volume_score: Volume analysis score (0-100)
    - ob_height_percent: Order block height as percentage
    - current_price: Current market price
    - ob_direction: Order block direction (1 for bullish, -1 for bearish)

    Returns:
    - Dictionary containing risk analysis and trade recommendations
    """
    # Get volume analysis
    volume_analysis = analyze_volume_patterns(data, lookback=50)
    volume_score = volume_analysis['volume_score']

    # Calculate basic metrics
    price_distance = abs(current_price - entry_price) / current_price
    stop_distance = abs(entry_price - stop_loss) / entry_price

    # Calculate setup quality with volume analysis
    setup_quality = calculate_setup_quality(
        data=data,
        volume_score=volume_score,
        price_distance=price_distance,
        ob_height_percent=ob_height_percent,
        stop_distance=stop_distance
    )

    # Determine if volume aligns with trade direction
    volume_alignment = (
        # Bullish with buying pressure
        (ob_direction == 1 and volume_analysis['pressure_ratio'] > 1.2) or
        # Bearish with selling pressure
        (ob_direction == -1 and volume_analysis['pressure_ratio'] < 0.83)
    )

    # Update risk factors with detailed volume analysis
    risk_factors = {
        'volume': {
            'score': volume_score,
            'weight': 0.3,
            'contribution': (volume_score / 100) * 0.3
        },
        'volume_trend': {
            'score': max(0, min(100, 50 + volume_analysis['volume_trend'] * 5)),
            'weight': 0.2,
            'contribution': max(0, min(0.2, (50 + volume_analysis['volume_trend'] * 5) / 100 * 0.2))
        },
        'pressure_alignment': {
            'score': 100 if volume_alignment else 0,
            'weight': 0.2,
            'contribution': 0.2 if volume_alignment else 0
        },
        'price_distance': {
            'score': max(0, (1 - price_distance * 10) * 100),
            'weight': 0.2,
            'contribution': max(0, 0.2 * (1 - price_distance * 10))
        },
        'ob_height': {
            'score': max(0, (1 - ob_height_percent / 5) * 100),
            'weight': 0.15,
            'contribution': max(0, 0.15 * (1 - ob_height_percent / 5))
        },
        'trend_alignment': {
            'score': 100 if (
                (ob_direction == 1 and current_price > entry_price) or
                (ob_direction == -1 and current_price < entry_price)
            ) else 0,
            'weight': 0.15,
            'contribution': 0.15 if (
                (ob_direction == 1 and current_price > entry_price) or
                (ob_direction == -1 and current_price < entry_price)
            ) else 0
        }
    }

    # Include volume analysis warnings

    # Calculate total risk score from all factors
    total_risk_score = sum(factor['contribution']
                           for factor in risk_factors.values())

    # Update trade recommendation based on volume patterns
    trade_recommendation = get_trade_recommendation(
        setup_quality=setup_quality,
        volume_analysis=volume_analysis,
        price_distance=price_distance,
        total_risk_score=total_risk_score
    )

    # Risk Assessment
    risk_assessment = {
        'setup_quality': setup_quality,
        'risk_percentage': 0,
        'recommended_leverage': 0,
        'trade_recommendation': '',
        'warning_messages': [],
        'entry_quality': '',
        'risk_factors': {},
        'volume_analysis': volume_analysis,
        'trade_recommendation': trade_recommendation,
    }

    # Determine entry quality based on setup quality and volume patterns
    if setup_quality >= 80 and volume_analysis['volume_trend'] > 0:
        risk_assessment['entry_quality'] = 'Excellent'
        base_risk = 1.0
    elif setup_quality >= 65 and volume_analysis['volume_trend'] > -5:
        risk_assessment['entry_quality'] = 'Good'
        base_risk = 0.75
    elif setup_quality >= 50 and volume_analysis['volume_trend'] > -10:
        risk_assessment['entry_quality'] = 'Moderate'
        base_risk = 0.5
    else:
        risk_assessment['entry_quality'] = 'Poor'
        base_risk = 0
        risk_assessment['trade_recommendation'] = 'DO NOT TRADE - Setup quality too low'

    # Add warning messages based on conditions
    if price_distance > 0.03:  # More than 3% away
        risk_assessment['warning_messages'].append(
            "⚠️ Entry far from current price - Higher risk")

    if volume_analysis['volume_trend'] < -5:
        risk_assessment['warning_messages'].append(
            "⚠️ Declining volume trend - Monitor volume")

    if volume_analysis['volume_score'] < 40:
        risk_assessment['warning_messages'].append(
            "⚠️ Poor volume conditions - Consider smaller position")

    if ob_height_percent > 3:
        risk_assessment['warning_messages'].append(
            "⚠️ Large order block - Wide stop required")

    if setup_quality < 65 and volume_analysis['volume_trend'] < 0:
        risk_assessment['warning_messages'].append(
            "⚠️ Lower quality setup with weak volume")

    # Update final risk assessment
    risk_assessment.update({
        'risk_factors': risk_factors,
        'risk_percentage': base_risk * total_risk_score,
        'trade_recommendation': trade_recommendation,
        'setup_quality': setup_quality,
        'entry_quality': risk_assessment['entry_quality'],
        'warning_messages': risk_assessment['warning_messages'],
        'volume_metrics': {
            'trend': volume_analysis['volume_trend'],
            'volatility': volume_analysis['volume_trend'],
            'current_vs_average': data['volume'].tail(10).iloc[-1] / data['volume'].tail(10).mean()
        }
    })

    if risk_assessment['risk_percentage'] == 0:
        risk_assessment['recommended_leverage'] = 75
    else:

        # Adjust recommended leverage based on risk assessment
        if setup_quality >= 80 and volume_analysis['volume_trend'] > 0:
            risk_assessment['recommended_leverage'] = min(
                20, int(1 / risk_assessment['risk_percentage']))
        elif setup_quality >= 65:
            risk_assessment['recommended_leverage'] = min(
                15, int(1 / risk_assessment['risk_percentage']))
        else:
            risk_assessment['recommended_leverage'] = min(
                10, int(1 / risk_assessment['risk_percentage']))

    return risk_assessment


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
    data['EMA5'] = data['close'].ewm(span=5, adjust=False).mean()
    data['EMA12'] = data['close'].ewm(span=12, adjust=False).mean()

    # Calculate RSI
    rsi = calculate_rsi(data)
    current_rsi = rsi.iloc[-1] - 5

    # Get current values
    current_price = data['close'].iloc[-1]
    current_sma50 = data['SMA50'].iloc[-1]
    current_ema5 = data['EMA5'].iloc[-1]
    current_ema12 = data['EMA12'].iloc[-1]

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
        data['EMA5'].iloc[-2] <= data['EMA12'].iloc[-2] and
        current_ema5 > current_ema12
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
            'ema5': current_ema5,
            'ema12': current_ema12
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

    # Analyze each order block
    for i in range(len(ob_results)):
        if pd.notna(ob_results["OB"][i]):
            # Check if order block is still active
            x1 = int(ob_results["MitigatedIndex"][i]
                     if ob_results["MitigatedIndex"][i] != 0 else len(data) - 1)

            # Skip if order block has been mitigated
            if not (current_time > data.index[i] and current_time < data.index[x1] + timedelta(minutes=15)):
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
                ob_direction=ob_direction
            )
            volume_analysis = risk_assessment['volume_analysis']

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
                max_leverage = 50  # More conservative setups allow higher leverage
            elif risk_percentage <= 0.75:
                max_leverage = 30  # Moderate risk setups
            else:
                max_leverage = 10  # Higher risk setups get limited leverage

            if risk_percentage <= 0:
                suggested_leverage = 75
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
                'volume_analysis': risk_assessment['volume_analysis'],
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
