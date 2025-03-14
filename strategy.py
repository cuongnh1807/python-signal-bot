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

    macd_current = macd.iloc[-1]
    macd_prev = macd.iloc[-2]
    signal_current = signal.iloc[-1]
    signal_prev = signal.iloc[-2]
    hist_current = histogram.iloc[-1]
    hist_prev = histogram.iloc[-2]

    macd_cross_up = macd_current > signal_current and macd_prev <= signal_prev
    macd_cross_down = macd_current < signal_current and macd_prev >= signal_prev
    hist_pos_to_neg = hist_current < 0 and hist_prev >= 0
    hist_neg_to_pos = hist_current >= 0 and hist_prev < 0

    signals = {
        "buy_signal": macd_cross_up,
        "sell_signal": macd_cross_down,
        "hist_confirmation_buy": hist_neg_to_pos,
        "hist_confirmation_sell": hist_pos_to_neg,
        "macd_direction": "UP" if macd_current > macd_prev else "DOWN",
        "histogram_direction": "UP" if hist_current > hist_prev else "DOWN"
    }
    return signals


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


def calculate_dynamic_risk_percentage(data: pd.DataFrame,
                                      volume_score: float,
                                      ob_height_percent: float,
                                      ob_direction: int,
                                      ) -> dict:
    """
    Calculate dynamic risk percentage with momentum analysis
    """

    # Calculate momentum indicators
    momentum_data = calculate_price_momentum(data, lookback=20)

    # Initialize risk factors dictionary
    risk_factors = {
        'momentum': {'score': 0, 'weight': 0.3, 'contribution': 0},
        'volume': {'score': 0, 'weight': 0.2, 'contribution': 0},
        'ob_quality': {'score': 0, 'weight': 0.4, 'contribution': 0},
        'price_action': {'score': 0, 'weight': 0.1, 'contribution': 0}
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


def calculate_velocity(data: pd.DataFrame, lookback: int = 3) -> dict:
    """
    Calculate price and volume velocity with MA and RSI confirmations
    """
    # Calculate MAs
    data['EMA50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['EMA9'] = data['close'].ewm(span=5, adjust=False).mean()
    data['EMA21'] = data['close'].ewm(span=12, adjust=False).mean()
    data['EMA200'] = data['close'].ewm(span=200, adjust=False).mean()

    # Calculate RSI
    rsi = calculate_rsi(data)
    current_rsi = rsi.iloc[-1]
    # Get current values
    current_price = data['close'].iloc[-1]
    current_ema50 = data['EMA50'].iloc[-1]
    current_ema200 = data['EMA200'].iloc[-1]
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
        'above_ema50': current_price > current_ema50,
        'ema_crossover': ema_crossover,
        'rsi_above_50': current_rsi > 50,
        'rsi_oversold': current_rsi < 30,
        'rsi_overbought': current_rsi > 70
    }

    signal_macd = calculate_macd(data)

    # Generate signals based on conditions
    signals = []
    if conditions['above_ema50'] and conditions['rsi_above_50']:
        signals.append("Price above EMA50 with bullish RSI")
    if conditions['ema_crossover']:
        signals.append("EMA9 crossed above EMA21")
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
            'above_ema50': conditions['above_ema50'],
            'ema_crossover': conditions['ema_crossover'],
            'current_price': current_price,
            'ema200': current_ema200,
            'ema50': current_ema50,
            'ema9': current_ema9,
            'ema21': current_ema21
        },
        'rsi_analysis': {
            'current': current_rsi,
            'above_50': conditions['rsi_above_50'],
            'oversold': conditions['rsi_oversold'],
            'overbought': conditions['rsi_overbought']
        },
        'signals': signals,
        'macd_signals': signal_macd
    }


def analyze_trading_setup(data, ignore_old_ob=True):
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
    ob_results = smc.ob(data, type_orderblock='pivot_volume')

    # Add velocity analysis
    velocity = calculate_velocity(data, 30)

    volume_analysis = analyze_volume_patterns(data, lookback=30)

    # Analyze each order block
    for i in range(len(ob_results)):
        ob_volume = ob_results[i]['volume']
        ob_direction = ob_results[i]["direction"]
        ob_top = ob_results[i]["top"]
        ob_bottom = ob_results[i]["bottom"]
        ob_height = ob_top - ob_bottom
        ob_height_percent = (ob_height/current_price) * 100

        # Calculate volume metrics
        volume_ratio = ob_volume / avg_volume
        volume_score = min(100, int((volume_ratio - 1) * 50))

        # Get risk assessment
        risk_assessment = calculate_dynamic_risk_percentage(
            data=data,
            volume_score=volume_score,
            ob_height_percent=ob_height_percent,
            ob_direction=ob_direction,
        )

        # Determine setup type based on OB direction, trend, and volume pressure
        if ob_direction == "bullish":  # Bullish OB
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
        setup.update({
            'setup_type': setup_type,
            'setup_strength': setup_strength,
            'setup_description': setup_classification[setup_type],
            'position_type': 'LONG' if setup_type.startswith('LONG') else 'SHORT',
        })

        # Add velocity analysis to each setup
        trade_setups.append(setup)

    # Sort setups by:
    # 2. Setup quality (highest first)
    # 3. Volume score (highest first)
    trade_setups.sort(key=lambda x: (
        # Primary: closest to price
        -x['setup_quality'],          # Secondary: highest quality
        -x['volume_score']            # Tertiary: highest volume
    ))

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
