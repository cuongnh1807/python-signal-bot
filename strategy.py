from smartmoneyconcepts.smc import smc
from datetime import datetime, timedelta
import pandas as pd
from typing import Union, Dict
from helpers.price import calculate_price_momentum
from indicators.rsi import calculate_macd, calculate_rsi
from indicators.candles import analyze_candle_volume

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


def analyze_volume_patterns(data: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Analyze volume patterns with RSI integration
    This function now uses the comprehensive analyze_candle_volume

    Parameters:
    -----------
    data: DataFrame with price and volume data
    lookback: Number of periods to look back

    Returns:
    --------
    Dict with volume analysis information
    """
    # Use the enhanced analyze_candle_volume function instead
    analysis = analyze_candle_volume(
        df=data, current_index=len(data)-1, lookback=lookback)

    # The analysis object already contains all the required fields
    # from the original analyze_volume_patterns
    return analysis


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


def analyze_trading_setup(data, lookback_volume: int = 30):
    """
    Analyze trading setups and calculate order percentages for active order blocks only
    """
    trade_setups = []

    # Calculate EMAs for trend
    ema_34 = data['close'].ewm(span=34, adjust=False).mean()
    ema_89 = data['close'].ewm(span=89, adjust=False).mean()

    # Get current price and volume metrics
    current_price = float(data['close'].iloc[-1])
    avg_volume = data['volume'].tail(lookback_volume).mean()
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
    ob_results = smc.ob(data, type_orderblock='sensitive')

    # Add velocity analysis
    velocity = calculate_velocity(data, 20)

    volume_analysis = analyze_volume_patterns(data, lookback=20)

    # Analyze each order block
    for i in range(len(ob_results)):
        ob_volume = ob_results[i]['volume']
        ob_direction = ob_results[i]["direction"]
        ob_top = ob_results[i]["top"]
        ob_bottom = ob_results[i]["bottom"]
        ob_height = ob_top - ob_bottom

        # Calculate volume metrics
        volume_ratio = ob_volume / avg_volume
        volume_score = min(100, int((volume_ratio - 1) * 50))

        setup_quality = ob_results[i].get('setup_quality', 0)
        warnings = ob_results[i].get('warnings', [])
        entry_quality = ob_results[i].get('entry_quality', 'Unknown')

        if setup_quality >= 80:
            risk_percentage = 0.5
        elif setup_quality >= 65:
            risk_percentage = 0.7
        elif setup_quality >= 50:
            risk_percentage = 0.9
        else:
            risk_percentage = 1.0

        # Determine setup type based on OB direction, trend, and volume pressure
        if ob_direction == 1:  # Bullish OB
            if current_trend == 'DOWNTREND':
                if volume_analysis['analysis']['pressure'] in ['Strong Buying', 'Moderate Buying']:
                    setup_type = "BOS"  # Break of Structure
                    setup_strength = 'Strong'
                else:
                    setup_type = "CHoCH"  # Change of Character
                    setup_strength = 'Moderate'
            else:  # UPTREND
                if volume_analysis['analysis']['pressure'] in ['Strong Buying Climax', 'Moderate Buying Climax']:
                    setup_type = "CONTINUATION"  # Strong Continuation
                    setup_strength = 'Strong'
                else:
                    setup_type = "CONTINUATION"  # Pullback treated as Continuation
                    setup_strength = 'Moderate'
        else:  # Bearish OB
            if current_trend == 'UPTREND':
                if volume_analysis['analysis']['pressure'] in ['Strong Selling Climax', 'Moderate Selling Climax']:
                    setup_type = "BOS"  # Break of Structure
                    setup_strength = 'Strong'
                else:
                    setup_type = "CHoCH"  # Change of Character
                    setup_strength = 'Moderate'
            else:  # DOWNTREND
                if volume_analysis['analysis']['pressure'] in ['Strong Selling Climax', 'Moderate Selling Climax']:
                    setup_type = "CONTINUATION"  # Strong Continuation
                    setup_strength = 'Strong'
                else:
                    setup_type = "CONTINUATION"  # Pullback treated as Continuation
                    setup_strength = 'Moderate'

        # Add leverage based on risk
        if setup_quality >= 80:
            max_leverage = 30  # More conservative setups allow higher leverage
        elif setup_quality >= 65:
            max_leverage = 50  # Moderate risk setups
        else:
            max_leverage = 70  # Higher risk setups get limited leverage

        suggested_leverage = min(max_leverage, int(1 / risk_percentage * 50))

        # Create setup dictionary
        setup = {
            'current_price': current_price,
            'type': setup_type,
            'atr': ob_results[i]['atr'],
            'strength': ob_results[i]['strength'],
            'ob_direction': 'Bullish' if ob_direction == 1 else 'Bearish',
            'current_trend': current_trend,
            'ob_levels': {'top': ob_top, 'bottom': ob_bottom},
            'volume_score': volume_score,
            'volume_ratio': volume_ratio,
            'ob_volume': ob_volume,
            'ob_volume_ratio': ob_volume / avg_volume,
            'risk_percentage': risk_percentage,
            'suggested_leverage': suggested_leverage,
            'setup_quality': setup_quality,
            'entry_quality': entry_quality,
            'warning_messages': warnings,
            'risk_rating': 'Low' if risk_percentage <= 0.5 else
            'Moderate' if risk_percentage <= 0.75 else 'High',
            'effective_risk': risk_percentage * suggested_leverage
        }
        setup.update({
            'setup_type': setup_type,
            'setup_strength': setup_strength,
            'position_type': 'LONG' if ob_direction == 1 else 'SHORT',
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
