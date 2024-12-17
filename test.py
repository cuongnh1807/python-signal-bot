import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from datetime import datetime, timedelta
import mplfinance as mpf
from smartmoneyconcepts.smc import smc
import plotly.graph_objects as go
import pytz
from binance.enums import (
    SIDE_BUY, SIDE_SELL,
    ORDER_TYPE_LIMIT,
    FUTURE_ORDER_TYPE_STOP_MARKET,
    FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
    TIME_IN_FORCE_GTC
)
from dotenv import load_dotenv
import os


class BinanceDataFetcher:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Get API credentials from environment variables
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')

        # Initialize client with API credentials
        self.client = Client(api_key, api_secret)

    def get_historical_klines(self, symbol: str, interval: str, start_time: datetime) -> pd.DataFrame:
        """Fetch historical klines/candlestick data"""
        start_ms = int(start_time.timestamp() * 1000)

        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_ms,
            limit=1000
        )

        # Ensure column names are lowercase
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df.set_index('timestamp', inplace=True)

        # Verify the DataFrame has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return df


def add_OB(fig, df, ob_data):
    def format_volume(volume):
        if volume >= 1e12:
            return f"{volume / 1e12:.3f}T"
        elif volume >= 1e9:
            return f"{volume / 1e9:.3f}B"
        elif volume >= 1e6:
            return f"{volume / 1e6:.3f}M"
        elif volume >= 1e3:
            return f"{volume / 1e3:.3f}k"
        else:
            return f"{volume:.2f}"

    # Ensure DataFrame index is timezone-aware
    df.index = df.index.tz_localize('UTC')

    for i in range(len(ob_data["OB"])):
        if ob_data["OB"][i] == 1:
            x1 = int(
                ob_data["MitigatedIndex"][i]
                if ob_data["MitigatedIndex"][i] != 0
                else len(df) - 1
            )
            print("Bullish OB, Price: ",
                  ob_data["Bottom"][i], " - ", ob_data["Top"][i], "percentage: ", ob_data["Percentage"][i], "the width is: ", df.index[i],  df.index[x1], "Volume BTC: ", ob_data["OBVolume"][i])
            # Convert current time to UTC for comparison
            current_time = datetime.now().astimezone().astimezone(tz=None)
            if (current_time > df.index[i] and current_time < df.index[x1] + timedelta(minutes=15)):
                fig.add_shape(
                    type="rect",
                    x0=df.index[i],
                    y0=ob_data["Bottom"][i],
                    x1=df.index[x1],
                    y1=ob_data["Top"][i],
                    line=dict(color="Green"),
                    fillcolor="Green",
                    opacity=0.2,
                    name="Bullish OB",
                    legendgroup="bullish ob",
                    showlegend=True,
                )

                if ob_data["MitigatedIndex"][i] > 0:
                    x_center = df.index[int(
                        i + (ob_data["MitigatedIndex"][i] - i) / 2)]
                else:
                    x_center = df.index[int(i + (len(df) - i) / 2)]

                y_center = (ob_data["Bottom"][i] + ob_data["Top"][i]) / 2
                volume_text = format_volume(ob_data["OBVolume"][i])
                annotation_text = f'OB: {volume_text} ({ob_data["Percentage"][i]}%)'

                fig.add_annotation(
                    x=x_center,
                    y=y_center,
                    xref="x",
                    yref="y",
                    align="center",
                    text=annotation_text,
                    font=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                    showarrow=False,
                )

    for i in range(len(ob_data["OB"])):
        if ob_data["OB"][i] == -1:
            print("Bearish OB, Price: ",
                  ob_data["Bottom"][i], " - ", ob_data["Top"][i], "percentage: ", ob_data["Percentage"][i],  "the width is: ", df.index[i],  df.index[x1], "Volume BTC: ", ob_data["OBVolume"][i])
            x1 = int(
                ob_data["MitigatedIndex"][i]
                if ob_data["MitigatedIndex"][i] != 0
                else len(df) - 1
            )

            # convert time utc to local time
            current_time = datetime.now().astimezone().astimezone(tz=None)
            if (current_time > df.index[i] and current_time < df.index[x1] + timedelta(minutes=15)):
                fig.add_shape(
                    type="rect",
                    x0=df.index[i],
                    y0=ob_data["Bottom"][i],
                    x1=df.index[x1],
                    y1=ob_data["Top"][i],
                    line=dict(color="Red"),
                    fillcolor="Red",
                    opacity=0.2,
                    name="Bearish OB",
                    legendgroup="bearish ob",
                    showlegend=True,
                )

                if ob_data["MitigatedIndex"][i] > 0:
                    x_center = df.index[int(
                        i + (ob_data["MitigatedIndex"][i] - i) / 2)]
                else:
                    x_center = df.index[int(i + (len(df) - i) / 2)]

                y_center = (ob_data["Bottom"][i] + ob_data["Top"][i]) / 2
                volume_text = format_volume(ob_data["OBVolume"][i])
                annotation_text = f'OB: {volume_text} ({ob_data["Percentage"][i]}%)'

                fig.add_annotation(
                    x=x_center,
                    y=y_center,
                    xref="x",
                    yref="y",
                    align="center",
                    text=annotation_text,
                    font=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                    showarrow=False,
                )
    return fig


def add_liquidity(fig, df, liquidity_data):
    def format_liquidity(liquidity):
        if liquidity >= 1e12:
            return f"{liquidity / 1e12:.3f}T"
        elif liquidity >= 1e9:
            return f"{liquidity / 1e9:.3f}B"
        elif liquidity >= 1e6:
            return f"{liquidity / 1e6:.3f}M"
        elif liquidity >= 1e3:
            return f"{liquidity / 1e3:.3f}k"
        else:
            return f"{liquidity:.2f}"

    # Handle timezone conversion safely
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        elif str(df.index.tz) != 'UTC':
            df.index = df.index.tz_convert('UTC')
    except Exception as e:
        print(f"Timezone conversion warning: {str(e)}")
        pass

    for i in range(len(liquidity_data["Liquidity"])):
        if liquidity_data["Liquidity"][i] == 1:
            x1 = int(
                # Using "End" instead of "MitigatedIndex"
                liquidity_data["End"][i]
                if not pd.isna(liquidity_data["End"][i])
                else len(df) - 1
            )
            print("Bullish Liquidity Found:")
            print(f"Level: {liquidity_data['Level'][i]:.2f}")
            print(f"Time Range: {df.index[i]} to {df.index[x1]}")

            # Convert current time to UTC for comparison
            current_time = datetime.now().astimezone(tz=None)
            if (current_time > df.index[i] and current_time < df.index[x1] + timedelta(minutes=15)):
                fig.add_shape(
                    type="rect",
                    x0=df.index[i],
                    # Slight offset for visibility
                    y0=liquidity_data["Level"][i] * 0.999,
                    x1=df.index[x1],
                    y1=liquidity_data["Level"][i] * 1.001,
                    line=dict(color="Green"),
                    fillcolor="Green",
                    opacity=0.2,
                    name="Bullish Liquidity",
                    legendgroup="bullish liquidity",
                    showlegend=True,
                )

                # Add label
                x_center = df.index[int(i + (x1 - i) / 2)]
                fig.add_annotation(
                    x=x_center,
                    y=liquidity_data["Level"][i],
                    text=f"BLQ {liquidity_data['Level'][i]:.0f}",
                    font=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                    showarrow=False,
                )

        if liquidity_data["Liquidity"][i] == -1:
            x1 = int(
                # Using "End" instead of "MitigatedIndex"
                liquidity_data["End"][i]
                if not pd.isna(liquidity_data["End"][i])
                else len(df) - 1
            )
            print("Bearish Liquidity Found:")
            print(f"Level: {liquidity_data['Level'][i]:.2f}")
            print(f"Time Range: {df.index[i]} to {df.index[x1]}")

            current_time = datetime.now().astimezone(tz=None)
            if (current_time > df.index[i] and current_time < df.index[x1] + timedelta(minutes=15)):
                fig.add_shape(
                    type="rect",
                    x0=df.index[i],
                    # Slight offset for visibility
                    y0=liquidity_data["Level"][i] * 0.999,
                    x1=df.index[x1],
                    y1=liquidity_data["Level"][i] * 1.001,
                    line=dict(color="Red"),
                    fillcolor="Red",
                    opacity=0.2,
                    name="Bearish Liquidity",
                    legendgroup="bearish liquidity",
                    showlegend=True,
                )

                # Add label
                x_center = df.index[int(i + (x1 - i) / 2)]
                fig.add_annotation(
                    x=x_center,
                    y=liquidity_data["Level"][i],
                    text=f"BLQ {liquidity_data['Level'][i]:.0f}",
                    font=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                    showarrow=False,
                )

    return fig


def calculate_pivot_points(data: pd.DataFrame) -> dict:
    """
    Calculate standard pivot points including support and resistance levels

    Formula:
    Pivot Point (PP) = (High + Low + Close) / 3

    Resistance levels:
    R1 = (2 × PP) - Low
    R2 = PP + (High - Low)
    R3 = High + 2 × (PP - Low)

    Support levels:
    S1 = (2 × PP) - High
    S2 = PP - (High - Low)
    S3 = Low - 2 × (High - PP)
    """
    high = data['high'].iloc[-1]
    low = data['low'].iloc[-1]
    close = data['close'].iloc[-1]

    # Calculate Pivot Point
    pp = (high + low + close) / 3

    # Calculate Resistance Levels
    r1 = (2 * pp) - low
    r2 = pp + (high - low)
    r3 = high + 2 * (pp - low)

    # Calculate Support Levels
    s1 = (2 * pp) - high
    s2 = pp - (high - low)
    s3 = low - 2 * (high - pp)

    return {
        'PP': pp,
        'R1': r1,
        'R2': r2,
        'R3': r3,
        'S1': s1,
        'S2': s2,
        'S3': s3
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
    print(f"\nMarket Analysis:")
    print(f"Current Price: {current_price:.2f}")
    print(f"Current Trend: {current_trend}")
    print("-" * 50)

    # Get order blocks
    ob_results = smc.ob(data, swing_hl)

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

            setup_type = ""
            if current_trend == 'UPTREND':
                if ob_direction == -1:
                    setup_type = "REVERSAL_SHORT"
                else:
                    setup_type = "CONTINUATION_LONG"
            else:  # DOWNTREND
                if ob_direction == 1:
                    setup_type = "REVERSAL_LONG"
                else:
                    setup_type = "CONTINUATION_SHORT"

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
                entry_price=entry_prices['moderate'],
                stop_loss=stop_loss,
                volume_score=volume_score,
                ob_height_percent=ob_height_percent,
                current_price=current_price,
                ob_direction=ob_direction
            )

            # Get the risk percentage from the assessment
            risk_percentage = risk_assessment['risk_percentage']

            # Adjust leverage based on risk
            if risk_percentage <= 0.5:
                max_leverage = 20  # More conservative setups allow higher leverage
            elif risk_percentage <= 0.75:
                max_leverage = 15  # Moderate risk setups
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

    if not trade_setups:
        print("\nNo valid active setups found")
    else:
        print(f"\nFound {len(trade_setups)} potential active setups")
        print("\nSetups Summary (Sorted by proximity to current price):")
        for i, setup in enumerate(trade_setups, 1):
            print(f"\n{'='*50}")
            print(f"🎯 Trade Setup Analysis {i}:")
            print(f"Type: {setup['type']}")
            print(f"Setup Quality: {setup['setup_quality']}/100")
            print(f"Entry Quality: {setup['entry_quality']}")
            print(f"Risk Percentage: {setup['risk_percentage']:.3f}%")
            print(f"Recommended Leverage: {setup['suggested_leverage']}x")
            print("\n📊 Risk Factors:")
            for name, factor in setup['risk_factors'].items():
                print(f"{name.title()}: {factor['score']:.1f}/100 "
                      f"(Contributing {factor['contribution']:.3f}%)")
            print("\n⚠️ Warnings:")
            for warning in setup['warning_messages']:
                print(warning)
            print(f"\n📝 Recommendation: {setup['trade_recommendation']}")
            print(
                f"Distance from current price: {setup['price_distance']:.2f}%")
            print(f"Entry price: {setup['entry_prices']['moderate']:.2f}")
            print(f"Current price: {current_price:.2f}")
            print(f"{'='*50}")

    return trade_setups


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


def place_strategic_orders(client, trade_setups, max_positions=3, risk_percent=1, leverage=30):
    """Place orders based on the analyzed setups"""
    try:
        client.futures_change_leverage(
            symbol='BTCUSDT', leverage=leverage)
        # Get account info
        account_info = client.futures_account()
        available_balance = float(account_info['availableBalance'])

        # Get current positions
        positions = client.futures_position_information()
        active_positions = [
            p for p in positions if float(p['positionAmt']) != 0]

        if len(active_positions) >= max_positions:
            print("Maximum positions reached")
            return

        # Take only the best setups based on available slots
        available_slots = max_positions - len(active_positions)
        best_setups = trade_setups[:available_slots]

        for setup in best_setups:
            # Calculate position size based on risk
            risk_amount = available_balance * (risk_percent / 100)
            risk_per_coin = abs(setup['entry'] - setup['stop_loss'])
            position_size = risk_amount / risk_per_coin

            print(f"\nPlacing {setup['type']} Order:")
            print(f"Entry: {setup['entry']:.2f}")
            print(f"Stop Loss: {setup['stop_loss']:.2f}")
            print(f"Take Profit: {setup['take_profit']:.2f}")
            print(f"Position Size: {position_size:.4f}")

            # Place the main limit order
            limit_order = client.futures_create_order(
                symbol='BTCUSDT',
                side=SIDE_BUY if setup['type'] == 'LONG' else SIDE_SELL,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=position_size,
                price=setup['entry']
            )

            # Place stop loss
            sl_order = client.futures_create_order(
                symbol='BTCUSDT',
                side=SIDE_SELL if setup['type'] == 'LONG' else SIDE_BUY,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=setup['stop_loss'],
                quantity=position_size,
                timeInForce=TIME_IN_FORCE_GTC,
                workingType='MARK_PRICE',
                priceProtect=True,
                reduceOnly=True
            )

            # Place take profit
            tp_order = client.futures_create_order(
                symbol='BTCUSDT',
                side=SIDE_SELL if setup['type'] == 'LONG' else SIDE_BUY,
                type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=setup['take_profit'],
                quantity=position_size,
                timeInForce=TIME_IN_FORCE_GTC,
                workingType='MARK_PRICE',
                priceProtect=True,
                reduceOnly=True
            )

            print("Orders placed successfully!")
            print("-" * 50)

    except Exception as e:
        print(f"Error placing orders: {str(e)}")


def plot_analysis(data: pd.DataFrame, ob_results: pd.DataFrame, trade_setups: list):
    """Plot the analysis results with order blocks and moderate entry points"""

    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close']
    )])

    # Add order blocks
    for i in range(len(ob_results)):
        if pd.notna(ob_results["OB"][i]):
            color = "Green" if ob_results["OB"][i] == 1 else "Red"
            x1 = int(ob_results["MitigatedIndex"][i]
                     if ob_results["MitigatedIndex"][i] != 0 else len(data) - 1)

            # Add OB rectangle
            fig.add_shape(
                type="rect",
                x0=data.index[i],
                y0=ob_results["Bottom"][i],
                x1=data.index[x1],
                y1=ob_results["Top"][i],
                line=dict(color=color),
                fillcolor=color,
                opacity=0.2
            )

    # Add moderate entry points
    if trade_setups:
        for setup in trade_setups:
            # Only plot moderate entry
            moderate_price = setup['entry_prices']['moderate']
            marker_color = 'green' if 'LONG' in setup['type'] else 'red'

            fig.add_trace(go.Scatter(
                x=[data.index[-1]],
                y=[moderate_price],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if 'LONG' in setup['type'] else 'triangle-down',
                    size=12,
                    color=marker_color
                ),
                showlegend=False
            ))

    # Update layout
    fig.update_layout(
        title='Market Analysis with Order Blocks and Moderate Entries',
        yaxis_title='Price',
        xaxis_title='Date',
        template='plotly_dark'
    )

    fig.show()


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


def calculate_suggested_leverage(entry_percentage: float) -> int:
    """
    Calculate suggested leverage based on entry percentage
    More aggressive entries = higher leverage potential

    Parameters:
    - entry_percentage: How far into the OB the entry is placed

    Returns:
    - Suggested leverage (max 20x)

    Logic:
    - Aggressive entries (<=15%): 15-20x leverage
    - Moderate entries (15-30%): 10-15x leverage
    - Conservative entries (>30%): 5-10x leverage
    """
    if entry_percentage <= 15.0:  # Aggressive entries
        # Scale leverage from 15x to 20x based on how close to edge
        leverage = 20 - ((entry_percentage / 15.0) * 5)
    elif entry_percentage <= 30.0:  # Moderate entries
        # Scale leverage from 10x to 15x
        leverage = 15 - (((entry_percentage - 15.0) / 15.0) * 5)
    else:  # Conservative entries
        # Scale leverage from 5x to 10x
        leverage = 10 - (((entry_percentage - 30.0) / 20.0) * 5)

    # Ensure minimum leverage of 5x and maximum of 20x
    return max(5, min(20, int(leverage)))


def calculate_position_size(account_balance: float, risk_percentage: float, entry_price: float, stop_loss: float, leverage: int) -> float:
    """
    Calculate position size based on risk parameters and leverage

    Parameters:
    - account_balance: Total account balance
    - risk_percentage: Percentage of account willing to risk (e.g., 1.0 for 1%)
    - entry_price: Entry price level
    - stop_loss: Stop loss price level
    - leverage: Chosen leverage multiplier

    Returns:
    - Position size in base currency

    Formula:
    position_size = (account_balance * risk_percentage) / (price_difference / leverage)
    """
    # Calculate risk amount in dollars
    risk_amount = account_balance * (risk_percentage / 100)

    # Calculate price difference between entry and stop loss
    price_difference = abs(entry_price - stop_loss)

    # Calculate risk per coin with leverage
    risk_per_coin = price_difference / leverage

    # Calculate position size
    if risk_per_coin == 0:
        return 0

    position_size = risk_amount / risk_per_coin

    # Print calculation details for verification
    print(f"  Risk Amount: ${risk_amount:.2f}")
    print(f"  Price Difference: ${price_difference:.2f}")
    print(f"  Risk per Coin: ${risk_per_coin:.2f}")

    return position_size


def calculate_dynamic_risk_percentage(
    entry_price: float,
    stop_loss: float,
    volume_score: float,
    ob_height_percent: float,
    current_price: float,
    ob_direction: int,
) -> dict:
    """
    Calculate risk percentage based on market conditions and setup quality

    Parameters:
    - entry_price: Planned entry price
    - stop_loss: Stop loss level
    - volume_score: Volume analysis score (0-100)
    - ob_height_percent: Order block height as percentage of price
    - current_price: Current market price
    - ob_direction: Order block direction (1 for bullish, -1 for bearish)

    Returns:
    - Dictionary containing risk analysis and trade recommendations
    """
    # Calculate price volatility
    price_distance = abs(current_price - entry_price) / current_price
    stop_distance = abs(entry_price - stop_loss) / entry_price

    # Setup Quality Score (0-100)
    setup_quality = calculate_setup_quality(
        volume_score=volume_score,
        price_distance=price_distance,
        ob_height_percent=ob_height_percent,
        stop_distance=stop_distance
    )

    # Risk Assessment
    risk_assessment = {
        'setup_quality': setup_quality,
        'risk_percentage': 0,
        'recommended_leverage': 0,
        'trade_recommendation': '',
        'warning_messages': [],
        'entry_quality': '',
        'risk_factors': {}
    }

    # Base Risk Calculation
    if setup_quality >= 80:
        base_risk = 1.0  # High-quality setup
        risk_assessment['entry_quality'] = 'Excellent'
    elif setup_quality >= 65:
        base_risk = 0.75  # Good setup
        risk_assessment['entry_quality'] = 'Good'
    elif setup_quality >= 50:
        base_risk = 0.5  # Moderate setup
        risk_assessment['entry_quality'] = 'Moderate'
    else:
        base_risk = 0  # Poor setup
        risk_assessment['entry_quality'] = 'Poor'
        risk_assessment['trade_recommendation'] = 'DO NOT TRADE - Setup quality too low'
        return risk_assessment

    # Risk Factors Analysis
    risk_factors = {
        'volume': {
            'score': volume_score,
            'weight': 0.3,
            'contribution': (volume_score / 100) * 0.3
        },
        'price_distance': {
            'score': max(0, (1 - price_distance * 10) * 100),
            'weight': 0.3,
            'contribution': max(0, 0.3 * (1 - price_distance * 10))
        },
        'ob_height': {
            'score': max(0, (1 - ob_height_percent / 5) * 100),
            'weight': 0.2,
            'contribution': max(0, 0.2 * (1 - ob_height_percent / 5))
        },
        'trend_alignment': {
            'score': 100 if (
                (ob_direction == 1 and current_price > entry_price) or
                (ob_direction == -1 and current_price < entry_price)
            ) else 0,
            'weight': 0.2,
            'contribution': 0.2 if (
                (ob_direction == 1 and current_price > entry_price) or
                (ob_direction == -1 and current_price < entry_price)
            ) else 0
        }
    }

    # Calculate final risk percentage
    risk_percentage = base_risk * sum(factor['contribution']
                                      for factor in risk_factors.values())

    # Cap risk percentage
    risk_percentage = min(1.5, max(0.25, risk_percentage))

    # Determine recommended leverage based on risk
    if risk_percentage <= 0.5:
        recommended_leverage = 20
    elif risk_percentage <= 0.75:
        recommended_leverage = 15
    elif risk_percentage <= 1.0:
        recommended_leverage = 10
    else:
        recommended_leverage = 5

    # Generate trade recommendation
    if setup_quality >= 80:
        trade_recommendation = "STRONG ENTRY - High-quality setup"
    elif setup_quality >= 65:
        trade_recommendation = "GOOD ENTRY - Consider position size carefully"
    elif setup_quality >= 50:
        trade_recommendation = "MODERATE ENTRY - Reduce position size"
    else:
        trade_recommendation = "DO NOT TRADE - Wait for better setup"

    # Warning messages
    warnings = []
    if volume_score < 50:
        warnings.append("⚠️ Low volume - Consider reducing position size")
    if price_distance > 0.02:
        warnings.append("⚠️ Entry far from current price - Higher risk")
    if ob_height_percent > 3:
        warnings.append("⚠️ Large order block - Less precise entry")
    if stop_distance > 0.02:
        warnings.append("⚠️ Wide stop loss - Consider reducing position size")

    # Update risk assessment
    risk_assessment.update({
        'risk_percentage': round(risk_percentage, 3),
        'recommended_leverage': recommended_leverage,
        'trade_recommendation': trade_recommendation,
        'warning_messages': warnings,
        'risk_factors': risk_factors
    })

    # Print detailed analysis
    # print("\n🎯 Trade Setup Analysis:")
    # print(f"Setup Quality: {setup_quality}/100")
    # print(f"Entry Quality: {risk_assessment['entry_quality']}")
    # print(f"Risk Percentage: {risk_percentage:.3f}%")
    # print(f"Recommended Leverage: {recommended_leverage}x")
    # print("\n📊 Risk Factors:")
    # for name, factor in risk_factors.items():
    #     print(f"{name.title()}: {factor['score']:.1f}/100 "
    #           f"(Contributing {factor['contribution']:.3f}%)")
    # print("\n⚠️ Warnings:")
    # for warning in warnings:
    #     print(warning)
    # print(f"\n📝 Recommendation: {trade_recommendation}")

    return risk_assessment


def calculate_setup_quality(
    volume_score: float,
    price_distance: float,
    ob_height_percent: float,
    stop_distance: float
) -> float:
    """Calculate overall setup quality score"""

    # Volume quality (0-30 points)
    volume_quality = (volume_score / 100) * 30

    # Price distance quality (0-25 points)
    # Better score for entries closer to current price
    distance_quality = max(0, (1 - price_distance * 10) * 25)

    # Order block height quality (0-25 points)
    # Better score for more precise order blocks
    height_quality = max(0, (1 - ob_height_percent / 5) * 25)

    # Stop loss quality (0-20 points)
    # Better score for tighter stops
    stop_quality = max(0, (1 - stop_distance * 20) * 20)

    # Calculate total quality score (0-100)
    total_quality = (
        volume_quality +
        distance_quality +
        height_quality +
        stop_quality
    )

    return round(total_quality, 1)


def main():
    # Initialize Binance data fetcher
    fetcher = BinanceDataFetcher()

    # Get data for last 2 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=10)

    # Fetch BTCUSDT 15-minute data
    data = fetcher.get_historical_klines('BTCUSDT', '15m', start_time)

    current_price = float(
        fetcher.client.get_symbol_ticker(symbol="BTCUSDT")['price'])
    print(f"Current price: {current_price}")

    # Ensure column names are lowercase
    data.columns = [col.lower() for col in data.columns]

    # Verify required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate swing highs and lows
    swing_hl = smc.swing_highs_lows(data, swing_length=5)

    # Get order blocks
    ob_results = smc.ob(data, swing_hl)

    # Analyze trading setups
    trade_setups = analyze_trading_setup(
        data,
        swing_hl,
    )

    # Plot analysis with order blocks and trading setups
    plot_analysis(data, ob_results, trade_setups)

    if trade_setups:
        place_strategic_orders(
            client=fetcher.client,
            trade_setups=trade_setups,
            max_positions=3,
            risk_percent=1
        )
    else:
        print("No valid trade setups found")


if __name__ == "__main__":
    main()
