from binance.client import Client
from binance.enums import (
    SIDE_BUY, SIDE_SELL,
    ORDER_TYPE_LIMIT,
    FUTURE_ORDER_TYPE_STOP_MARKET,
    FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
    TIME_IN_FORCE_GTC
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from smartmoneyconcepts.smc import smc
import plotly.graph_objects as go


class PivotPointsStrategy:
    def __init__(self, api_key="", api_secret=""):
        self.client = Client(api_key, api_secret)

    def get_historical_data(self, symbol='BTCUSDT', interval='15m', lookback_days=2):
        start_time = datetime.now() - timedelta(days=lookback_days)
        start_ms = int(start_time.timestamp() * 1000)

        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_ms,
            limit=1000
        )

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df.set_index('timestamp', inplace=True)
        return df

    def analyze_setup(self, data, method='traditional', timeframe='m15', min_volume_percentile=80):
        """Analyze potential trading setups based on pivot points, EMAs, and volume"""
        trade_setups = []

        # Calculate pivot points
        pivot_data = smc.pivot_points(
            data, method=method, timeframe=timeframe, levels=3)

        # Calculate EMAs for trend
        ema_34 = data['close'].ewm(span=34, adjust=False).mean()
        ema_89 = data['close'].ewm(span=89, adjust=False).mean()

        # Get current price and previous prices
        current_price = float(data['close'].iloc[-1])
        prev_price = float(data['close'].iloc[-2])

        # Calculate trend change signals
        trend_change = None

        # Bullish trend change conditions
        if (ema_34.iloc[-2] <= ema_89.iloc[-2] and  # Previous EMA34 below EMA89
            # Current EMA34 crosses above EMA89
            ema_34.iloc[-1] > ema_89.iloc[-1] and
                current_price > ema_34.iloc[-1]):         # Price above EMA34
            trend_change = 'BULLISH_CONVERSION'

        # Bearish trend change conditions
        elif (ema_34.iloc[-2] >= ema_89.iloc[-2] and  # Previous EMA34 above EMA89
              # Current EMA34 crosses below EMA89
              ema_34.iloc[-1] < ema_89.iloc[-1] and
              current_price < ema_34.iloc[-1]):        # Price below EMA34
            trend_change = 'BEARISH_CONVERSION'

        # Current trend state
        trend = 'UPTREND' if ema_34.iloc[-1] > ema_89.iloc[-1] else 'DOWNTREND'

        print(f"\nMarket Analysis:")
        print(f"Current Price: {current_price:.2f}")
        print(f"Current Trend: {trend}")
        if trend_change:
            print(f"Trend Change Signal: {trend_change}")
        print(f"EMA 34: {ema_34.iloc[-1]:.2f}")
        print(f"EMA 89: {ema_89.iloc[-1]:.2f}")
        print("-" * 50)

        # Get latest pivot levels
        pivot = pivot_data['pivot'].iloc[-1]
        r1 = pivot_data['r1'].iloc[-1] if 'r1' in pivot_data.columns else None
        r2 = pivot_data['r2'].iloc[-1] if 'r2' in pivot_data.columns else None
        s1 = pivot_data['s1'].iloc[-1] if 's1' in pivot_data.columns else None
        s2 = pivot_data['s2'].iloc[-1] if 's2' in pivot_data.columns else None

        # Calculate volume conditions
        volume_threshold = data['volume'].quantile(min_volume_percentile/100)
        current_volume = data['volume'].iloc[-1]

        # Long Setup Conditions
        if ((trend == 'UPTREND' or trend_change == 'BULLISH_CONVERSION') and
                current_volume > volume_threshold):

            # Price between S1 and Pivot with trend confirmation
            if s1 and pivot and s1 < current_price < pivot:
                trade_setups.append({
                    'type': 'LONG',
                    'entry': current_price,
                    'stop_loss': s1 * 0.99,  # Stop below S1
                    'take_profit': r1 if r1 else pivot * 1.02,
                    'volume': current_volume,
                    'trend_signal': trend_change if trend_change else trend,
                    'pivot_level': 'S1-P'
                })

            # Aggressive long at S2 level
            elif s2 and current_price < s1 and current_price > s2:
                trade_setups.append({
                    'type': 'LONG',
                    'entry': current_price,
                    'stop_loss': s2 * 0.99,  # Stop below S2
                    'take_profit': s1 * 1.02,
                    'volume': current_volume,
                    'trend_signal': trend_change if trend_change else trend,
                    'pivot_level': 'S2-S1'
                })

        # Short Setup Conditions
        elif ((trend == 'DOWNTREND' or trend_change == 'BEARISH_CONVERSION') and
              current_volume > volume_threshold):

            # Price between Pivot and R1 with trend confirmation
            if r1 and pivot and pivot < current_price < r1:
                trade_setups.append({
                    'type': 'SHORT',
                    'entry': current_price,
                    'stop_loss': r1 * 1.01,  # Stop above R1
                    'take_profit': s1 if s1 else pivot * 0.98,
                    'volume': current_volume,
                    'trend_signal': trend_change if trend_change else trend,
                    'pivot_level': 'P-R1'
                })

            # Aggressive short at R2 level
            elif r2 and current_price > r1 and current_price < r2:
                trade_setups.append({
                    'type': 'SHORT',
                    'entry': current_price,
                    'stop_loss': r2 * 1.01,  # Stop above R2
                    'take_profit': r1 * 0.98,
                    'volume': current_volume,
                    'trend_signal': trend_change if trend_change else trend,
                    'pivot_level': 'R1-R2'
                })

        # Print setup information
        for setup in trade_setups:
            print(f"\nFound {setup['type']} Setup:")
            print(f"Pivot Level: {setup['pivot_level']}")
            print(f"Entry: {setup['entry']:.2f}")
            print(f"Stop Loss: {setup['stop_loss']:.2f}")
            print(f"Take Profit: {setup['take_profit']:.2f}")
            print(f"Trend Signal: {setup['trend_signal']}")
            print(f"Volume: {setup['volume']:.2f}")

        # Sort setups by trend change signals and volume
        trade_setups.sort(key=lambda x: (
            x['trend_signal'].startswith(
                'BULLISH') or x['trend_signal'].startswith('BEARISH'),
            x['volume']
        ), reverse=True)

        return trade_setups

    def place_orders(self, trade_setup, leverage=5, risk_percent=1):
        """Place orders with proper position sizing"""
        try:
            # Set leverage
            self.client.futures_change_leverage(
                symbol='BTCUSDT', leverage=leverage)

            # Calculate position size
            account_info = self.client.futures_account()
            available_balance = float(account_info['availableBalance'])
            risk_amount = available_balance * (risk_percent / 100)
            risk_per_coin = abs(
                trade_setup['entry'] - trade_setup['stop_loss'])
            position_size = (risk_amount / risk_per_coin) * leverage

            print(f"\nPlacing {trade_setup['type']} Order:")
            print(f"Entry: {trade_setup['entry']:.2f}")
            print(f"Stop Loss: {trade_setup['stop_loss']:.2f}")
            print(f"Take Profit: {trade_setup['take_profit']:.2f}")
            print(f"Position Size: {position_size:.4f}")
            print(f"Pivot Level: {trade_setup['pivot_level']}")

            # Place main limit order
            limit_order = self.client.futures_create_order(
                symbol='BTCUSDT',
                side=SIDE_BUY if trade_setup['type'] == 'LONG' else SIDE_SELL,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=position_size,
                price=trade_setup['entry']
            )

            # Place stop loss
            sl_order = self.client.futures_create_order(
                symbol='BTCUSDT',
                side=SIDE_SELL if trade_setup['type'] == 'LONG' else SIDE_BUY,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=trade_setup['stop_loss'],
                quantity=position_size,
                reduceOnly=True
            )

            # Place take profit
            tp_order = self.client.futures_create_order(
                symbol='BTCUSDT',
                side=SIDE_SELL if trade_setup['type'] == 'LONG' else SIDE_BUY,
                type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                stopPrice=trade_setup['take_profit'],
                quantity=position_size,
                reduceOnly=True
            )

            print("Orders placed successfully!")
            return True

        except Exception as e:
            print(f"Error placing orders: {str(e)}")
            return False


def main():
    # Initialize strategy
    strategy = PivotPointsStrategy()

    # Get historical data
    data = strategy.get_historical_data(
        symbol='BTCUSDT',
        interval='15m',
        lookback_days=2
    )

    # Analyze setups
    trade_setups = strategy.analyze_setup(
        data=data,
        method='traditional',
        timeframe='m15',
        min_volume_percentile=80
    )

    # Execute trades if setups found
    if trade_setups:
        print(f"\nFound {len(trade_setups)} potential trade setups")
        for setup in trade_setups:
            strategy.place_orders(
                trade_setup=setup,
                leverage=5,
                risk_percent=1
            )
    else:
        print("\nNo valid trade setups found")


if __name__ == "__main__":
    main()
