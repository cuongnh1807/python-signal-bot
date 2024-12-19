from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
from smartmoneyconcepts import smc
from strategy import calculate_rsi, find_closest_signal, generate_signals


def get_binance_data(symbol='BTCUSDT', interval='15m', lookback='1 day ago UTC'):
    # Initialize Binance client - no API keys needed for public data
    client = Client()

    # Get historical klines/candlestick data
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=lookback
    )

    # Create DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    # Convert string values to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Keep only the OHLCV columns
    df = df[['open', 'high', 'low', 'close', 'volume']]
    current_price = df['close'].iloc[-1]

    return df, current_price


def main():

    # Get 15m data
    df_15m, current_price = get_binance_data(
        interval='15m', lookback='1 day ago UTC')
    print("\n15m BTC Data:")
    df_15m = generate_signals(df_15m)
    print(df_15m['volume'].iloc[-1])

    # Calculate swing highs and lows
    swing_highs_lows = smc.swing_highs_lows(df_15m.tail(300), swing_length=5)

    # Calculate Fibonacci levels
    fib_levels = smc.fibonacci_retracement(df_15m, swing_highs_lows)

    # Print Fibonacci levels
    print("\nFibonacci Retracement Levels:")
    print(fib_levels)

    # Get 1h data

    result = find_closest_signal(df_15m, current_price)

    current_rsi = (calculate_rsi(df_15m, rsi_length=14, ma_type="SMA")
                   ).iloc[-1]  # Get the last RSI value
    print(f"\nCurrent RSI: {current_rsi:.2f}")
    print(result)

    # btc_dominance = float(btc_dom['markPrice'])


if __name__ == "__main__":
    main()
