# import os
# from openai import OpenAI

# XAI_API_KEY = 'xai-YZWQumTiG1kNeYvCOqJFtWS9HCfHHxpsoEqMRLi9Yv3VXy3iMaV2q4TQFyQxouGcnOavfLleFFZJmPr8'
# client = OpenAI(
#     api_key=XAI_API_KEY,
#     base_url="https://api.x.ai/v1",
# )

# # Initialize conversation history
# messages = [
#     {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."}
# ]


# def chat_with_grok():
#     print("Welcome to Grok AI Chat! (Type 'quit' to exit)")
#     print("-" * 50)

#     while True:
#         # Get user input
#         user_input = input("\nYou: ").strip()

#         # Check if user wants to quit
#         if user_input.lower() in ['quit', 'exit', 'bye']:
#             print("\nGoodbye! Thanks for chatting!")
#             break

#         # Add user message to conversation history
#         messages.append({"role": "user", "content": user_input})

#         try:
#             # Get response from Grok
#             completion = client.chat.completions.create(
#                 model="grok-2-vision-latest",
#                 messages=messages
#             )

#             # Extract and print response
#             response = completion.choices[0].message.content
#             print("\nGrok:", response)

#             # Add assistant's response to conversation history
#             messages.append({"role": "assistant", "content": response})

#         except Exception as e:
#             print(f"\nError: {str(e)}")
#             messages.pop()  # Remove the last user message if there was an error


# if __name__ == "__main__":
#     chat_with_grok()
# Import necessary libraries
from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
import time
from ai_smc import SmartMoneyConcept, plot_order_blocks, generate_order_blocks_list
from ai_strategy_smc import SMCFuturesStrategy
# Initialize Binance client
# api_key = 'YOUR_API_KEY'       # Replace with your API key
# api_secret = 'YOUR_API_SECRET'  # Replace with your API secret
client = Client()

# Initialize SmartMoneyConcept analyzer
smc = SmartMoneyConcept(
    show_internals=True,
    show_structure=True,
    show_iob=True,
    show_ob=True,
    show_eq=True,
    show_fvg=True,
    swing_length=5,
    internal_swing_length=4,
    eq_len=3,
    eq_threshold=0.05,
    fvg_auto=True
)

# Initialize SMC Futures Strategy
strategy = SMCFuturesStrategy(
    smc_analyzer=smc,
    symbol="BTCUSDT",
    risk_per_trade=0.02,    # 2% risk per trade
    leverage=5,             # 5x leverage
    order_validity=24,      # Orders valid for 24 hours
    ob_proximity=0.03       # Consider OBs within 3% of current price
)


def fetch_latest_data(symbol, interval, lookback_hours=24 * 7):
    """Fetch the latest market data"""
    end_time = int(time.time() * 1000)
    start_time = int(
        (datetime.now() - timedelta(hours=lookback_hours)).timestamp() * 1000)

    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_time,
        end_str=end_time
    )

    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    # Convert string values to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df[['open', 'high', 'low', 'close', 'volume']]


def run_strategy_loop(interval='15m', check_interval_minutes=15):
    """Run the strategy in a loop"""
    while True:
        try:
            # 1. Fetch latest data
            print(f"Fetching latest {strategy.symbol} data...")
            latest_data = fetch_latest_data(strategy.symbol, interval)
            order_blocks = generate_order_blocks_list(smc, latest_data)
            print(order_blocks)
            plot_order_blocks(smc, latest_data)

            # 2. Analyze market with SMC
            print("Analyzing with Smart Money Concepts...")
            analyzed_data = strategy.analyze_market(latest_data)

            # 3. Find active order blocks near current price
            market_analysis = strategy.find_active_order_blocks(analyzed_data)
            print(market_analysis)

            # 4. Print market analysis summary
            current_price = market_analysis['current_price']
            print(f"\nCurrent {strategy.symbol} price: {current_price}")
            print(
                f"Current trend: {'Bullish' if market_analysis['trend'] > 0 else 'Bearish' if market_analysis['trend'] < 0 else 'Neutral'}")
            print(
                f"Found {len(market_analysis['bullish_obs'])} bullish order blocks near price")
            print(
                f"Found {len(market_analysis['bearish_obs'])} bearish order blocks near price")

            # 5. Generate trading signals
            signals = strategy.generate_trade_signals(
                market_analysis, analyzed_data)

            # 6. Execute signals (simulation)
            if signals:
                print("\n=== New Trading Signals ===")
                strategy.execute_signals(signals)
            else:
                print("\nNo new trading signals generated.")

            # 7. Plot order blocks (optional - for visual analysis)
            if market_analysis['bullish_obs'] or market_analysis['bearish_obs']:
                strategy.plot_order_blocks(latest_data, market_analysis)

            # Wait for next check
            print(
                f"\nWaiting {check_interval_minutes} minutes until next check...")
            time.sleep(check_interval_minutes * 60)

        except Exception as e:
            print(f"Error: {e}")
            print("Waiting 5 minutes before retrying...")
            time.sleep(300)


# Run the strategy
if __name__ == "__main__":
    run_strategy_loop(interval='15m', check_interval_minutes=15)
