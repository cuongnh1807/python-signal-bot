from binance_data_fetcher import BinanceDataFetcher
from telegram_bot import TelegramBot
from strategy import analyze_trading_setup
from smartmoneyconcepts.smc import smc
import time
from datetime import datetime, timedelta
import logging
import asyncio
from dotenv import load_dotenv
import os
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize components
        self.data_fetcher = BinanceDataFetcher()
        self.telegram_bot = TelegramBot(
            token=os.getenv('TELEGRAM_BOT_TOKEN'),
            chat_id=os.getenv('TELEGRAM_CHAT_ID')
        )

        # Trading parameters
        self.symbols = ['BTCUSDT']
        self.timeframes = {
            '15m': 15,
            # '1h': 60,
            # '4h': 240
        }

    async def fetch_and_analyze(self, symbol: str, interval: str):
        """Fetch data and analyze trading setups for a symbol/timeframe"""
        try:
            logger.info(f"Analyzing {symbol} on {interval} timeframe")

            # Calculate start time (5 days of data)
            start_time = datetime.now() - timedelta(days=7)

            # Fetch market data
            df = self.data_fetcher.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_time
            )

            if df.empty:
                logger.warning(f"No data received for {symbol} {interval}")
                return

            # Calculate swing highs/lows
            swing_hl = smc.swing_highs_lows(df, swing_length=5)
            # Analyze trading setups
            trade_setups = analyze_trading_setup(df, swing_hl)

            if trade_setups['trade_setups']:
                # Send trading setups through Telegram
                await self.telegram_bot.send_trading_setup(trade_setups, symbol, interval, trade_setups['rsi'])
                logger.info(
                    f"Sent {len(trade_setups)} trading setups for {symbol} {interval}")
            else:
                logger.info(f"No valid setups found for {symbol} {interval}")

        except Exception as e:
            logger.exception(e)
            error_msg = f"Error analyzing {symbol} {interval}: {str(e)}"
            logger.error(error_msg)
            await self.telegram_bot.send_error_alert(error_msg)

    async def scan_markets(self):
        """Scan all markets for trading setups"""
        try:
            logger.info("Starting market scan")

            for symbol in self.symbols:
                for interval in self.timeframes:
                    await self.fetch_and_analyze(symbol, interval)
                    await asyncio.sleep(1)  # Rate limiting

            logger.info("Market scan completed")

        except Exception as e:
            error_msg = f"Error in market scan: {str(e)}"
            logger.error(error_msg)
            await self.telegram_bot.send_error_alert(error_msg)

    async def run_periodic(self, interval_seconds: int):
        """Run periodic scans with specified interval"""
        while True:
            await self.scan_markets()
            await asyncio.sleep(interval_seconds)


async def main():
    """Main function to run the trading bot"""
    try:
        bot = TradingBot()

        # Create tasks for different timeframe scans
        tasks = [
            bot.run_periodic(240),  # 3 minutes for 15m timeframe
        ]

        # Run initial scan

        # Run periodic scans
        await asyncio.gather(*tasks)

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        if 'bot' in locals():
            await bot.telegram_bot.send_error_alert(
                f"Trading bot crashed: {str(e)}"
            )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Trading bot terminated by user")
    except Exception as e:
        logger.error(f"Failed to start trading bot: {str(e)}")
