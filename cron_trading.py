from binance_data_fetcher import BinanceDataFetcher
from telegram_bot import TelegramBot
from strategy import analyze_trading_setup, find_closest_signal
from smartmoneyconcepts.smc import smc
import time
from datetime import datetime, timedelta
import logging
import asyncio
from dotenv import load_dotenv
import os
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from aiohttp import web
import asyncio


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


async def health_check(request):
    """Simple health check endpoint"""
    return web.Response(text="OK", status=200)


async def start_web_server():
    """Start web server for health checks"""
    app = web.Application()
    app.router.add_get('/', health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8000)
    await site.start()


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
            start_time = datetime.now() - timedelta(days=10)

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
            result = find_closest_signal(df, trade_setups['current_price'])

            # Create signal lines
            signal_lines = []
            for signal in result['signals']:

                signal_line = (
                    f"{signal['signal_type']} | "
                    f"Entry: {signal['entry_price']:.2f} | "
                    f"Distance: {signal['price_distance']:.1f}% | "
                    f"{signal['safety_emoji']} {signal['safety_score']:.1f}%"
                )
                signal_lines.append(signal_line)

            message = (
                f"📊 <b>Signal Analysis</b>\n"
                f"• Current Price: {result['current_price']:.2f}\n"
                f"• Market Trend: {result['trend']}\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>Trading Signals</b>\n"
                f"{chr(10).join(signal_lines)}\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
            )

            # Send trading setups through Telegram
            await self.telegram_bot.send_trading_setup(trade_setups, symbol, interval, header_message=message)
            logger.info(
                f"Sent {len(trade_setups)} trading setups for {symbol} {interval}")

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


async def main():
    """Main function to run the trading bot"""
    try:
        await start_web_server()

        bot = TradingBot()

        # await bot.scan_markets()

        # await bot.scan_markets()

        # Create tasks for different timeframe scans
        scheduler = AsyncIOScheduler(timezone=pytz.timezone('UTC'))

        # Schedule the scan_markets to run at minutes 1, 16, 31, 46
        # scheduler.add_job(bot.scan_markets, 'cron', minute='*/10',  # Run every 10 minutes
        #                   second='30')
        scheduler.add_job(bot.scan_markets, 'cron', minute='14,29,44,59',  # Run every 10 minutes
                          second='30')

        scheduler.start()

        # Keep the main program running
        while True:
            await asyncio.sleep(1)

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
