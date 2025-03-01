import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from datetime import datetime
import pytz


class TelegramBot:
    def __init__(self, token: str, chat_id: str, topic_id: str = None):
        """
        Initialize the Trading Bot

        Parameters:
        - token: Telegram bot token from BotFather
        - chat_id: Telegram chat ID where messages will be sent
        - topic_id: Telegram topic ID where messages will be sent
        """
        self.token = token
        self.chat_id = chat_id
        self.topic_id = topic_id
        self.app = ApplicationBuilder().token(self.token).build()

        # Configure logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )

    async def send_message(self, message: str):
        """Send a simple text message to specific topic"""
        try:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                message_thread_id=self.topic_id,  # Add topic support
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            logging.error(f"Error sending message: {str(e)}")

    def get_trend_emoji(self, trend: str) -> str:
        """
        Get emoji based on market trend

       Parameters:
      - trend: Market trend string (BULLISH, BEARISH, UPTREND, etc.)

        Returns:
        - Appropriate emoji for the trend
        """
        trend = trend.upper()
        if trend == 'UPTREND':
            return '📈'
        elif trend == 'DOWNTREND':
            return '📉'
        elif trend == 'SIDEWAYS':
            return '↔️'
        else:
            return '❓'

    def format_timestamp(self):
        """Format timestamp in ICT (UTC+7)"""
        ict_tz = pytz.timezone('Asia/Bangkok')
        current_time = datetime.now(pytz.UTC).astimezone(ict_tz)
        return current_time.strftime('%Y-%m-%d %H:%M:%S ICT')

    async def send_trading_setup(self, trade_setups, symbol: str, timeframe: str = '15m', rsi: float = None, header_message: str = None):
        """Send all trading setups in a single formatted message"""
        try:
            setups = trade_setups['trade_setups']
            velocity = trade_setups['velocity']
            vol_analysis = trade_setups.get('volume_analysis', {})

            # get macd info from volume_analysis
            macd_info = velocity.get('macd_signals', {})
            current_price = trade_setups.get('current_price', 0)
            current_trend = trade_setups.get('current_trend', 'N/A')

            # Format volume trend indicators
            volume_trend_value = vol_analysis.get('volume_trend', 0)
            volume_emoji = "📈" if volume_trend_value > 5 else "📉" if volume_trend_value < -5 else "➡️"

            pressure_ratio = vol_analysis.get('pressure_ratio', 1)
            pressure_emoji = "🟢" if pressure_ratio > 1.2 else "🔴" if pressure_ratio < 0.83 else "⚪️"

            # Get volume analysis details
            last_candle = vol_analysis['last_candle']
            pressure = vol_analysis['analysis']

            # Get emojis based on scores
            candle_emoji = "🟢" if last_candle['score'] >= 70 else "🔴" if last_candle['score'] <= 30 else "🟡"
            pressure_emoji = "🟢" if pressure['score'] >= 70 else "🔴" if pressure['score'] <= 30 else "🟡"

            # Get MACD signals
            macd_buy = macd_info.get('buy_signal', False)
            macd_sell = macd_info.get('sell_signal', False)
            macd_direction = macd_info.get('macd_direction', 'NEUTRAL')
            histogram_direction = macd_info.get(
                'histogram_direction', 'NEUTRAL')

            # Get EMA price information
            ma_analysis = velocity.get('ma_analysis', {})
            ema50 = ma_analysis.get('ema50', 0)
            ema200 = ma_analysis.get('ema200', 0)

            # Format MACD signal
            macd_signal = "🟢 BUY" if macd_buy else "🔴 SELL" if macd_sell else "⚪️ NEUTRAL"
            macd_trend = f"MACD: {macd_direction} | Histogram: {histogram_direction}"

            # Calculate price position relative to EMAs
            price_to_ema50 = ((current_price / ema50) - 1) * \
                100 if ema50 > 0 else 0
            price_to_ema200 = ((current_price / ema200) -
                               1) * 100 if ema200 > 0 else 0

            message = (
                f"💹 <b>Market Status</b>\n"
                f"• Symbol: {symbol}\n"
                f"• Timeframe: {timeframe}\n"
                f"• Current Price: {current_price:.2f}\n"
                f"• Market Trend: {self.get_trend_emoji(current_trend)} {current_trend}\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>Volume Analysis</b>\n"
                f"• Volume Trend: {volume_emoji} {volume_trend_value:.1f}%\n"
                f"• Pressure: {pressure_emoji} {pressure['pressure']}\n"
                f"• Current Volume: {trade_setups.get('current_volume', 0):.1f}\n"
                f"• Current Volume Ratio: {trade_setups.get('current_volume_ratio', 0):.1f}\n"
                f"• Last Candle: {candle_emoji} {last_candle['type']} ({last_candle['score']}%)\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚡️ <b>Momentum Analysis</b>\n"
                f"• MA Status: {'Above EMA50 ↗️' if velocity.get('ma_analysis', {}).get('above_ema50') else 'Below EMA50 ↘️'}\n"
                f"• EMA50: {ema50:.2f} ({price_to_ema50:.2f}%)\n"
                f"• EMA200: {ema200:.2f} ({price_to_ema200:.2f}%)\n"
                f"• EMA Signal: {'✅ Bullish Cross' if velocity.get('ma_analysis', {}).get('ema_crossover') else '❌ No Cross'}\n"
                f"• RSI ({velocity.get('rsi_analysis', {}).get('current', 0):.1f}): "
                f"{'🔴 Overbought' if velocity.get('rsi_analysis', {}).get('overbought') else '🟢 Oversold' if velocity.get('rsi_analysis', {}).get('oversold') else '⚪️ Neutral'}\n"
                f"• MACD Signal: {macd_signal}\n"
                f"• MACD Trend: {macd_trend}\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
            )
            message += header_message
            if len(setups) == 0:
                message += (
                    f"🔍 <b>Trading Analysis: {symbol} {timeframe}</b>\n\n"
                    "No valid trading setups within optimal range detected."
                )

            # Add timestamp
            message += f"\n⏰ <i>Generated at {self.format_timestamp()}</i>"

            # Send message with splitting if needed
            if len(message) > 4000:
                messages = []
                current_message = ""
                header = (
                    "🎯 <b>Trading Setups</b> (Continued...)\n"
                    "━━━━━━━━━━━━━━━━━━━━━\n\n"
                )

                for line in message.split('\n'):
                    if len(current_message + line + '\n') > 3800:
                        messages.append(current_message)
                        current_message = header + line + '\n'
                    else:
                        current_message += line + '\n'

                if current_message:
                    messages.append(current_message)

                for msg in messages:
                    await self.send_message(msg)
            else:
                await self.send_message(message)

        except Exception as e:
            print(e)
            logging.error(f"Error sending trading setups: {str(e)}")
            await self.send_error_alert(f"Failed to send trading setups: {str(e)}")

    async def send_market_update(self,
                                 symbol: str,
                                 current_price: float,
                                 ob_data: dict,
                                 liquidity_data: dict):
        """
        Send market update with order blocks and liquidity levels

        Parameters:
        - symbol: Trading pair symbol
        - current_price: Current market price
        - ob_data: Order block data
        - liquidity_data: Liquidity levels data
        """
        try:
            # Format active order blocks
            active_obs = []
            for i in range(len(ob_data["OB"])):
                if ob_data["OB"][i] != 0:  # If OB exists
                    ob_type = "Bullish" if ob_data["OB"][i] == 1 else "Bearish"
                    active_obs.append(
                        f"• {ob_type} OB: {ob_data['Bottom'][i]:.2f} - {ob_data['Top'][i]:.2f}"
                    )

            # Format liquidity levels
            liquidity_levels = []
            for i in range(len(liquidity_data["Liquidity"])):
                if liquidity_data["Liquidity"][i] != 0:
                    liq_type = "Bullish" if liquidity_data["Liquidity"][i] == 1 else "Bearish"
                    liquidity_levels.append(
                        f"• {liq_type}: {liquidity_data['Level'][i]:.2f}"
                    )

            # Create message
            message = (
                f"📊 <b>Market Update: {symbol}</b>\n\n"
                f"💵 <b>Current Price:</b> {current_price:.2f}\n\n"
                f"📦 <b>Active Order Blocks:</b>\n"
                f"{chr(10).join(active_obs) if active_obs else 'No active OBs'}\n\n"
                f"💧 <b>Liquidity Levels:</b>\n"
                f"{chr(10).join(liquidity_levels) if liquidity_levels else 'No active liquidity levels'}\n\n"
                f"⏰ <i>Updated at {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC</i>"
            )

            await self.send_message(message)

        except Exception as e:
            logging.error(f"Error sending market update: {str(e)}")

    async def send_error_alert(self, error_message: str):
        """Send error alert message"""
        try:
            message = (
                f"🚫 <b>Error Alert</b> 🚫\n\n"
                f"{error_message}\n\n"
                f"⏰ <i>Error occurred at {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC</i>"
            )

            await self.send_message(message)
        except Exception as e:
            logging.error(f"Error sending error alert: {str(e)}")

    def start(self):
        """Start the bot"""
        try:
            self.app.run_polling()
        except Exception as e:
            logging.error(f"Error starting bot: {str(e)}")
