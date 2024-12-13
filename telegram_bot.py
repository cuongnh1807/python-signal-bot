import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from datetime import datetime
import pytz


class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        """
        Initialize the Trading Bot

        Parameters:
        - token: Telegram bot token from BotFather
        - chat_id: Telegram chat ID where messages will be sent
        """
        self.token = token
        self.chat_id = chat_id
        self.app = ApplicationBuilder().token(self.token).build()

        # Configure logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )

    async def send_message(self, message: str):
        """Send a simple text message"""
        try:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
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

    async def send_trading_setup(self, trade_setups, symbol: str, timeframe: str = '15m', rsi: float = None):
        """Send all trading setups in a single formatted message"""
        try:
            setups = trade_setups['trade_setups']
            velocity = trade_setups['velocity']
            # Filter out setups that are too far or have distance warning
            filtered_setups = [
                setup for setup in setups
                # Only include setups within 3% of current price
                if setup['price_distance'] <= 3.0 and
                "Entry far from current price - Higher risk" not in setup.get(
                    'warning_messages', [])
            ]

            # Sort setups by proximity and quality
            filtered_setups.sort(key=lambda x: (
                x['price_distance'],          # Primary: closest to price
                -x['setup_quality'],          # Secondary: highest quality
                -x['volume_score']            # Tertiary: highest volume
            ))

            # Limit to top 3 closest setups
            filtered_setups = filtered_setups[:3]

            # Get volume analysis from the first setup (if any)
            bullish_setups = [
                s for s in filtered_setups if s['position_type'].upper() == 'LONG']
            bearish_setups = [
                s for s in filtered_setups if s['position_type'].upper() == 'SHORT']

            if filtered_setups:
                vol_analysis = filtered_setups[0].get('volume_analysis', {})
                current_price = filtered_setups[0].get('current_price', 'N/A')
                current_trend = filtered_setups[0].get('current_trend', 'N/A')

                # Format volume trend indicators
                volume_emoji = "📈" if vol_analysis.get(
                    'volume_trend', 0) > 5 else "📉" if vol_analysis.get('volume_trend', 0) < -5 else "➡️"
                pressure_emoji = "🟢" if vol_analysis.get(
                    'pressure_ratio', 1) > 1.2 else "🔴" if vol_analysis.get('pressure_ratio', 1) < 0.83 else "⚪️"

                message = (
                    f"💹 <b>Market Status</b>\n"
                    f"• Symbol: {symbol}\n"
                    f"• Timeframe: {timeframe}\n"
                    f"• RSI: {rsi:.2f}\n"
                    f"• Current Price: {current_price:.2f}\n"
                    f"• Market Trend: {self.get_trend_emoji(current_trend)} {current_trend}\n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 <b>Volume Analysis</b>\n"
                    f"• Volume Score: {vol_analysis.get('volume_score', 0)}/100\n"
                    f"• Volume Trend: {volume_emoji} {vol_analysis.get('volume_trend', 0):.1f}%\n"
                    f"• Buy/Sell Ratio: {vol_analysis.get('buy_ratio', 0):.1f}% / {vol_analysis.get('sell_ratio', 0):.1f}%\n"
                    f"• Pressure: {pressure_emoji} {vol_analysis.get('analysis', {}).get('pressure', 'N/A')}\n"
                    f"• Volume Dominance: {vol_analysis.get('analysis', {}).get('dominance', 'N/A')}\n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⚡️ <b>Momentum Analysis</b>\n"
                    f"• Price: {velocity.get('price', {}).get('current', 0):.2f}% "
                    f"({'↗️' if velocity.get('price', {}).get('condition') == 'INCREASING' else '↘️'})\n"
                    f"• Volume: {velocity.get('volume', {}).get('current', 0):.2f}% "
                    f"({'📈' if velocity.get('volume', {}).get('condition') == 'INCREASING' else '📉'})\n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    "🎯 <b>Closest Trading Setups</b>\n"
                    f"📊 Valid Setups Within Range: {len(filtered_setups)}\n"
                    f"📈 Bullish: {len(bullish_setups)} | "
                    f"📉 Bearish: {len(bearish_setups)}\n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                )
            else:
                message = (
                    f"🔍 <b>Trading Analysis: {symbol} {timeframe}</b>\n\n"
                    "No valid trading setups within optimal range detected."
                )

            def format_setup(setup, index):
                # Add distance emoji based on price distance
                distance_emoji = "🎯" if setup['price_distance'] <= 1.0 else \
                    "📍" if setup['price_distance'] <= 2.0 else \
                    "📌"  # if distance > 2.0%

                # Get warning messages
                warnings = setup.get('warning_messages', [])
                warning_section = ""
                if warnings:
                    warning_section = (
                        f"⚠️ <b>Warnings</b>\n"
                        f"{chr(10).join(f'• {w}' for w in warnings)}\n\n"
                    )

                # Volume comparison section

                volume_section = (
                    f"📊 <b>Volume Analysis</b>\n"
                    f"• OB Volume: {setup.get('ob_volume', 'N/A')}\n"
                    f"• OB/Avg Ratio: {setup.get('ob_volume_ratio', 0):.2f}x\n"
                )

                # Entry timing
                limit_rec = setup.get('limit_order_recommendation', {})
                entry_timing = (
                    f"⏱️ <b>Entry Timing</b>\n"
                    f"• Status: {limit_rec.get('urgency', 'LOW')} "
                    f"({'✅ Ready' if limit_rec.get('place_order') else '⏳ Wait'})\n"
                    f"• Note: {limit_rec.get('reason', 'N/A')}\n\n"
                )

                return (
                    f"{distance_emoji} #{index} {setup['setup_type']}\n"
                    f"{entry_timing}"
                    f"{volume_section}"
                    f"{warning_section}"
                    f"📝 <b>Setup Description</b>\n"
                    f"• Type: {setup.get('setup_description', 'N/A')}\n"
                    f"• Strength: {setup.get('setup_strength', 'N/A')}\n"
                    f"• Distance from price: {setup['price_distance']:.2f}%\n"
                    f"• Setup Quality: {setup['setup_quality']}/100 ({setup['entry_quality']})\n"
                    f"• Order Block Range: {setup['ob_level']}\n\n"
                    f"• Trade Recommendation: {setup['trade_recommendation']}\n\n"
                    f"📐 <b>Entry Zones</b>\n"
                    f"  ▪️ Aggressive: {setup['entry_prices']['aggressive']:.2f}\n"
                    f"  ▪️ Moderate: {setup['entry_prices']['moderate']:.2f}\n"
                    f"  ▪️ Conservative: {setup['entry_prices']['conservative']:.2f}\n"
                    f"• Stop Loss: {setup['stop_loss']:.2f}\n"
                    f"🎯 <b>Take Profit Levels</b>\n"
                    f"  ▪️ TP1: {setup.get('take_profit_levels', {}).get('tp1', 0):.2f}\n"
                    f"  ▪️ TP2: {setup.get('take_profit_levels', {}).get('tp2', 0):.2f}\n"
                    f"  ▪️ TP3: {setup.get('take_profit_levels', {}).get('tp3', 0):.2f}\n\n"
                    f"⚖️ <b>Risk Management</b>\n"
                    f"• Risk: {setup['risk_percentage']:.2f}% ({setup['risk_rating']})\n"
                    f"• Suggested Leverage: {setup['suggested_leverage']}x\n"
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                )

            # Add bullish setups
            if bullish_setups:
                message += "🟢 <b>BULLISH SETUPS</b>\n\n"
                for i, setup in enumerate(bullish_setups, 1):
                    message += format_setup(setup, i)

            # Add bearish setups
            if bearish_setups:
                message += "🔴 <b>BEARISH SETUPS</b>\n\n"
                for i, setup in enumerate(bearish_setups, 1):
                    message += format_setup(setup, i)

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
