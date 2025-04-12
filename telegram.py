import requests
import logging
logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Handles sending notifications to Telegram with support for topics.
    """

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True, orders_topic_id: str = None, signals_topic_id: str = None):
        """
        Initialize the Telegram notifier with topic support.

        Parameters:
        -----------
        bot_token: Telegram bot token
        chat_id: Telegram chat ID to send messages to
        enabled: Whether notifications are enabled
        orders_topic_id: Topic ID for orders notifications
        signals_topic_id: Topic ID for signals notifications
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.orders_topic_id = orders_topic_id
        self.signals_topic_id = signals_topic_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        if enabled:
            self.send_message("🤖 Trading Bot initialized and ready.")
            logger.info("Telegram notifications enabled")
        else:
            logger.info("Telegram notifications disabled")

    def send_message(self, message: str, parse_mode: str = "HTML", topic_id: str = None):
        """Send a message to the Telegram chat with optional topic"""
        if not self.enabled:
            return

        try:
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }

            # Add message_thread_id for topic if provided
            if topic_id:
                data["message_thread_id"] = topic_id

            response = requests.post(self.base_url, data=data)

            if response.status_code != 200:
                logger.error(
                    f"Failed to send Telegram message: {response.text}")

        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
