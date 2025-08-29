import os
import sys
import requests
import argparse
import re
import datetime
from dotenv import load_dotenv
import random

# Add the scripts directory to the path to import openrouter_client
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'llm'))
from openrouter_client import call_openrouter_api, MODEL_MAPPING

# Load environment variables from .env file
load_dotenv()

# Environment variables
TELEGRAM_HABIT_BOT_API_KEY = os.environ.get("TELEGRAM_HABIT_BOT_API_KEY")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Telegram message length limit
TELEGRAM_MAX_LENGTH = 4096


def send_telegram_message(bot_token, chat_id, message):
    """Sends a message to a Telegram chat using the Telegram Bot API."""
    if not bot_token or not chat_id:
        print("Error: TELEGRAM_HABIT_BOT_API_KEY or TELEGRAM_CHAT_ID not set.")
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    # Remove Markdown asterisks and URLs to ensure Telegram compatibility
    message_no_stars = message.replace("*", "")
    url_pattern = re.compile(r"(https?://[^\s]+)")
    message_no_links = url_pattern.sub("", message_no_stars)
    # Split message if it exceeds Telegram's length limit
    messages = []
    msg = message_no_links
    while len(msg) > TELEGRAM_MAX_LENGTH:
        split_idx = msg.rfind("\n", 0, TELEGRAM_MAX_LENGTH)
        if split_idx == -1 or split_idx < TELEGRAM_MAX_LENGTH // 2:
            split_idx = TELEGRAM_MAX_LENGTH
        messages.append(msg[:split_idx])
        msg = msg[split_idx:]
    messages.append(msg)
    success = True
    for part in messages:
        params = {"chat_id": chat_id, "text": part, "parse_mode": "Markdown"}
        try:
            response = requests.post(url, params=params)
            response.raise_for_status()
            print(f"Successfully sent Telegram message part ({len(part)} chars).")
        except requests.exceptions.RequestException as e:
            print(f"Error sending Telegram message: {e}")
            success = False
    return success




def generate_copilot_message():
    """Generates a technical prompt sentence encouraging Copilot use via OpenRouter API."""
    prompt = "Provide a concise daily tip related to one of the following technologies: machine learning, backend engineering, Java, or Python. Only the tip content is needed."
    
    # Randomly select a model from all available models
    selected_model = random.choice(list(MODEL_MAPPING.keys()))
    
    try:
        print(f"Calling OpenRouter API with model: {selected_model}")
        message = call_openrouter_api(prompt, model=selected_model)
        if message:
            return message.strip()[:300]
    except Exception as e:
        print(f"OpenRouter API Error with {selected_model}: {e}")
    
    # Fallback message if API call fails
    return "Stuck on configuring Control-M order date? Ask Copilot!"


def main():
    parser = argparse.ArgumentParser(description="Telegram Habit Reminder Bot")
    parser.add_argument(
        "--job",
        choices=["send_reminder", "send_message"],
        required=True,
        help="Job to perform",
    )
    parser.add_argument(
        "--message", type=str, help="Message to send for 'send_message' job"
    )
    args = parser.parse_args()

    if args.job == "send_reminder":
        if TELEGRAM_HABIT_BOT_API_KEY and TELEGRAM_CHAT_ID:
            message = generate_copilot_message()
            send_telegram_message(TELEGRAM_HABIT_BOT_API_KEY, TELEGRAM_CHAT_ID, message)
        else:
            print("Error: TELEGRAM_HABIT_BOT_API_KEY or TELEGRAM_CHAT_ID not set.")
    elif args.job == "send_message":
        if TELEGRAM_HABIT_BOT_API_KEY and TELEGRAM_CHAT_ID:
            message = (
                args.message if args.message else "Default test message from the bot!"
            )
            send_telegram_message(TELEGRAM_HABIT_BOT_API_KEY, TELEGRAM_CHAT_ID, message)
        else:
            print("Error: TELEGRAM_HABIT_BOT_API_KEY or TELEGRAM_CHAT_ID not set.")


if __name__ == "__main__":
    main()
