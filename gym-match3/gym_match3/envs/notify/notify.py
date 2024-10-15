import os
from io import BytesIO

import requests


def notify_finish(is_success):
    MODEL_NAME = os.getenv('MODEL_NAME', "MODEL_NOT_DEFINED")
    USER_DEVICE = os.getenv('USER_DEVICE', "USER_DEVICE_NOT_DEFINED")
    message = f"{MODEL_NAME} training completed {'successfully' if is_success else 'fail'} on {USER_DEVICE}"

    send(message)


def send(message):
    BOT_TOKEN = os.getenv('BOT_TOKEN', None)
    CHAT_ID = os.getenv('CHAT_ID', None)

    if not BOT_TOKEN or not CHAT_ID: return

    # URL for the Telegram Bot API
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'

    # Parameters for the request
    params = {
        'chat_id': CHAT_ID,
        'text': message,
        'parse_mode': 'MarkdownV2'
    }

    # Send the message
    response = requests.post(url, params=params)

    # Check if the message was sent successfully
    if response.status_code != 200:
        print(f"Failed to send message. Status code: {response.status_code}")
        print(response.text)


def send_image(image, image_caption=""):
    BOT_TOKEN = os.getenv('BOT_TOKEN')
    CHAT_ID = os.getenv('CHAT_ID')

    if not BOT_TOKEN or not CHAT_ID: return

    bio = BytesIO()
    bio.name = 'image.png'
    image.save(bio, 'PNG')

    # URL for the Telegram Bot API
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto'
    files = {'photo': ('image.png', bio.getvalue())}

    # Parameters for the request
    params = {
        'chat_id': CHAT_ID,
        "caption": image_caption
    }

    response = requests.post(url, files=files, params=params)
    # Check if the message was sent successfully
    if response.status_code != 200:
        print(f"Failed to send message. Status code: {response.status_code}")
        print(response.text)