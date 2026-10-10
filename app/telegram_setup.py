import getpass

import requests


def private_chat_ids(token):
    try:
        response = requests.post(f'https://api.telegram.org/bot{token}/getUpdates',
                                 json={'timeout': 0, 'limit': 100}, timeout=15)
        payload = response.json()
    except (requests.RequestException, ValueError):
        raise RuntimeError('Cannot reach Telegram. The token was not printed or saved.') from None
    if not response.ok or not payload.get('ok'):
        raise RuntimeError('Telegram rejected getUpdates. Check the token and whether the bot has a webhook.')
    chats = set()
    for update in payload['result']:
        chat = update.get('message', {}).get('chat', {})
        if chat.get('type') == 'private' and isinstance(chat.get('id'), int):
            chats.add(chat['id'])
    return sorted(chats)


def main():
    print('Open your own bot in Telegram and press Start before continuing.')
    token = getpass.getpass('Bot token (hidden; not saved): ').strip()
    if not token:
        raise SystemExit('No token entered.')
    try:
        chats = private_chat_ids(token)
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
    if not chats:
        raise SystemExit('No private chat found. Send /start to your bot, then run this command again.')
    if len(chats) > 1:
        print('Multiple private chats found. Select your own chat ID; do not guess another recipient.')
    for chat_id in chats:
        print(f'TELEGRAM_CHAT_ID={chat_id}')


if __name__ == '__main__':
    main()
