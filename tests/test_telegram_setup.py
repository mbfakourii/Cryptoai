import pytest
import requests

from app import telegram_setup


def test_only_unique_private_chat_ids_are_returned(monkeypatch):
    class Response:
        ok = True

        def json(self):
            return {'ok': True, 'result': [
                {'message': {'chat': {'id': 123, 'type': 'private'}, 'text': 'sensitive text'}},
                {'message': {'chat': {'id': 123, 'type': 'private'}}},
                {'message': {'chat': {'id': -456, 'type': 'group'}}},
                {'callback_query': {}}]}

    monkeypatch.setattr(telegram_setup.requests, 'post', lambda *a, **kw: Response())
    assert telegram_setup.private_chat_ids('fake-token') == [123]


def test_network_failure_does_not_expose_token(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError('https://api.telegram.org/botfake-token/getUpdates')

    monkeypatch.setattr(telegram_setup.requests, 'post', fail)
    with pytest.raises(RuntimeError) as error:
        telegram_setup.private_chat_ids('fake-token')
    assert 'fake-token' not in str(error.value)
