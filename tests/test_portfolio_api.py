import base64
import json

from app import main
from app.portfolio import load_config, new_wallet


class Response:
    def __init__(self, status, payload):
        self.status_code = status
        self.payload = payload

    def raise_for_status(self):
        assert self.status_code == 200

    def json(self):
        return self.payload


def test_dashboard_reads_contents_api_when_raw_visibility_cache_returns_not_found(monkeypatch):
    wallet = new_wallet(load_config())
    wallet['report_status'] = 'sent'
    responses = iter([Response(404, {}), Response(200, {
        'content': base64.b64encode(json.dumps(wallet).encode()).decode()})])
    monkeypatch.setattr(main.requests, 'get', lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(main, 'portfolio_cache', {'at': 0, 'state': None})
    result = main.portfolio()
    assert result['available'] and result['equity'] == 20 and result['report_status'] == 'sent'
