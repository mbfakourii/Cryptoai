from datetime import datetime, timezone
from email.utils import format_datetime

import pytest

from app.core import INTERVAL_MS
from app.market_context import CONTEXT_DIMENSIONS, collect_context, context_vector, parse_macro, parse_news


def macro_fixture():
    return {'chart': {'result': [{'timestamp': [i * 900 for i in range(8)],
                                  'indicators': {'quote': [{'close': [100 + i for i in range(8)]}]}}]}}


def test_macro_uses_only_closed_candles_and_exposes_quote_age():
    result = parse_macro(macro_fixture(), 6 * INTERVAL_MS + 1000)
    assert result['price'] == 105
    assert result['as_of'] == 6 * INTERVAL_MS
    assert result['return_4'] == pytest.approx(105 / 101 - 1)


def test_weekend_macro_is_marked_delayed_and_very_old_quotes_are_rejected():
    result = parse_macro(macro_fixture(), 2 * 86400000)
    assert result['status'] == 'delayed_or_closed'
    with pytest.raises(ValueError, match='four days'):
        parse_macro(macro_fixture(), 6 * 86400000)


def test_news_rejects_future_old_and_undated_items_and_deduplicates_headlines():
    now = 1800000000000
    items = []
    for title, timestamp in [('Valid', now - 1000), ('Valid', now - 1000),
                             ('Future', now + 60000), ('Old', now - 2 * 86400000)]:
        date = format_datetime(datetime.fromtimestamp(timestamp / 1000, timezone.utc))
        items.append(f'<item><title>{title}</title><pubDate>{date}</pubDate></item>')
    items.append('<item><title>Undated</title></item>')
    content = ('<rss><channel>' + ''.join(items) + '</channel></rss>').encode()
    assert [item['title'] for item in parse_news(content, now)] == ['Valid']


def test_news_rejects_entity_documents():
    with pytest.raises(ValueError, match='XML'):
        parse_news(b'<!DOCTYPE rss [<!ENTITY x "hello">]><rss/>', 0)


def test_external_outage_is_explicit_and_does_not_abort_exit_processing(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('invalid response')

    monkeypatch.setattr('app.market_context.fetch_payload', fail)
    result = collect_context(1800000000000)
    assert not result['ready'] and len(result['errors']) == 4
    assert len(result['vector']) == CONTEXT_DIMENSIONS


def test_headlines_change_features_without_a_predefined_buy_or_sell_rule():
    context = {'macro': {}, 'news_sources': ['test'], 'news': [{'title': 'Bitcoin ETF approved'}]}
    first = context_vector(context)
    context['news'] = [{'title': 'Exchange hack and insolvency'}]
    assert context_vector(context) != first
