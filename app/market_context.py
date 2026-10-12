import math
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime

import requests
from sklearn.feature_extraction.text import HashingVectorizer

from app.core import INTERVAL_MS

NEWS_FEEDS = {
    'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'cointelegraph': 'https://cointelegraph.com/rss',
}
MACRO_SYMBOLS = {'gold': 'GC=F', 'dxy': 'DX-Y.NYB'}
NEWS_DIMENSIONS = 64
CONTEXT_DIMENSIONS = 10 + NEWS_DIMENSIONS


def fetch_payload(url, params=None):
    response = requests.get(url, params=params, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
    response.raise_for_status()
    if len(response.content) > 2_000_000:
        raise ValueError('External data exceeds size limit.')
    return response


def parse_macro(payload, now):
    result = payload['chart']['result'][0]
    rows = []
    for stamp, value in zip(result['timestamp'], result['indicators']['quote'][0]['close']):
        end = int(stamp) * 1000 + INTERVAL_MS
        if value is not None and end <= now and math.isfinite(value) and value > 0:
            rows.append((end, float(value)))
    rows = sorted(set(rows))
    if len(rows) < 5:
        raise ValueError('Insufficient closed macro candles.')
    stamp, price = rows[-1]
    age = now - stamp
    if age > 4 * 86400000:
        raise ValueError('Macro data is older than four days.')
    return {'price': price, 'as_of': stamp, 'age_minutes': age / 60000,
            'return_1': price / rows[-2][1] - 1,
            'return_4': price / rows[-5][1] - 1,
            'status': 'recent' if age <= 2 * INTERVAL_MS else 'delayed_or_closed'}


def parse_news(content, now):
    if b'<!DOCTYPE' in content.upper() or b'<!ENTITY' in content.upper():
        raise ValueError('Unsupported news XML declaration.')
    root = ET.fromstring(content)
    if root.tag != 'rss' or root.find('channel') is None:
        raise ValueError('Invalid RSS feed.')
    articles = {}
    for item in root.findall('./channel/item')[:100]:
        title = ' '.join((item.findtext('title') or '').split())[:300]
        try:
            published = parsedate_to_datetime(item.findtext('pubDate') or '')
            if published.tzinfo is None:
                continue
            stamp = int(published.timestamp() * 1000)
        except (TypeError, ValueError, OverflowError):
            continue
        if title and now - 86400000 <= stamp <= now:
            articles[title.casefold()] = {'title': title, 'published_at': stamp}
    return list(articles.values())


def collect_context(now):
    def macro(symbol):
        response = fetch_payload(f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}',
                                 {'interval': '15m', 'range': '5d'})
        return parse_macro(response.json(), now)

    def news(url):
        return parse_news(fetch_payload(url).content, now)

    result = {'observed_at': now, 'macro': {}, 'news': [], 'errors': {}, 'news_sources': []}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {name: pool.submit(macro, symbol) for name, symbol in MACRO_SYMBOLS.items()}
        jobs.update({name: pool.submit(news, url) for name, url in NEWS_FEEDS.items()})
        for name, task in jobs.items():
            try:
                value = task.result()
                if name in MACRO_SYMBOLS:
                    result['macro'][name] = value
                else:
                    result['news_sources'].append(name)
                    result['news'].extend(value)
            except (requests.RequestException, ValueError, KeyError, TypeError, IndexError, ET.ParseError):
                result['errors'][name] = 'Source unavailable or invalid.'
    result['news'] = list({item['title'].casefold(): item for item in result['news']}.values())[:100]
    result['ready'] = len(result['macro']) == 2 and bool(result['news_sources'])
    result['vector'] = context_vector(result)
    return result


def context_vector(context):
    values = []
    for name in MACRO_SYMBOLS:
        item = context['macro'].get(name)
        values.extend([item['return_1'] * 100, item['return_4'] * 100,
                       min(item['age_minutes'] / 1440, 4), 1.] if item else [0., 0., 4., 0.])
    values.extend([min(len(context['news']) / 20, 5), float(bool(context['news_sources']))])
    text = ' '.join(item['title'] for item in context['news'])
    hashed = HashingVectorizer(n_features=NEWS_DIMENSIONS, alternate_sign=True,
                              norm='l2', ngram_range=(1, 2), stop_words='english').transform([text])
    return [round(max(-5., min(5., value)), 6) for value in values + hashed.toarray()[0].tolist()]
