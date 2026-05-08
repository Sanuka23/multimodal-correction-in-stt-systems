"""Thin wrapper around the Wikipedia opensearch API with caching.

Used by Phase 3 evidence gathering to look up whether a suspected
mis-transcribed word corresponds to a real Wikipedia entity. Example:
query "Cloudware" → returns ["Cloudware", "Cloudflare", ...].

Stdlib-only (urllib.request). Results are cached in-process via lru_cache
to avoid hammering Wikipedia on repeat lookups during a run.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache

logger = logging.getLogger(__name__)

_API_BASE = "https://en.wikipedia.org/w/api.php"
_USER_AGENT = "ScreenApp-FYP-Eval/1.0 (student research; contact: sanuka23thamudithaalles@gmail.com)"


@lru_cache(maxsize=2048)
def opensearch(query: str, limit: int = 5, timeout_sec: float = 5.0) -> list[str]:
    """Return a list of Wikipedia article titles matching `query`.

    The MediaWiki `opensearch` action accepts a free-text query and returns
    an array of [query, titles, descriptions, urls]. We return just the
    titles — the caller handles phonetic scoring.

    Returns an empty list on any network error, JSON parse error, or
    malformed response. Never raises.
    """
    if not query or not query.strip():
        return []

    params = {
        "action": "opensearch",
        "search": query.strip(),
        "limit": str(limit),
        "namespace": "0",
        "format": "json",
    }
    url = f"{_API_BASE}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.warning("Wikipedia opensearch failed for %r: %s", query, e)
        return []
    except json.JSONDecodeError as e:
        logger.warning("Wikipedia opensearch bad JSON for %r: %s", query, e)
        return []

    if not isinstance(data, list) or len(data) < 2:
        return []
    titles = data[1]
    if not isinstance(titles, list):
        return []
    return [t for t in titles if isinstance(t, str)]


def clear_cache() -> None:
    """Clear the opensearch LRU cache. For test isolation."""
    opensearch.cache_clear()
