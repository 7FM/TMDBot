import logging
import time
import threading
import requests

from botlib import state
from botlib.config import settings, load_settings

logger = logging.getLogger(__name__)

OL_BASE = "https://openlibrary.org"
OL_COVERS = "https://covers.openlibrary.org"
OL_SEARCH_FIELDS = ",".join([
    "key", "title", "author_name", "first_publish_year",
    "cover_i", "subject", "ratings_average", "ratings_count",
    "number_of_pages_median", "edition_count",
])

_session = requests.Session()

# Rate limiter: 3 requests per second with identified User-Agent
_MIN_INTERVAL = 1.0 / 3.0  # ~333ms between requests
_last_request_time = 0.0
_rate_lock = threading.Lock()


def _rate_limited_get(url, **kwargs):
    """GET with rate limiting to respect OL's 3 req/s limit."""
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()
    return _session.get(url, **kwargs)


def ol_search(query, limit=20):
    """Search Open Library for books."""
    r = _rate_limited_get(f"{OL_BASE}/search.json", params={
        "q": query, "limit": limit, "fields": OL_SEARCH_FIELDS,
    })
    r.raise_for_status()
    return r.json().get("docs", [])


def ol_work(work_id):
    """Get work details by numeric ID (e.g., 27482 for OL27482W)."""
    r = _rate_limited_get(f"{OL_BASE}/works/OL{work_id}W.json")
    r.raise_for_status()
    return r.json()


def ol_search_authors(query, limit=20):
    """Search for authors."""
    r = _rate_limited_get(f"{OL_BASE}/search/authors.json", params={
        "q": query, "limit": limit,
    })
    r.raise_for_status()
    return r.json().get("docs", [])


def ol_author_works(author_olid, limit=50):
    """Get works by an author OLID (e.g., 'OL26320A')."""
    r = _rate_limited_get(f"{OL_BASE}/authors/{author_olid}/works.json", params={
        "limit": limit,
    })
    r.raise_for_status()
    return r.json().get("entries", [])


def ol_trending(limit=20):
    """Get trending books."""
    r = _rate_limited_get(f"{OL_BASE}/trending/daily.json", params={
        "limit": limit,
    })
    r.raise_for_status()
    return r.json().get("works", [])


def ol_subject(subject, limit=20):
    """Get books by subject."""
    r = _rate_limited_get(f"{OL_BASE}/subjects/{subject}.json", params={
        "limit": limit,
    })
    r.raise_for_status()
    data = r.json()
    return data.get("works", [])


def user_data_initialize():
    ud = state.user_data
    for user in settings['allowed_users']:
        if user not in ud:
            ud[user] = dict()
            ud[user]["watched"] = {"book": {}}
            ud[user]["watchlists"] = {
                "book": {"to-read": []},
            }
            ud[user]["onboarded"] = False
            ud[user]["mode"] = "book"
            ud[user]["name"] = ""
    if "shared_watchlists" not in ud:
        ud["shared_watchlists"] = {}
    if "_shared_wl_next_id" not in ud:
        ud["_shared_wl_next_id"] = 1
    state.save_user_data()


def init(settings_file, user_data_file):
    settings.update(load_settings(settings_file))

    # Set User-Agent with contact email for OL API (3 req/s instead of 1)
    email = settings.get("email", "")
    _session.headers["User-Agent"] = f"BookBot/0.1 ({email})"

    state.init(user_data_file)
    from bookbot.migration import migrate
    migrate()
    user_data_initialize()
