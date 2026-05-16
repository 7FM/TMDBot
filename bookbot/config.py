import logging
import time
import threading

import requests

from botlib import state
from botlib.config import settings, load_settings

logger = logging.getLogger(__name__)

HC_ENDPOINT = "https://api.hardcover.app/v1/graphql"

_session = requests.Session()

# Hardcover rate limit: 60 req/min
_MIN_INTERVAL = 1.0
_last_request_time = 0.0
_rate_lock = threading.Lock()


def _rate_limited_post(payload):
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()
    return _session.post(HC_ENDPOINT, json=payload)


def hc_query(query, variables=None):
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    r = _rate_limited_post(payload)
    r.raise_for_status()
    body = r.json()
    if "errors" in body:
        raise RuntimeError(f"Hardcover GraphQL error: {body['errors']}")
    return body.get("data") or {}


_BOOK_FIELDS = """
  id
  title
  description
  release_year
  rating
  image { url }
  cached_tags
  contributions { author { name } }
"""


def _tags_to_subjects(cached_tags):
    if not isinstance(cached_tags, dict):
        return []
    out = []
    for cat in ("Genre", "Tag", "Mood"):
        for t in (cached_tags.get(cat) or [])[:5]:
            if isinstance(t, dict):
                name = t.get("tag")
                if name and name not in out:
                    out.append(name)
    return out


def _normalize_book(b):
    if not b:
        return None
    authors = []
    for c in b.get("contributions") or []:
        a = (c or {}).get("author") or {}
        name = a.get("name")
        if name and name not in authors:
            authors.append(name)
    image = b.get("image") or {}
    cover_url = image.get("url") if isinstance(image, dict) else None
    return {
        "id": b.get("id"),
        "title": b.get("title") or "Unknown",
        "authors": authors,
        "year": b.get("release_year"),
        "cover_url": cover_url,
        "rating": b.get("rating"),
        "subjects": _tags_to_subjects(b.get("cached_tags")),
        "description": b.get("description"),
    }


def _normalize_search_doc(doc):
    if not doc:
        return None
    bid = doc.get("id")
    if isinstance(bid, str):
        try:
            bid = int(bid)
        except ValueError:
            return None
    if bid is None:
        return None
    image = doc.get("image") or {}
    if isinstance(image, dict):
        cover_url = image.get("url")
    elif isinstance(image, str):
        cover_url = image
    else:
        cover_url = None
    subjects = []
    for key in ("genres", "tags", "moods"):
        vals = doc.get(key) or []
        if isinstance(vals, list):
            for v in vals[:5]:
                if isinstance(v, str) and v not in subjects:
                    subjects.append(v)
    return {
        "id": bid,
        "title": doc.get("title") or "Unknown",
        "authors": doc.get("author_names") or [],
        "year": doc.get("release_year"),
        "cover_url": cover_url,
        "rating": doc.get("rating"),
        "subjects": subjects,
        "description": doc.get("description"),
    }


def _search_hits(query, query_type, per_page):
    q = """
    query S($q: String!, $type: String!, $per_page: Int!) {
      search(query: $q, query_type: $type, per_page: $per_page) {
        results
      }
    }
    """
    data = hc_query(q, {"q": query, "type": query_type, "per_page": per_page})
    results = (data.get("search") or {}).get("results") or {}
    if isinstance(results, dict):
        return results.get("hits") or []
    return []


def hc_search(query, limit=20):
    """Search books, returns list of normalized book records."""
    hits = _search_hits(query, "Book", limit)
    out = []
    seen = set()
    for h in hits:
        nb = _normalize_search_doc(h.get("document"))
        if nb and nb["id"] not in seen:
            seen.add(nb["id"])
            out.append(nb)
    return out


def hc_book(book_id):
    """Fetch a single book by ID, normalized."""
    q = f"query B($id: Int!) {{ books_by_pk(id: $id) {{ {_BOOK_FIELDS} }} }}"
    data = hc_query(q, {"id": int(book_id)})
    return _normalize_book(data.get("books_by_pk"))


def hc_books(book_ids):
    """Batch fetch books by IDs. Returns {id: normalized_book}."""
    ids = [int(b) for b in book_ids]
    if not ids:
        return {}
    q = f"query Bs($ids: [Int!]!) {{ books(where: {{id: {{_in: $ids}}}}) {{ {_BOOK_FIELDS} }} }}"
    data = hc_query(q, {"ids": ids})
    out = {}
    for b in data.get("books") or []:
        nb = _normalize_book(b)
        if nb and nb["id"] is not None:
            out[nb["id"]] = nb
    return out


def hc_search_authors(query, limit=5):
    """Search authors, returns list of {id, name, slug}."""
    hits = _search_hits(query, "Author", limit)
    out = []
    for h in hits:
        doc = h.get("document") or {}
        aid = doc.get("id")
        if isinstance(aid, str):
            try:
                aid = int(aid)
            except ValueError:
                continue
        if aid is None:
            continue
        out.append({
            "id": aid,
            "name": doc.get("name") or "Unknown",
            "slug": doc.get("slug"),
        })
    return out


def hc_author_works(author_id, limit=50):
    """Get books by an author, ordered by rating desc."""
    q = f"""
    query AW($id: Int!, $limit: Int!) {{
      books(
        where: {{contributions: {{author_id: {{_eq: $id}}}}}},
        order_by: {{rating: desc_nulls_last}},
        limit: $limit
      ) {{ {_BOOK_FIELDS} }}
    }}
    """
    data = hc_query(q, {"id": int(author_id), "limit": limit})
    return [_normalize_book(b) for b in (data.get("books") or [])]


def hc_trending(limit=20):
    """Recent popular books, sorted by users_count."""
    q = f"""
    query T($limit: Int!) {{
      books(
        order_by: {{users_count: desc_nulls_last}},
        where: {{release_year: {{_gte: 2020}}, rating: {{_gte: 3.5}}}},
        limit: $limit
      ) {{ {_BOOK_FIELDS} }}
    }}
    """
    data = hc_query(q, {"limit": limit})
    return [_normalize_book(b) for b in (data.get("books") or [])]


def hc_subject(subject, limit=20):
    """Find books matching a subject — uses search ranking."""
    return hc_search(subject, limit=limit)


def hc_book_isbn(book_id):
    """Fetch an ISBN for a book (for on_add hook). Returns ISBN-13 or ISBN-10 or ''."""
    q = """
    query Isbn($id: Int!) {
      editions(
        where: {book_id: {_eq: $id}, _or: [{isbn_13: {_is_null: false}}, {isbn_10: {_is_null: false}}]},
        limit: 1
      ) { isbn_13 isbn_10 }
    }
    """
    try:
        data = hc_query(q, {"id": int(book_id)})
    except Exception:
        return ""
    for ed in data.get("editions") or []:
        return ed.get("isbn_13") or ed.get("isbn_10") or ""
    return ""


def user_data_initialize():
    ud = state.user_data
    for user in settings['allowed_users']:
        if user not in ud:
            ud[user] = dict()
            ud[user]["watched"] = {"book": {}}
            ud[user]["watchlists"] = {"book": {"to-read": []}}
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

    token = settings.get("hardcover_token", "")
    _session.headers["Authorization"] = f"Bearer {token}"
    _session.headers["Content-Type"] = "application/json"

    state.init(user_data_file)
    from bookbot.migration import migrate
    migrate()
    user_data_initialize()
