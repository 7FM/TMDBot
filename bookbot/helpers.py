from bookbot.config import hc_book

# Re-export generic helpers
from botlib.helpers import (  # noqa: F401
    esc, _esc_plain, get_user_id, check_user_invalid,
    split_into_chunks, parse_callback_data,
    _mode_to_type, _type_to_mode, sort_by_rating,
    get_watched_rating, get_watched_category,
    is_in_any_watchlist, find_all_watchlists,
    _get_shared_wl, _next_shared_wl_id, _user_shared_watchlists,
    _get_user_display_name, is_in_any_shared_watchlist,
    find_all_shared_watchlists,
)


def extract_book_info(book):
    """Format a normalized book record into list-display tuple.

    Returns (sort_rating, cover_url, desc, book_id).
    """
    if not book:
        return (0, None, "Unknown book", None)
    title = book.get("title") or "Unknown"
    authors = book.get("authors") or []
    year = book.get("year")
    cover_url = book.get("cover_url")
    rating = book.get("rating")
    subjects = book.get("subjects") or []
    book_id = book.get("id")

    author_str = ", ".join(authors[:2]) if authors else "Unknown author"
    parts = [f'`{title}`', author_str]
    if year:
        parts.append(str(year))
    clean_subjects = [s for s in subjects if len(s) < 30
                      and s.lower() not in ("fiction", "general")][:3]
    if clean_subjects:
        parts.append(", ".join(clean_subjects))
    if rating:
        parts.append(f"{round(rating, 1)}/5")
    else:
        parts.append("?/5")

    desc = " - ".join(parts)
    sort_rating = rating if rating else 0
    return (sort_rating, cover_url, desc, book_id)


def extract_book_detail(book_id):
    """Fetch and format detailed book info for a detail card."""
    try:
        book = hc_book(book_id)
    except Exception:
        return None
    if not book:
        return None

    title = book.get("title") or "Unknown"
    cover_url = book.get("cover_url")
    subjects = (book.get("subjects") or [])[:5]
    desc_raw = book.get("description") or ""
    if len(desc_raw) > 500:
        desc_raw = desc_raw[:497] + "..."

    parts = [f'`{title}`']
    clean = [s for s in subjects if len(s) < 30][:5]
    if clean:
        parts.append(", ".join(clean))
    if desc_raw:
        parts.append(f"\n{desc_raw}")

    return cover_url, "\n".join(parts)
