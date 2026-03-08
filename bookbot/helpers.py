from bookbot.config import OL_COVERS, ol_work

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


def get_cover_url(cover_id, size="M"):
    """Get Open Library cover URL. size: S, M, or L."""
    if cover_id:
        return f"{OL_COVERS}/b/id/{cover_id}-{size}.jpg"
    return None


def work_key_to_id(key):
    """Convert OL work key '/works/OL27482W' to numeric ID 27482."""
    if key and key.startswith("/works/OL") and key.endswith("W"):
        try:
            return int(key[len("/works/OL"):-1])
        except ValueError:
            pass
    return None


def id_to_work_key(work_id):
    """Convert numeric ID 27482 to OL work key '/works/OL27482W'."""
    return f"/works/OL{work_id}W"


def extract_book_info(doc, from_search=True):
    """Extract display info from a search result or work detail.

    Returns (rating, cover_url, description_str, work_id).
    """
    if from_search:
        title = doc.get("title", "Unknown")
        authors = doc.get("author_name", [])
        year = doc.get("first_publish_year")
        cover_id = doc.get("cover_i")
        ol_rating = doc.get("ratings_average")
        subjects = doc.get("subject", [])[:3]
        work_id = work_key_to_id(doc.get("key"))
    else:
        title = doc.get("title", "Unknown")
        authors = []
        for a in doc.get("authors", []):
            author_key = a.get("author", {}).get("key", "")
            if author_key:
                authors.append(author_key)
        year = None
        fpd = doc.get("first_publish_date")
        if fpd:
            # Try to extract year from various date formats
            parts = fpd.split()
            for p in reversed(parts):
                if p.isdigit() and len(p) == 4:
                    year = int(p)
                    break
        covers = doc.get("covers", [])
        cover_id = covers[0] if covers else None
        ol_rating = None
        subjects = doc.get("subjects", [])[:3]
        key = doc.get("key", "")
        work_id = work_key_to_id(key)

    cover_url = get_cover_url(cover_id)
    author_str = ", ".join(authors[:2]) if authors else "Unknown author"
    title_str = f'`{title}`'

    parts = [title_str, author_str]
    if year:
        parts.append(str(year))
    if subjects:
        # Clean up subject names — take only short, useful ones
        clean_subjects = [s for s in subjects if len(s) < 30
                          and s.lower() not in ("fiction", "general")][:3]
        if clean_subjects:
            parts.append(", ".join(clean_subjects))
    if ol_rating:
        parts.append(f"{round(ol_rating, 1)}/5")
    else:
        parts.append("?/5")

    desc = " - ".join(parts)
    sort_rating = ol_rating if ol_rating else 0
    return (sort_rating, cover_url, desc, work_id)


def extract_book_detail(work_id):
    """Fetch and format detailed book info for a detail card."""
    try:
        data = ol_work(work_id)
    except Exception:
        return None

    title = data.get("title", "Unknown")
    desc_raw = data.get("description", "")
    if isinstance(desc_raw, dict):
        desc_raw = desc_raw.get("value", "")
    if len(desc_raw) > 500:
        desc_raw = desc_raw[:497] + "..."

    covers = data.get("covers", [])
    cover_url = get_cover_url(covers[0]) if covers else None
    subjects = data.get("subjects", [])[:5]

    parts = [f'`{title}`']
    if subjects:
        clean = [s for s in subjects if len(s) < 30][:5]
        if clean:
            parts.append(", ".join(clean))
    if desc_raw:
        parts.append(f"\n{desc_raw}")

    return cover_url, "\n".join(parts)
