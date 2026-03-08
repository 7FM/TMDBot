import re

from telegram import Update

from botlib import state
from botlib.config import settings


def get_user_id(update: Update):
    if update.callback_query:
        return update.callback_query.from_user.id
    return update.message.from_user.id


def check_user_invalid(user):
    return user not in settings['allowed_users']


def esc(s):
    # Escape reserved chars for telegram markdown v2, preserving [text](url) links and `code`
    parts = []
    last = 0
    for m in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)|`[^`]+`', s):
        parts.append(_esc_plain(s[last:m.start()]))
        if m.group().startswith('`'):
            parts.append(m.group())
        else:
            text = m.group(1)
            url = m.group(2)
            escaped_url = url.replace('\\', '\\\\').replace(')', '\\)')
            parts.append(f'[{_esc_plain(text)}]({escaped_url})')
        last = m.end()
    parts.append(_esc_plain(s[last:]))
    return "".join(parts)


def _esc_plain(s):
    # Escape all MarkdownV2 special chars in plain text
    special = '_*[]()~`>#+-=|{}.!'
    return "".join(f"\\{c}" if c in special else c for c in s)


def sort_by_rating(movie_list):
    return sorted(movie_list, key=lambda x: x[0], reverse=True)


def get_watched_rating(entry):
    """Extract rating from a watched entry (dict or legacy int/None)."""
    if isinstance(entry, dict):
        return entry.get("rating")
    return entry


def get_watched_category(entry):
    """Extract category from a watched entry."""
    if isinstance(entry, dict):
        return entry.get("category")
    return None


def is_in_any_watchlist(media_id, user, mode=None):
    if mode is None:
        mode = state.user_data[user].get("mode", "movie")
    for wn, w in state.user_data[user]["watchlists"][mode].items():
        if media_id in w:
            return wn
    return None


def find_all_watchlists(media_id, user, mode=None):
    if mode is None:
        mode = state.user_data[user].get("mode", "movie")
    return [wn for wn, w in state.user_data[user]["watchlists"][mode].items() if media_id in w]


def _get_shared_wl(sw_id):
    return state.user_data.get("shared_watchlists", {}).get(sw_id)


def _next_shared_wl_id():
    nid = state.user_data.get("_shared_wl_next_id", 1)
    state.user_data["_shared_wl_next_id"] = nid + 1
    return nid


def _user_shared_watchlists(user):
    return [(sw_id, sw) for sw_id, sw in state.user_data.get("shared_watchlists", {}).items()
            if user in sw.get("members", [])]


def _get_user_display_name(user_id):
    ud = state.user_data.get(user_id, {})
    return ud.get("name") or str(user_id)


def is_in_any_shared_watchlist(media_id, user, mode=None):
    if mode is None:
        mode = state.user_data[user].get("mode", "movie")
    for sw_id, sw in _user_shared_watchlists(user):
        if media_id in sw.get("items", {}).get(mode, []):
            return (sw_id, sw["name"])
    return None


def find_all_shared_watchlists(media_id, user, mode=None):
    if mode is None:
        mode = state.user_data[user].get("mode", "movie")
    return [(sw_id, sw["name"]) for sw_id, sw in _user_shared_watchlists(user)
            if media_id in sw.get("items", {}).get(mode, [])]


def split_into_chunks(text, max_chunk_size=4096):
    chunks = []
    current_position = 0
    text_length = len(text)
    while current_position < text_length:
        if current_position + max_chunk_size >= text_length:
            chunks.append(text[current_position:])
            break
        last_newline = text.rfind(
            '\n', current_position, current_position + max_chunk_size)
        if last_newline == -1:
            last_newline = current_position + max_chunk_size
        chunks.append(text[current_position:last_newline + 1])
        current_position = last_newline + 1
    return chunks


def parse_callback_data(data: str):
    """Parse callback data in format action:type:id or action:type:id:watchlist.
    Returns (action, media_type, media_id, watchlist)."""
    parts = data.split(":", 3)
    if len(parts) < 3:
        return None, None, None, None
    action = parts[0]
    media_type = parts[1]  # "m" or "tv"
    try:
        media_id = int(parts[2])
    except (ValueError, IndexError):
        return None, None, None, None
    watchlist = parts[3] if len(parts) > 3 else None
    return action, media_type, media_id, watchlist


def _mode_to_type(mode):
    """Convert user mode to callback type prefix."""
    return {"movie": "m", "tv": "tv", "book": "b"}.get(mode, mode)


def _type_to_mode(media_type):
    """Convert callback type prefix to user mode."""
    return {"m": "movie", "tv": "tv", "b": "book"}.get(media_type, media_type)
