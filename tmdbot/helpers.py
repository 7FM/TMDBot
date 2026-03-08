import re
import time
import datetime

from telegram import Update

from tmdbot import state
from tmdbot.config import settings, get_api


def get_image_url(path):
    if path:
        return f"https://image.tmdb.org/t/p/original{path}"
    return None


def get_user_id(update: Update):
    if update.callback_query:
        return update.callback_query.from_user.id
    return update.message.from_user.id


def check_user_invalid(user):
    return user not in settings['allowed_users']


def _count_released_seasons(details):
    """Count seasons that have aired (excludes specials and unaired seasons)."""
    try:
        seasons = details.get("seasons") or details["seasons"]
    except (KeyError, AttributeError, TypeError):
        return details.get("number_of_seasons") or 0
    if not seasons:
        return details.get("number_of_seasons") or 0
    today = datetime.date.today().isoformat()
    count = 0
    for s in seasons:
        try:
            sn = s.get("season_number", 0)
            if sn == 0:
                continue
            air_date = s.get("air_date")
        except (AttributeError, TypeError):
            continue
        if air_date and air_date <= today:
            count += 1
    return count if count > 0 else (details.get("number_of_seasons") or 1)


def get_all_movie_provider(region):
    from tmdbot.config import provider as provider_api
    if region in state._provider_cache:
        cached_time, cached_result = state._provider_cache[region]
        if time.time() - cached_time < state._PROVIDER_CACHE_TTL:
            return cached_result
    movie_provider = provider_api.movie_providers(region=region)
    result = []
    if movie_provider:
        movie_provider = movie_provider["results"]
        result = [mp["provider_name"] for mp in movie_provider]
    state._provider_cache[region] = (time.time(), result)
    return result


def get_free_provider(id, country_code, mode="movie"):
    details = get_api(mode).details(id, append_to_response="watch/providers")
    return _parse_providers_from_details(details, country_code)


def _match_providers(my_providers, provider_list):
    """Check which of my_providers match the given provider list."""
    available = []
    if provider_list:
        for p, logo in provider_list:
            for mp in my_providers:
                if p.startswith(mp):
                    available.append((p, logo))
    return len(available) > 0, available


def is_available_for_free(my_providers, id, country_code, mode="movie"):
    prov = get_free_provider(id, country_code, mode=mode)
    return _match_providers(my_providers, prov)


def _parse_providers_from_details(details, country_code):
    """Extract flatrate providers from a details response with watch/providers appended."""
    try:
        wp = details["watch/providers"]
        country_data = wp["results"][country_code]
        return [(p["provider_name"], get_image_url(p["logo_path"]))
                for p in country_data["flatrate"]]
    except (KeyError, TypeError, IndexError, AttributeError):
        return None


def create_available_at_str(provider):
    return "Available at: " + (", ".join([p[0] for p in provider]))


def extract_trailer_url(m, mode="movie"):
    api = get_api(mode)
    if mode == "movie":
        if "trailers" not in m:
            m = api.details(m["id"], append_to_response="trailers")
        if "trailers" in m and "youtube" in m["trailers"]:
            for t in m["trailers"]["youtube"]:
                if t["type"] == "Trailer":
                    return f'https://www.youtube.com/watch?v={t["source"]}'
    else:
        if "videos" not in m:
            m = api.details(m["id"], append_to_response="videos")
        try:
            for v in m["videos"]["results"]:
                if v.get("site") == "YouTube" and v.get("type") == "Trailer":
                    return f'https://www.youtube.com/watch?v={v["key"]}'
        except (KeyError, TypeError, AttributeError):
            pass
    return None


def extract_genre(m, mode="movie"):
    import tmdbot.config as cfg
    gd = cfg.get_genre_dict(mode)
    result = []
    if "genre_ids" in m:
        for g in m["genre_ids"]:
            if g not in gd:
                if mode == "movie":
                    cfg.movie_genre_dict = cfg.get_movie_genres()
                else:
                    cfg.tv_genre_dict = cfg.get_tv_genres()
                gd = cfg.get_genre_dict(mode)
            if g in gd:
                result.append(gd[g])
    elif "genres" in m:
        for g in m["genres"]:
            result.append(g["name"])
    return result


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


def extract_movie_info(m, skip_trailer=False, mode="movie"):
    title = m.get("title") or m.get("name") or "Unknown"
    poster_path = get_image_url(m.get('poster_path'))
    rating = m["vote_average"] if "vote_average" in m and m.get(
        "vote_count", 0) > 0 else None
    if mode == "movie":
        date_str = m.get("release_date") or None
    else:
        date_str = m.get("first_air_date") or None
    if date_str == "":
        date_str = None
    mid = m["id"]
    genres = extract_genre(m, mode=mode)
    trailer = None if skip_trailer else extract_trailer_url(m, mode=mode)
    if trailer:
        title_str = f'[{title}]({trailer})'
    else:
        title_str = f'`{title}`'
    parts = [title_str]
    if date_str:
        parts.append(date_str)
    if mode == "tv":
        seasons = _count_released_seasons(m)
        if seasons:
            parts.append(f"{seasons} season{'s' if seasons != 1 else ''}")
    if genres:
        parts.append(", ".join(genres))
    parts.append((str(round(rating, 1)) if rating else "?") + '/10')
    desc = " - ".join(parts)
    return (rating if rating else 0, poster_path, desc, mid)


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


def is_valid_media_id(media_id, mode="movie"):
    if not media_id.isdigit():
        return "ID is not a number"
    try:
        get_api(mode).details(media_id)
    except Exception:
        return "Movie not found" if mode == "movie" else "TV show not found"
    return None


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
    return "m" if mode == "movie" else "tv"


def _type_to_mode(media_type):
    """Convert callback type prefix to user mode."""
    return "movie" if media_type == "m" else "tv"
