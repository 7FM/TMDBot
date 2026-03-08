import time
import datetime

from botlib import state
from tmdbot.config import get_api

# Re-export all generic helpers from botlib
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


# --- TMDb-specific helpers below ---

def get_image_url(path):
    if path:
        return f"https://image.tmdb.org/t/p/original{path}"
    return None


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


def is_valid_media_id(media_id, mode="movie"):
    if not media_id.isdigit():
        return "ID is not a number"
    try:
        get_api(mode).details(media_id)
    except Exception:
        return "Movie not found" if mode == "movie" else "TV show not found"
    return None
