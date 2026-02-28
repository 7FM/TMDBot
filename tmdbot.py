import sys
import os
import re
import asyncio
import time
import datetime
import yaml
import logging
import random
import concurrent.futures
import multiprocessing
from collections import Counter
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton, LinkPreviewOptions, ForceReply
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, CallbackContext, ContextTypes
from telegram.constants import ParseMode
from tmdbv3api import TMDb, Movie, TV, Search, Genre, Provider

logger = logging.getLogger(__name__)


def load_settings(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


SETTINGS_FILE = "settings.yaml" if len(sys.argv) < 2 else sys.argv[1]
settings = load_settings(SETTINGS_FILE)

# Set up TMDb API
tmdb = TMDb()
tmdb.api_key = settings["tmdb_api_key"]
movie = Movie()
tv = TV()
search = Search()
genre = Genre()
provider = Provider()


def get_api(mode):
    return movie if mode == "movie" else tv


# File path for storing the user data
USER_DATA_FILE = 'user_data.yaml' if len(sys.argv) < 3 else sys.argv[2]
user_data = dict()
region = "DE"


def save_user_data():
    with open(USER_DATA_FILE, 'w') as file:
        yaml.safe_dump(user_data, file)


REGIONS = [
    ("AR", "Argentina"), ("AT", "Austria"), ("AU", "Australia"),
    ("BE", "Belgium"), ("BR", "Brazil"), ("CA", "Canada"),
    ("CH", "Switzerland"), ("CL", "Chile"), ("CO", "Colombia"),
    ("CZ", "Czech Republic"), ("DE", "Germany"), ("DK", "Denmark"),
    ("EC", "Ecuador"), ("EE", "Estonia"), ("ES", "Spain"),
    ("FI", "Finland"), ("FR", "France"), ("GB", "United Kingdom"),
    ("GR", "Greece"), ("HU", "Hungary"), ("ID", "Indonesia"),
    ("IE", "Ireland"), ("IN", "India"), ("IT", "Italy"),
    ("JP", "Japan"), ("KR", "South Korea"), ("LT", "Lithuania"),
    ("LV", "Latvia"), ("MX", "Mexico"), ("MY", "Malaysia"),
    ("NL", "Netherlands"), ("NO", "Norway"), ("NZ", "New Zealand"),
    ("PE", "Peru"), ("PH", "Philippines"), ("PL", "Poland"),
    ("PT", "Portugal"), ("RO", "Romania"), ("RU", "Russia"),
    ("SE", "Sweden"), ("SG", "Singapore"), ("TH", "Thailand"),
    ("TR", "Turkey"), ("TW", "Taiwan"), ("US", "United States"),
    ("VE", "Venezuela"), ("ZA", "South Africa"),
]
REGIONS_PER_PAGE = 8


def _flag_emoji(code):
    return "".join(chr(0x1F1E6 + ord(c) - ord('A')) for c in code)


def _region_name(code):
    return next((name for c, name in REGIONS if c == code), code)


def user_data_initialize():
    for user in settings['allowed_users']:
        if user not in user_data:
            user_data[user] = dict()
            user_data[user]["region"] = region
            user_data[user]["watched"] = {"movie": {}, "tv": {}}
            user_data[user]["watchlists"] = {
                "movie": {"normal": [], "trash": []},
                "tv": {"normal": [], "trash": []},
            }
            user_data[user]["providers"] = []
            user_data[user]["onboarded"] = False
            user_data[user]["mode"] = "movie"
            user_data[user]["tv_season_counts"] = {}
        else:
            ud = user_data[user]
            # Migrate mode
            if "mode" not in ud:
                ud["mode"] = "movie"
            # Migrate flat watchlists to nested
            if "watchlists" in ud and "movie" not in ud["watchlists"]:
                old_wl = ud["watchlists"]
                ud["watchlists"] = {
                    "movie": old_wl,
                    "tv": {"normal": [], "trash": []},
                }
            # Migrate flat watched to nested
            if "watched" in ud and "movie" not in ud["watched"]:
                old_watched = ud["watched"]
                ud["watched"] = {"movie": old_watched, "tv": {}}
            # Migrate tv_season_counts
            if "tv_season_counts" not in ud:
                ud["tv_season_counts"] = {}
    save_user_data()


# Load user data from file if it exists
if os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, 'r') as file:
        user_data = yaml.safe_load(file)
user_data_initialize()


def get_user_id(update: Update):
    if update.callback_query:
        return update.callback_query.from_user.id
    return update.message.from_user.id


def check_user_invalid(user):
    return user not in settings['allowed_users']


def get_movie_genres():
    genre_dict = dict()
    movie_genres = genre.movie_list()["genres"]
    for mg in movie_genres:
        genre_dict[mg["id"]] = mg["name"]
    return genre_dict


def get_tv_genres():
    gd = dict()
    tv_genres = genre.tv_list()["genres"]
    for tg in tv_genres:
        gd[tg["id"]] = tg["name"]
    return gd


movie_genre_dict = get_movie_genres()
tv_genre_dict = get_tv_genres()


def get_genre_dict(mode):
    return movie_genre_dict if mode == "movie" else tv_genre_dict

# Helper functions


def get_image_url(path):
    if path:
        return f"https://image.tmdb.org/t/p/original{path}"
    return None


_provider_cache = {}  # region -> (timestamp, list[str])
_PROVIDER_CACHE_TTL = 86400  # 24 hours
_pending_new_watchlist = {}
_pending_search = {}
_chunk_movies = {}
_chunk_id_counter = 0
_search_results = {}  # user_id -> (chat_id, [message_ids])
_search_more = {}  # user_id -> (remaining_sorted_results, query)
_rate_list_messages = {}  # user_id -> (chat_id, [message_ids])
_rec_genre_filter = {}  # user_id -> {"watchlist": str, "genres": set}
# user_id -> {"mid": int, "watchlist": str|None, "prev_rating": int|None|"absent"}
_last_watched = {}
_pending_season = {}  # user_id -> {"mid": int, "total": int, "media_type": str}

_MODE_SWITCH_TV = "\U0001f4fa Switch to TV"
_MODE_SWITCH_MOVIE = "\U0001f3ac Switch to Movies"


def get_main_keyboard(user):
    mode = user_data.get(user, {}).get("mode", "movie")
    toggle_label = _MODE_SWITCH_TV if mode == "movie" else _MODE_SWITCH_MOVIE
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("/search"), KeyboardButton("/list"),
             KeyboardButton("/check")],
            [KeyboardButton("/recommend"), KeyboardButton("/popular")],
            [KeyboardButton("/pick"), KeyboardButton("/clear"),
             KeyboardButton(toggle_label)],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


def get_all_movie_provider(region):
    if region in _provider_cache:
        cached_time, cached_result = _provider_cache[region]
        if time.time() - cached_time < _PROVIDER_CACHE_TTL:
            return cached_result
    movie_provider = provider.movie_providers(region=region)
    result = []
    if movie_provider:
        movie_provider = movie_provider["results"]
        result = [mp["provider_name"] for mp in movie_provider]
    _provider_cache[region] = (time.time(), result)
    return result


# TODO JustWatch Attribution Required
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
    # Movies use "trailers" key, TV uses "videos"
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
    global movie_genre_dict, tv_genre_dict
    gd = get_genre_dict(mode)
    result = []
    if "genre_ids" in m:
        for g in m["genre_ids"]:
            if g not in gd:
                if mode == "movie":
                    movie_genre_dict = get_movie_genres()
                else:
                    tv_genre_dict = get_tv_genres()
                gd = get_genre_dict(mode)
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
        seasons = m.get("number_of_seasons")
        if seasons:
            parts.append(f"{seasons} season{'s' if seasons != 1 else ''}")
    if genres:
        parts.append(", ".join(genres))
    parts.append((str(round(rating, 1)) if rating else "?") + '/10')
    desc = " - ".join(parts)
    return (rating if rating else 0, poster_path, desc, mid)


def sort_by_rating(movie_list):
    return sorted(movie_list, key=lambda x: x[0], reverse=True)


def is_in_any_watchlist(media_id, user, mode=None):
    if mode is None:
        mode = user_data[user].get("mode", "movie")
    for wn, w in user_data[user]["watchlists"][mode].items():
        if media_id in w:
            return wn
    return None


def find_all_watchlists(media_id, user, mode=None):
    if mode is None:
        mode = user_data[user].get("mode", "movie")
    return [wn for wn, w in user_data[user]["watchlists"][mode].items() if media_id in w]


def is_valid_media_id(media_id, mode="movie"):
    if not media_id.isdigit():
        return "ID is not a number"
    try:
        get_api(mode).details(media_id)
    except Exception:
        return "Movie not found" if mode == "movie" else "TV show not found"
    return None


def split_into_chunks(text, max_chunk_size=4096):
    # Initialize variables
    chunks = []
    current_position = 0
    text_length = len(text)

    while current_position < text_length:
        # Find the maximum possible chunk size
        if current_position + max_chunk_size >= text_length:
            # If the remaining text is smaller than the max_chunk_size, append it as the last chunk
            chunks.append(text[current_position:])
            break

        # Find the last newline character within the max_chunk_size
        last_newline = text.rfind(
            '\n', current_position, current_position + max_chunk_size)

        if last_newline == -1:
            # If no newline found, force split at max_chunk_size (if necessary, but not ideal)
            last_newline = current_position + max_chunk_size

        # Append the chunk up to the last newline or the max_chunk_size
        chunks.append(text[current_position:last_newline + 1])
        # Move current_position to after the newline (or split point)
        current_position = last_newline + 1

    return chunks


def _is_search_message(user, message_id):
    if user not in _search_results:
        return False
    _, msg_ids = _search_results[user]
    return message_id in msg_ids


async def _cleanup_search_results(bot, user):
    _search_more.pop(user, None)
    _pending_search.pop(user, None)
    if user not in _search_results:
        return
    chat_id, msg_ids = _search_results.pop(user)
    for mid in msg_ids:
        try:
            await bot.delete_message(chat_id, mid)
        except Exception:
            pass


async def _cleanup_rate_list(bot, user):
    if user not in _rate_list_messages:
        return
    chat_id, msg_ids = _rate_list_messages.pop(user)
    for mid in msg_ids:
        try:
            await bot.delete_message(chat_id, mid)
        except Exception:
            pass


async def send_back_text(update: Update, msg, user=None):
    if user is None:
        user = get_user_id(update)
    kb = get_main_keyboard(user)
    chunks = split_into_chunks(msg)
    sent = []
    for c in chunks:
        sent.append(await update.message.reply_text(
            esc(c), parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=kb))
    return sent


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


def build_media_keyboard(media_id: int, user: int, mode=None) -> InlineKeyboardMarkup:
    if mode is None:
        mode = user_data[user].get("mode", "movie")
    mt = _mode_to_type(mode)
    buttons = []
    watchlist_name = is_in_any_watchlist(media_id, user, mode=mode)
    if watchlist_name:
        buttons.append(InlineKeyboardButton(
            "Remove", callback_data=f"rm:{mt}:{media_id}"))
    else:
        buttons.append(InlineKeyboardButton(
            "Add", callback_data=f"pick:{mt}:{media_id}"))
    already_watched = media_id in user_data[user].get(
        "watched", {}).get(mode, {})
    if not already_watched:
        buttons.append(InlineKeyboardButton(
            "Watched", callback_data=f"w:{mt}:{media_id}"))
    return InlineKeyboardMarkup([buttons])


def build_watchlist_picker_keyboard(media_id: int, user: int, mode=None) -> InlineKeyboardMarkup:
    if mode is None:
        mode = user_data[user].get("mode", "movie")
    mt = _mode_to_type(mode)
    rows = []
    for wn in user_data[user]["watchlists"][mode]:
        cb_data = f"a:{mt}:{media_id}:{wn}"
        if len(cb_data.encode('utf-8')) > 64:
            continue
        rows.append([InlineKeyboardButton(wn, callback_data=cb_data)])
    rows.append([
        InlineKeyboardButton("New", callback_data=f"new:{mt}:{media_id}"),
        InlineKeyboardButton("Back", callback_data=f"back:{mt}:{media_id}")
    ])
    return InlineKeyboardMarkup(rows)


def build_chunk_keyboard(chunk_id: int, movies_list, expanded: bool, detail_action: str = "det", media_type: str = "m") -> InlineKeyboardMarkup:
    if expanded:
        rows = [[InlineKeyboardButton(
            m_title, callback_data=f"{detail_action}:{media_type}:{m_id}")]
            for m_id, m_title in movies_list]
        rows.append([InlineKeyboardButton(
            "Collapse", callback_data=f"col:{chunk_id}")])
    else:
        label = "Show items" if media_type == "tv" else "Show movies"
        rows = [[InlineKeyboardButton(
            label, callback_data=f"exp:{chunk_id}")]]
    return InlineKeyboardMarkup(rows)


def build_watchlist_select_keyboard(user: int, edit_mode: bool = False, mode=None) -> InlineKeyboardMarkup:
    if mode is None:
        mode = user_data[user].get("mode", "movie")
    wls = user_data[user]["watchlists"][mode]
    rows = []
    for wn in wls:
        if len(f"wl:{wn}".encode('utf-8')) > 64:
            continue
        count = len(wls[wn])
        if edit_mode:
            rows.append([InlineKeyboardButton(
                f"Delete {wn}?", callback_data=f"dwl:{wn}")])
        else:
            rows.append([InlineKeyboardButton(
                f"{wn} ({count})", callback_data=f"wl:{wn}")])
    if edit_mode:
        rows.append([InlineKeyboardButton("Back", callback_data="wlback")])
    else:
        rows.append([
            InlineKeyboardButton("New watchlist", callback_data="nwl"),
            InlineKeyboardButton("Edit", callback_data="wledit")
        ])
    return InlineKeyboardMarkup(rows)


def build_rating_keyboard(media_id: int, media_type: str, action_prefix: str = "rate") -> InlineKeyboardMarkup:
    row1 = [InlineKeyboardButton(str(i), callback_data=f"{action_prefix}:{media_type}:{media_id}:{i}")
            for i in range(1, 6)]
    row2 = [InlineKeyboardButton(str(i), callback_data=f"{action_prefix}:{media_type}:{media_id}:{i}")
            for i in range(6, 11)]
    row3 = [InlineKeyboardButton(
        "Skip", callback_data=f"{action_prefix}:{media_type}:{media_id}:0")]
    return InlineKeyboardMarkup([row1, row2, row3])


def build_season_picker_keyboard(media_id: int, media_type: str, num_seasons: int, action_prefix: str = "ws") -> InlineKeyboardMarkup:
    rows = []
    row = []
    for i in range(1, num_seasons + 1):
        row.append(InlineKeyboardButton(
            f"S{i}", callback_data=f"{action_prefix}:{media_type}:{media_id}:{i}"))
        if len(row) == 5:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton(
        "All seasons", callback_data=f"{action_prefix}:{media_type}:{media_id}:{num_seasons}")])
    return InlineKeyboardMarkup(rows)


def build_region_keyboard(page: int = 0) -> InlineKeyboardMarkup:
    start = page * REGIONS_PER_PAGE
    end = min(start + REGIONS_PER_PAGE, len(REGIONS))
    page_regions = REGIONS[start:end]
    rows = []
    for i in range(0, len(page_regions), 2):
        row = [InlineKeyboardButton(
            f"{_flag_emoji(code)} {name}", callback_data=f"reg:{code}")
            for code, name in page_regions[i:i + 2]]
        rows.append(row)
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(
            "\u25c0 Back", callback_data=f"regp:{page - 1}"))
    if end < len(REGIONS):
        nav.append(InlineKeyboardButton(
            "Next \u25b6", callback_data=f"regp:{page + 1}"))
    if nav:
        rows.append(nav)
    return InlineKeyboardMarkup(rows)


def build_services_keyboard(user: int) -> InlineKeyboardMarkup:
    all_providers = get_all_movie_provider(user_data[user]["region"])
    my_providers = user_data[user]["providers"]
    rows = []
    for i, name in enumerate(all_providers):
        prefix = "\u2705 " if name in my_providers else "\u274c "
        rows.append([InlineKeyboardButton(
            prefix + name, callback_data=f"sp:{i}")])
    region_label = f"{_flag_emoji(user_data[user]['region'])} Region: {_region_name(user_data[user]['region'])}"
    rows.append([InlineKeyboardButton(region_label, callback_data="chreg")])
    return InlineKeyboardMarkup(rows)


def build_genre_picker_keyboard(selected_genres: set, mode="movie") -> InlineKeyboardMarkup:
    rows = []
    genre_items = sorted(get_genre_dict(mode).items(), key=lambda x: x[1])
    for i in range(0, len(genre_items), 2):
        row = []
        for gid, gname in genre_items[i:i + 2]:
            prefix = "\u2705 " if gid in selected_genres else ""
            row.append(InlineKeyboardButton(
                f"{prefix}{gname}", callback_data=f"gf:{gid}"))
        rows.append(row)
    rows.append([
        InlineKeyboardButton("Skip (all genres)", callback_data="recgo:skip"),
        InlineKeyboardButton("Go!", callback_data="recgo:filter")
    ])
    return InlineKeyboardMarkup(rows)


async def send_movie_message(update: Update, caption: str, poster_path, media_id: int, user: int, mode=None):
    keyboard = build_media_keyboard(media_id, user, mode=mode)
    escaped_caption = esc(caption)
    if poster_path:
        return await update.message.reply_photo(
            poster_path,
            escaped_caption,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard
        )
    else:
        return await update.message.reply_text(
            escaped_caption,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard
        )


_CHUNK_MOVIES_MAX = 200


async def send_movie_list(bot, chat_id, header: str, movies_info, detail_action: str = "det", media_type: str = "m"):
    """Send a chunked itemized list with expand/collapse buttons.

    movies_info: list of (media_id, title, description) tuples
    detail_action: callback action prefix for detail buttons (default "det")
    media_type: "m" or "tv" for callback data
    Returns list of sent Message objects.
    """
    global _chunk_id_counter
    if len(_chunk_movies) > _CHUNK_MOVIES_MAX:
        oldest_keys = sorted(_chunk_movies.keys())[
            :len(_chunk_movies) - _CHUNK_MOVIES_MAX]
        for k in oldest_keys:
            del _chunk_movies[k]
    chunk_text = header + "\n"
    chunk_movies_list = []
    max_size = 4096
    sent = []
    for mid, title, desc in movies_info:
        line = f"\u2022 {desc}\n"
        if len(chunk_text) + len(line) > max_size and chunk_movies_list:
            cid = _chunk_id_counter
            _chunk_id_counter += 1
            _chunk_movies[cid] = (chunk_movies_list, detail_action, media_type)
            kb = build_chunk_keyboard(
                cid, chunk_movies_list, expanded=False, media_type=media_type)
            sent.append(await bot.send_message(
                chat_id=chat_id,
                text=esc(chunk_text),
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=kb,
                link_preview_options=LinkPreviewOptions(is_disabled=True)))
            chunk_text = ""
            chunk_movies_list = []
        chunk_text += line
        chunk_movies_list.append((mid, title))
    if chunk_movies_list:
        if len(chunk_movies_list) == 1:
            mid, m_title = chunk_movies_list[0]
            kb = InlineKeyboardMarkup([[InlineKeyboardButton(
                m_title, callback_data=f"{detail_action}:{media_type}:{mid}")]])
        else:
            cid = _chunk_id_counter
            _chunk_id_counter += 1
            _chunk_movies[cid] = (chunk_movies_list, detail_action, media_type)
            kb = build_chunk_keyboard(
                cid, chunk_movies_list, expanded=False, media_type=media_type)
        sent.append(await bot.send_message(
            chat_id=chat_id,
            text=esc(chunk_text),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=kb,
            link_preview_options=LinkPreviewOptions(is_disabled=True)))
    return sent


# Define command handlers
async def unauthorized_msg(update: Update) -> None:
    user_id = get_user_id(update)
    await send_back_text(update, f'*Unauthorized user detected!*\nPlease contact the bot admin to whitelist your user id = `{user_id}`.\nOtherwise, consider hosting your own bot instance. The source code is publicly available at [GitHub](https://github.com/7FM/TMDBot).')


async def list_watchlists(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    keyboard = build_watchlist_select_keyboard(user)
    await update.message.reply_text(
        "Select a watchlist:",
        reply_markup=keyboard
    )


async def show_my_providers(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    keyboard = build_services_keyboard(user)
    await update.message.reply_text(
        "Your streaming services:",
        reply_markup=keyboard
    )


async def add_to_watchlist_helper(watchlist, media_id, user, update: Update):
    mode = user_data[user].get("mode", "movie")
    err_msg = is_valid_media_id(media_id, mode)
    if err_msg:
        await send_back_text(update, f'The provided ID is invalid: ' + err_msg)
        return
    media_id = int(media_id)
    already_in = is_in_any_watchlist(media_id, user, mode=mode)
    if already_in:
        await send_back_text(update, f'Already in your "{already_in}" watchlist.')
    else:
        if media_id in user_data[user]["watched"][mode]:
            await send_back_text(update, "Warning: you have already watched this!")
        user_data[user]["watchlists"][mode][watchlist].append(media_id)
        save_user_data()
        await send_back_text(update, 'Added to watchlist.')


async def add_to_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not context.args:
        await send_back_text(update, 'Please provide the ID.')
        return

    mode = user_data[user].get("mode", "movie")
    watchlist = "normal"
    if len(context.args) == 2:
        watchlist = context.args[1]
    if watchlist not in user_data[user]["watchlists"][mode]:
        await send_back_text(update, f'Info: creating new watchlist "{watchlist}"')
        user_data[user]["watchlists"][mode][watchlist] = []

    media_id = context.args[0]
    await add_to_watchlist_helper(watchlist, media_id, user, update)


async def add_to_trash_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not context.args:
        await send_back_text(update, 'Please provide the ID.')
        return

    media_id = context.args[0]
    await add_to_watchlist_helper("trash", media_id, user, update)


async def add_to_watched(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not context.args:
        await send_back_text(update, 'Please provide the ID.')
        return

    mode = user_data[user].get("mode", "movie")
    mt = _mode_to_type(mode)
    media_id = context.args[0]
    err_msg = is_valid_media_id(media_id, mode)
    if err_msg:
        await send_back_text(update, f'The provided ID is invalid: ' + err_msg)
        return
    media_id = int(media_id)
    if mode == "tv":
        try:
            details = tv.details(media_id)
            num_seasons = details.get("number_of_seasons") or 1
        except Exception:
            num_seasons = 1
        _pending_season[user] = {"mid": media_id,
                                 "total": num_seasons, "media_type": mt}
        keyboard = build_season_picker_keyboard(media_id, mt, num_seasons)
        await update.message.reply_text(
            "Which season did you watch up to?",
            reply_markup=keyboard)
    else:
        if media_id in user_data[user]["watched"][mode]:
            await send_back_text(update, 'Already marked as watched.')
            return
        keyboard = build_rating_keyboard(media_id, mt)
        await update.message.reply_text(
            "Rate this (1\u201310):",
            reply_markup=keyboard)


async def remove_from_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not context.args:
        await send_back_text(update, 'Please provide the ID.')
        return

    mode = user_data[user].get("mode", "movie")
    media_id = context.args[0]
    err_msg = is_valid_media_id(media_id, mode)
    if err_msg:
        await send_back_text(update, f'The provided ID is invalid: ' + err_msg)
        return
    media_id = int(media_id)
    removed_smth = False
    for _, w in user_data[user]["watchlists"][mode].items():
        if media_id in w:
            w.remove(media_id)
            removed_smth = True
    if removed_smth:
        save_user_data()
        await send_back_text(update, 'Removed from watchlist.')
    else:
        await send_back_text(update, 'Not in any watchlist.')


async def _do_recommend(bot, chat_id, user, watchlist, genre_filter=None):
    """Core recommendation logic. genre_filter is a set of genre IDs or None for all."""
    mode = user_data[user].get("mode", "movie")
    mt = _mode_to_type(mode)
    api = get_api(mode)
    sources = [(mid, 1.0)
               for mid in user_data[user]["watchlists"][mode][watchlist]]
    rated_watched = sorted(
        [(mid, r) for mid, r in user_data[user]["watched"][mode].items()
         if r is not None and r >= 7],
        key=lambda x: -x[1])[:20]
    sources.extend((mid, rating / 10.0) for mid, rating in rated_watched)

    if not sources:
        await bot.send_message(
            chat_id,
            esc(f'Your "{watchlist}" watchlist is empty and you have no highly-rated watched items.'),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=get_main_keyboard(user))
        return

    total = len(sources)

    def query_recommendations(source):
        source_id, weight = source
        available_recommendations = []
        results = api.recommendations(source_id)
        items = results["results"]
        for m in items:
            if genre_filter:
                item_genres = set(m.get("genre_ids", []))
                if not item_genres.intersection(genre_filter):
                    continue
            in_watchlist = is_in_any_watchlist(m["id"], user, mode=mode)
            if not in_watchlist and m["id"] not in user_data[user]["watched"][mode]:
                available, prov = is_available_for_free(
                    user_data[user]["providers"], m["id"], user_data[user]["region"], mode=mode)
                if available:
                    popularity, poster, desc, mid = extract_movie_info(
                        m, skip_trailer=True, mode=mode)
                    title = m.get("title") or m.get("name") or "Unknown"
                    available_recommendations.append(
                        (popularity, poster, desc, prov, mid, title, weight))
        return available_recommendations

    def do_recommend(tick):
        num_threads = min(multiprocessing.cpu_count(), 8)
        all_recs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for result in executor.map(query_recommendations, sources):
                all_recs.extend(result)
                tick()
        return all_recs

    available_recommendations = await _with_progress_bar(
        bot, chat_id, "Finding recommendations...", total, do_recommend)

    id_scores = {}
    for item in available_recommendations:
        mid = item[4]
        id_scores[mid] = id_scores.get(mid, 0) + item[6]

    def custom_sort_key(item):
        return (-id_scores[item[4]], -item[0])

    seen_ids = set()
    unique_tuples = []
    for tuple_item in available_recommendations:
        if tuple_item[4] not in seen_ids:
            seen_ids.add(tuple_item[4])
            unique_tuples.append(tuple_item)

    available_recommendations = sorted(unique_tuples, key=custom_sort_key)

    num_rec = min(50, len(available_recommendations))
    if available_recommendations:
        movies_info = []
        for _, poster_path, caption, prov, mid, title, _ in available_recommendations[:num_rec]:
            provider_str = create_available_at_str(prov)
            movies_info.append((mid, title, caption + "\n" + provider_str))
        label = "Recommended based on" if mode == "movie" else "Recommended shows based on"
        await send_movie_list(
            bot, chat_id,
            f'{label} your "{watchlist}" watchlist:',
            movies_info, media_type=mt)
    else:
        await bot.send_message(
            chat_id,
            esc(
                f'No recommendations found based on your "{watchlist}" watchlist.'),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=get_main_keyboard(user))


async def recommend(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    mode = user_data[user].get("mode", "movie")
    watchlist = "normal"
    if context.args:
        watchlist = context.args[0]

    if watchlist not in user_data[user]["watchlists"][mode]:
        await send_back_text(update, f'Watchlist "{watchlist}" not found.')
        return

    _rec_genre_filter[user] = {"watchlist": watchlist, "genres": set()}
    keyboard = build_genre_picker_keyboard(set(), mode=mode)
    await update.message.reply_text(
        "Filter recommendations by genre (or skip for all):",
        reply_markup=keyboard)


async def check_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    my_providers = user_data[user]["providers"]
    if not my_providers:
        await send_back_text(update, "Set up your streaming services first with /services.")
        return

    mode = user_data[user].get("mode", "movie")
    mt = _mode_to_type(mode)
    api = get_api(mode)

    def fetch_details(media_id):
        details = api.details(
            media_id, append_to_response="watch/providers")
        providers = _parse_providers_from_details(
            details, user_data[user]["region"])
        avail, matched = _match_providers(my_providers, providers)
        if avail:
            return (media_id, matched, details)
        return None

    bot = update.get_bot()
    chat_id = update.message.chat_id
    wls = user_data[user]["watchlists"][mode]
    total = sum(len(w) for w in wls.values())

    watchlists_snapshot = list(wls.items())

    def do_check(tick):
        num_threads = min(multiprocessing.cpu_count(), 8)
        available_items = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for wn, w in watchlists_snapshot:
                results = []
                for r in executor.map(fetch_details, w):
                    if r:
                        results.append(r)
                    tick()
                available_items.append((wn, results))
        return available_items

    available_items = await _with_progress_bar(
        bot, chat_id, "Checking streaming availability...", total, do_check)

    for wn, items in available_items:
        if items:
            movies_info = []
            for media_id, prov, details in items:
                _, _, desc, mid = extract_movie_info(
                    details, skip_trailer=True, mode=mode)
                title = details.get("title") or details.get(
                    "name") or "Unknown"
                provider_str = create_available_at_str(prov)
                movies_info.append((mid, title, desc + "\n" + provider_str))
            await send_movie_list(
                update.get_bot(), update.message.chat_id,
                f'Items on your {wn} watchlist available on streaming services:',
                movies_info, media_type=mt)
        else:
            await send_back_text(update, f'None of the items on your {wn} watchlist are available on streaming services.')


async def popular_movies(update: Update, context: CallbackContext) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not user_data[user]["providers"]:
        await send_back_text(update, "Set up your streaming services first with /services.")
        return

    mode = user_data[user].get("mode", "movie")
    mt = _mode_to_type(mode)
    api = get_api(mode)
    label = "Popular movies" if mode == "movie" else "Popular TV shows"

    status_msg = await update.message.reply_text(
        f"Finding {label.lower()}...", reply_markup=get_main_keyboard(user))
    target_count = 10
    page = 1
    results = api.popular(page=page)
    total_pages = results["total_pages"]
    candidates = []
    max_candidates = target_count * 3
    while page < total_pages and len(candidates) < max_candidates:
        if page != 1:
            results = api.popular(page=page)
        for m in results["results"]:
            if m["id"] not in user_data[user]["watched"][mode]:
                candidates.append(m)
        page += 1

    my_providers = user_data[user]["providers"]
    user_region = user_data[user]["region"]

    def check_popular_item(m):
        available, prov = is_available_for_free(
            my_providers, m["id"], user_region, mode=mode)
        if available:
            _, poster, desc, mid = extract_movie_info(
                m, skip_trailer=True, mode=mode)
            title = m.get("title") or m.get("name") or "Unknown"
            return (mid, title, desc, prov)
        return None

    num_threads = min(multiprocessing.cpu_count(), 8)
    pop_items = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(check_popular_item, m) for m in candidates]
        for future in futures:
            if future.cancelled():
                continue
            result = future.result()
            if result:
                pop_items.append(result)
                if len(pop_items) >= target_count:
                    for f in futures:
                        f.cancel()
                    break

    try:
        await status_msg.delete()
    except Exception:
        pass

    if pop_items:
        movies_info = []
        for mid, title, caption, prov in pop_items[:target_count]:
            provider_str = create_available_at_str(prov)
            movies_info.append((mid, title, caption + "\n" + provider_str))
        await send_movie_list(
            update.get_bot(), update.message.chat_id,
            f'{label} available on your streaming services:',
            movies_info, media_type=mt)
    else:
        await send_back_text(update, f'No {label.lower()} found on your streaming services.')


def _progress_bar(done, total, width=10):
    """Build a text progress bar like [#####-----] 5/10."""
    filled = round(width * done / total) if total else 0
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"[{bar}] {done}/{total}"


async def _with_progress_bar(bot, chat_id, label, total, work_fn):
    """Run work_fn in a background thread with live progress bar.
    work_fn(tick) should call tick() after each unit of work completes."""
    counter = [0]
    msg = await bot.send_message(
        chat_id, f"{label}\n{_progress_bar(0, total)}")
    msg_id = msg.message_id

    def tick():
        counter[0] += 1

    async def updater():
        last = 0
        while counter[0] < total:
            if counter[0] != last:
                try:
                    await bot.edit_message_text(
                        text=f"{label}\n{_progress_bar(counter[0], total)}",
                        chat_id=chat_id, message_id=msg_id)
                except Exception:
                    pass
                last = counter[0]
            await asyncio.sleep(0.5)

    task = asyncio.create_task(updater())
    result = await asyncio.to_thread(work_fn, tick)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # Final update
    try:
        await bot.edit_message_text(
            text=f"{label}\n{_progress_bar(counter[0], total)}",
            chat_id=chat_id, message_id=msg_id)
    except Exception as e:
        pass
    # Delete progress message
    try:
        await bot.delete_message(chat_id, msg_id)
    except Exception:
        pass
    return result


def _check_new_seasons_for_user(user, tick=None):
    """Check watched TV shows for new seasons. Returns (new_list, newly_recorded).
    new_list: [(media_id, title, old_total, new_total, watched_season, details), ...]
    newly_recorded: count of shows that had no prior season data (baseline recorded)."""
    watched_tv = user_data[user].get("watched", {}).get("tv", {})
    stored = user_data[user].get("tv_season_counts", {})
    new_list = []
    newly_recorded = 0
    num_threads = min(multiprocessing.cpu_count(), 8)

    def fetch_show(mid):
        try:
            details = tv.details(mid)
            if tick:
                tick()
            return (mid, details)
        except Exception:
            if tick:
                tick()
            return (mid, None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for mid, details in executor.map(fetch_show, list(watched_tv.keys())):
            if details is None:
                continue
            title = details.get("name") or details.get("title") or "Unknown"
            current_total = details.get("number_of_seasons") or 0
            if mid in stored:
                old_total = stored[mid].get("total", 0)
                watched_season = stored[mid].get("watched", old_total)
                if current_total > old_total:
                    new_list.append(
                        (mid, title, old_total, current_total, watched_season, details))
                stored[mid]["total"] = current_total
            else:
                stored[mid] = {"total": current_total,
                               "watched": current_total}
                newly_recorded += 1
    user_data[user]["tv_season_counts"] = stored
    return new_list, newly_recorded


async def new_seasons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    watched_tv = user_data[user].get("watched", {}).get("tv", {})
    if not watched_tv:
        await send_back_text(update, "You have no watched TV shows.")
        return

    bot = update.get_bot()
    chat_id = update.message.chat_id
    total = len(watched_tv)

    def do_check(tick):
        return _check_new_seasons_for_user(user, tick)

    new_list, newly_recorded = await _with_progress_bar(
        bot, chat_id, "Checking for new seasons\u2026", total, do_check)
    save_user_data()

    parts = []
    if new_list:
        movies_info = []
        for mid, title, old_total, new_total, watched_season, details in new_list:
            _, _, desc, _ = extract_movie_info(
                details, skip_trailer=True, mode="tv")
            diff = new_total - old_total
            season_word = "season" if diff == 1 else "seasons"
            extra = f"\u2728 {diff} new {season_word} (you watched S{watched_season}, now has {new_total} seasons)"
            movies_info.append((mid, title, desc + "\n" + extra))
        await send_movie_list(bot, chat_id,
                              f"New seasons available for {len(new_list)} show(s):",
                              movies_info, media_type="tv")
    else:
        parts.append("No new seasons detected for your watched shows.")

    if newly_recorded:
        parts.append(f"Recorded season data for {newly_recorded} new show(s).")

    if parts:
        await send_back_text(update, " ".join(parts))


async def _daily_season_check(context: ContextTypes.DEFAULT_TYPE):
    """Daily job: check all users' watched TV shows for new seasons."""
    for user in settings["allowed_users"]:
        watched_tv = user_data[user].get("watched", {}).get("tv", {})
        if not watched_tv:
            continue
        new_list, _ = await asyncio.to_thread(
            _check_new_seasons_for_user, user)
        if not new_list:
            continue
        save_user_data()
        movies_info = []
        for mid, title, old_total, new_total, watched_season, details in new_list:
            _, _, desc, _ = extract_movie_info(
                details, skip_trailer=True, mode="tv")
            diff = new_total - old_total
            season_word = "season" if diff == 1 else "seasons"
            extra = f"\u2728 {diff} new {season_word} (you watched S{watched_season}, now has {new_total} seasons)"
            movies_info.append((mid, title, desc + "\n" + extra))
        await send_movie_list(context.bot, user,
                              f"New seasons available for {len(new_list)} show(s):",
                              movies_info, media_type="tv")


async def view_seasons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    watched_tv = user_data[user].get("watched", {}).get("tv", {})
    if not watched_tv:
        await send_back_text(update, "You have no watched TV shows.")
        return

    stored = user_data[user].get("tv_season_counts", {})
    movies_info = []
    for mid, rating in watched_tv.items():
        try:
            details = tv.details(mid)
        except Exception:
            continue
        title = details.get("name") or details.get("title") or "Unknown"
        _, _, desc, _ = extract_movie_info(
            details, skip_trailer=True, mode="tv")
        season_data = stored.get(mid)
        if season_data:
            watched_s = season_data.get("watched", "?")
            total_s = season_data.get("total", "?")
            season_str = f"Watched: S{watched_s}/{total_s}"
        else:
            total_s = details.get("number_of_seasons") or "?"
            season_str = f"Watched: ?/{total_s}"
        rating_str = f"{rating}/10" if rating else "unrated"
        movies_info.append(
            (mid, title, f"{desc}\n{season_str} - {rating_str}"))

    await send_movie_list(
        update.get_bot(), update.message.chat_id,
        f"Season tracking for {len(movies_info)} TV show(s):",
        movies_info, detail_action="sdet", media_type="tv")


def _collect_pick_candidates(user, watchlist=None):
    """Collect IDs from one or all watchlists for current mode. Returns (candidates, label)."""
    mode = user_data[user].get("mode", "movie")
    wls = user_data[user]["watchlists"][mode]
    if watchlist:
        wl = wls.get(watchlist, [])
        return list(wl), f'"{watchlist}"'
    all_items = []
    for wl in wls.values():
        all_items.extend(wl)
    return list(set(all_items)), "your watchlists"


async def _do_pick(bot, chat_id, user, candidates, label, wl_cb_name):
    """Shared logic for picking a random available item from candidates.
    Returns True if an item was sent, False if none available."""
    mode = user_data[user].get("mode", "movie")
    mt = _mode_to_type(mode)
    api = get_api(mode)
    my_providers = user_data[user]["providers"]
    user_region = user_data[user]["region"]
    random.shuffle(candidates)

    picked_mid = None
    picked_details = None
    matched_providers = None
    for mid in candidates:
        details = api.details(mid, append_to_response="watch/providers")
        providers = _parse_providers_from_details(details, user_region)
        avail, matched = _match_providers(my_providers, providers)
        if avail:
            picked_mid = mid
            picked_details = details
            matched_providers = matched
            break

    if picked_mid is None:
        await bot.send_message(
            chat_id, esc(f'Nothing in {label} is available on your services.'),
            parse_mode=ParseMode.MARKDOWN_V2, reply_markup=get_main_keyboard(user))
        return False

    _, poster_path, desc, _ = extract_movie_info(picked_details, mode=mode)
    desc += "\n" + create_available_at_str(matched_providers)
    keyboard = build_media_keyboard(picked_mid, user, mode=mode)
    pick_cb = f"rpick:{wl_cb_name}"
    if len(pick_cb.encode('utf-8')) <= 64:
        rows = list(keyboard.inline_keyboard) + [
            [InlineKeyboardButton("Pick another", callback_data=pick_cb)]]
        keyboard = InlineKeyboardMarkup(rows)
    escaped = esc(desc)
    if poster_path:
        await bot.send_photo(
            chat_id, poster_path,
            caption=escaped,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard)
    else:
        await bot.send_message(
            chat_id, escaped,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard)
    return True


async def pick_movie(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not user_data[user]["providers"]:
        await send_back_text(update, "Set up your streaming services first with /services.")
        return

    mode = user_data[user].get("mode", "movie")
    watchlist = context.args[0] if context.args else None
    if watchlist and watchlist not in user_data[user]["watchlists"][mode]:
        await send_back_text(update, f'Watchlist "{watchlist}" not found.')
        return

    candidates, label = _collect_pick_candidates(user, watchlist)
    if not candidates:
        await send_back_text(update, f'{label} is empty.' if watchlist else 'All your watchlists are empty.')
        return

    await _do_pick(update.get_bot(), update.message.chat_id, user,
                   candidates, label, watchlist or '*')


async def clear_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    chat_id = update.message.chat_id
    current_id = update.message.message_id
    bot = update.get_bot()

    # Clear tracked state
    await _cleanup_search_results(bot, user)
    await _cleanup_rate_list(bot, user)
    _rec_genre_filter.pop(user, None)
    _last_watched.pop(user, None)

    deleted = 0
    consecutive_fails = 0
    for msg_id in range(current_id, max(current_id - 500, 0), -1):
        try:
            await bot.delete_message(chat_id, msg_id)
            deleted += 1
            consecutive_fails = 0
        except Exception:
            consecutive_fails += 1
            if consecutive_fails >= 30:
                break

    await bot.send_message(
        chat_id,
        f"Cleared {deleted} messages.",
        reply_markup=get_main_keyboard(user))


async def fix_keyboard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return
    await update.message.reply_text("Keyboard restored.", reply_markup=get_main_keyboard(user))


async def toggle_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return
    current = user_data[user].get("mode", "movie")
    new_mode = "tv" if current == "movie" else "movie"
    user_data[user]["mode"] = new_mode
    save_user_data()
    label = "TV Shows" if new_mode == "tv" else "Movies"
    await update.message.reply_text(
        f"Switched to {label} mode.",
        reply_markup=get_main_keyboard(user))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not user_data[user].get("onboarded", False):
        keyboard = build_region_keyboard()
        await update.message.reply_text(
            "Welcome to TMDBot! Let's get you set up.\n\nSelect your region:",
            reply_markup=keyboard)
    else:
        await update.message.reply_text(
            'Welcome back to TMDBot! Use /search to search for a movie.',
            reply_markup=get_main_keyboard(user)
        )


async def _send_rate_list(bot, chat_id, user):
    """Build and send the rate list for a user. Returns sent messages."""
    mode = user_data[user].get("mode", "movie")
    mt = _mode_to_type(mode)
    api = get_api(mode)
    watched = user_data[user]["watched"][mode]
    unrated = [(mid, r) for mid, r in watched.items() if r is None]
    rated = sorted(
        [(mid, r) for mid, r in watched.items() if r is not None],
        key=lambda x: -x[1])

    movies_info = []
    for mid, r in unrated + rated:
        details = api.details(mid)
        _, _, desc, _ = extract_movie_info(
            details, skip_trailer=True, mode=mode)
        title = details.get("title") or details.get("name") or "Unknown"
        rating_str = "unrated" if r is None else f"Your rating: {r}/10"
        movies_info.append((mid, title, f"{desc}\n{rating_str}"))

    n_unrated = len(unrated)
    n_rated = len(rated)
    label = "movie" if mode == "movie" else "show"
    if n_unrated > 0:
        header = f'{n_unrated} unrated, {n_rated} rated {label}(s):'
    else:
        header = f'{n_rated} watched {label}(s):'
    return await send_movie_list(
        bot, chat_id, header, movies_info, detail_action="rdet", media_type=mt)


async def rate_movies(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    mode = user_data[user].get("mode", "movie")
    if not user_data[user]["watched"][mode]:
        await send_back_text(update, "You haven't watched anything in this mode yet!")
        return

    # Clean up previous rate list
    await _cleanup_rate_list(update.get_bot(), user)

    msgs = await _send_rate_list(
        update.get_bot(), update.message.chat_id, user)
    _rate_list_messages[user] = (update.message.chat_id, [
                                 m.message_id for m in msgs])


async def do_search(update: Update, query: str, user: int) -> None:
    mode = user_data[user].get("mode", "movie")
    if mode == "movie":
        results = search.movies(query)
    else:
        results = search.tv_shows(query)
    if results and results["total_results"] > 0:
        await _cleanup_search_results(update.get_bot(), user)
        batch = []
        msgs = await send_back_text(update, f'Search results for "{query}":')
        batch.extend(m.message_id for m in msgs)
        items = results["results"]
        res = []
        for m in items:
            res.append(extract_movie_info(m, mode=mode))
        sorted_res = sort_by_rating(res)
        show_results = min(5, len(sorted_res))
        for _, poster_path, caption, mid in sorted_res[:show_results]:
            msg = await send_movie_message(update, caption, poster_path, mid, user, mode=mode)
            batch.append(msg.message_id)
        remaining = sorted_res[show_results:]
        if remaining:
            _search_more[user] = (remaining, query)
            btn = InlineKeyboardButton(
                f"Show more ({len(remaining)} remaining)",
                callback_data="smore")
            msg = await update.message.reply_text(
                "More results available:",
                reply_markup=InlineKeyboardMarkup([[btn]]))
            batch.append(msg.message_id)
        _search_results[user] = (update.message.chat_id, batch)
    else:
        await send_back_text(update, 'No results found.')


async def search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not context.args:
        _pending_search[user] = True
        await update.message.reply_text(
            "Enter a movie title to search:",
            reply_markup=ForceReply(selective=True))
        return

    query = ' '.join(context.args)
    await do_search(update, query, user)


async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = query.from_user.id

    if check_user_invalid(user):
        await query.answer("Unauthorized user.")
        return

    raw = query.data
    action = raw.split(":", 1)[0]

    if action == "nwl":
        mode = user_data[user].get("mode", "movie")
        _pending_new_watchlist[user] = (None, mode)
        await query.answer()
        await query.get_bot().send_message(
            query.message.chat_id,
            "Enter a name for the new watchlist:",
            reply_markup=ForceReply(selective=True))
        return

    if action == "wledit":
        keyboard = build_watchlist_select_keyboard(user, edit_mode=True)
        await query.edit_message_reply_markup(reply_markup=keyboard)
        await query.answer()
        return

    if action == "wlback":
        keyboard = build_watchlist_select_keyboard(user)
        await query.edit_message_reply_markup(reply_markup=keyboard)
        await query.answer()
        return

    if action == "dwl":
        wl_name = raw.split(":", 1)[1]
        await query.answer()
        mode = user_data[user].get("mode", "movie")
        count = len(user_data[user]["watchlists"][mode].get(wl_name, []))
        if count > 0:
            confirm_text = f'Delete "{wl_name}"? It contains {count} movie{"s" if count != 1 else ""}.'
        else:
            confirm_text = f'Delete "{wl_name}"?'
        confirm_kb = InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "Yes, delete", callback_data=f"dwly:{wl_name}"),
            InlineKeyboardButton("Cancel", callback_data="dwln")
        ]])
        await query.edit_message_text(
            text=esc(confirm_text),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=confirm_kb)
        return

    if action == "dwly":
        wl_name = raw.split(":", 1)[1]
        mode = user_data[user].get("mode", "movie")
        if wl_name in user_data[user]["watchlists"][mode]:
            del user_data[user]["watchlists"][mode][wl_name]
            save_user_data()
            await query.answer(f'Deleted "{wl_name}".')
        else:
            await query.answer("Watchlist not found.", show_alert=True)
        keyboard = build_watchlist_select_keyboard(user, edit_mode=True)
        await query.edit_message_text(
            text="Select a watchlist:",
            reply_markup=keyboard)
        return

    if action == "dwln":
        keyboard = build_watchlist_select_keyboard(user, edit_mode=True)
        await query.edit_message_text(
            text="Select a watchlist:",
            reply_markup=keyboard)
        await query.answer()
        return

    if action == "reg":
        code = raw.split(":", 1)[1]
        user_data[user]["region"] = code
        _provider_cache.pop(code, None)
        save_user_data()
        keyboard = build_services_keyboard(user)
        await query.edit_message_text(
            f"Region set to {_flag_emoji(code)} {_region_name(code)}.\n\nSelect your streaming services:",
            reply_markup=keyboard)
        await query.answer()
        return

    if action == "regp":
        page = int(raw.split(":", 1)[1])
        keyboard = build_region_keyboard(page)
        await query.edit_message_reply_markup(reply_markup=keyboard)
        await query.answer()
        return

    if action == "chreg":
        keyboard = build_region_keyboard()
        await query.edit_message_text(
            "Select your region:",
            reply_markup=keyboard)
        await query.answer()
        return

    if action == "gf":
        genre_id = int(raw.split(":", 1)[1])
        if user not in _rec_genre_filter:
            await query.answer("Session expired.", show_alert=True)
            return
        genres = _rec_genre_filter[user]["genres"]
        if genre_id in genres:
            genres.discard(genre_id)
        else:
            genres.add(genre_id)
        mode = user_data[user].get("mode", "movie")
        keyboard = build_genre_picker_keyboard(genres, mode=mode)
        await query.edit_message_reply_markup(reply_markup=keyboard)
        await query.answer()
        return

    if action == "recgo":
        if user not in _rec_genre_filter:
            await query.answer("Session expired.", show_alert=True)
            return
        state = _rec_genre_filter.pop(user)
        recgo_mode = raw.split(":", 1)[1]
        genre_filter = state["genres"] if recgo_mode == "filter" and state["genres"] else None
        await query.answer()
        try:
            await query.message.delete()
        except Exception:
            pass
        await _do_recommend(
            query.get_bot(), query.message.chat_id,
            user, state["watchlist"], genre_filter=genre_filter)
        return

    if action == "smore":
        if user not in _search_more:
            await query.answer("No more results.", show_alert=True)
            return
        remaining, search_query = _search_more[user]
        # Delete the "Show more" button message
        try:
            await query.message.delete()
        except Exception:
            pass
        # Remove the old button message from tracking
        if user in _search_results:
            chat_id, msg_ids = _search_results[user]
            if query.message.message_id in msg_ids:
                msg_ids.remove(query.message.message_id)
        next_batch = remaining[:5]
        new_remaining = remaining[5:]
        await query.answer()
        bot = query.get_bot()
        chat_id = query.message.chat_id
        batch = _search_results[user][1] if user in _search_results else []
        mode = user_data[user].get("mode", "movie")
        for _, poster_path, caption, mid in next_batch:
            keyboard = build_media_keyboard(mid, user, mode=mode)
            escaped = esc(caption)
            if poster_path:
                msg = await bot.send_photo(
                    chat_id, poster_path, escaped,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_markup=keyboard)
            else:
                msg = await bot.send_message(
                    chat_id, escaped,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_markup=keyboard)
            batch.append(msg.message_id)
        if new_remaining:
            _search_more[user] = (new_remaining, search_query)
            btn = InlineKeyboardButton(
                f"Show more ({len(new_remaining)} remaining)",
                callback_data="smore")
            msg = await bot.send_message(
                chat_id, "More results available:",
                reply_markup=InlineKeyboardMarkup([[btn]]))
            batch.append(msg.message_id)
        else:
            _search_more.pop(user, None)
        _search_results[user] = (chat_id, batch)
        return

    if action == "undo":
        if user not in _last_watched:
            await query.answer("Nothing to undo.", show_alert=True)
            return
        state = _last_watched.pop(user)
        mid = state["mid"]
        undo_mode = state.get("mode", "movie")
        prev_wls = state.get("watchlists", [])
        # Backwards compat: old state used singular "watchlist"
        if not prev_wls and state.get("watchlist"):
            prev_wls = [state["watchlist"]]
        prev_rating = state["prev_rating"]
        # Restore previous watched state
        if prev_rating == "absent":
            user_data[user]["watched"][undo_mode].pop(mid, None)
        else:
            user_data[user]["watched"][undo_mode][mid] = prev_rating
        # Restore to all watchlists it was in
        for wn in prev_wls:
            if wn in user_data[user]["watchlists"][undo_mode]:
                user_data[user]["watchlists"][undo_mode][wn].append(mid)
        # Restore tv_season_counts
        prev_season_data = state.get("prev_season_data")
        if undo_mode == "tv" and prev_season_data is not None:
            if prev_season_data == "absent":
                user_data[user]["tv_season_counts"].pop(mid, None)
            else:
                user_data[user]["tv_season_counts"][mid] = prev_season_data
        save_user_data()
        await query.answer("Undone!")
        try:
            await query.message.delete()
        except Exception:
            await query.edit_message_reply_markup(reply_markup=None)
        await query.get_bot().send_message(
            query.message.chat_id,
            "Last watched action undone.",
            reply_markup=get_main_keyboard(user))
        # Refresh rate list if open
        if user in _rate_list_messages:
            bot = query.get_bot()
            chat_id = query.message.chat_id
            await _cleanup_rate_list(bot, user)
            msgs = await _send_rate_list(bot, chat_id, user)
            _rate_list_messages[user] = (chat_id, [m.message_id for m in msgs])
        return

    if action in ("rate", "rrate"):
        parts = raw.split(":")
        try:
            mt = parts[1]
            mid = int(parts[2])
            rating = int(parts[3]) or None
        except (ValueError, IndexError):
            await query.answer("Invalid action.", show_alert=True)
            return
        rate_mode = _type_to_mode(mt)
        # Save undo state
        prev_wls = find_all_watchlists(mid, user, mode=rate_mode)
        prev_rating = user_data[user]["watched"][rate_mode].get(mid, "absent")
        prev_season_data = user_data[user]["tv_season_counts"].get(
            mid, "absent") if rate_mode == "tv" else None
        _last_watched[user] = {"mid": mid, "watchlists": prev_wls,
                               "prev_rating": prev_rating, "prev_season_data": prev_season_data, "mode": rate_mode}
        for wn in prev_wls:
            user_data[user]["watchlists"][rate_mode][wn].remove(mid)
        user_data[user]["watched"][rate_mode][mid] = rating
        # Save season tracking data for TV shows
        if rate_mode == "tv" and user in _pending_season and _pending_season[user]["mid"] == mid:
            pending = _pending_season.pop(user)
            user_data[user]["tv_season_counts"][mid] = {
                "total": pending["total"],
                "watched": pending.get("season", pending["total"]),
            }
        save_user_data()
        if rating:
            await query.answer(f"Rated {rating}/10 and marked as watched.")
        else:
            await query.answer("Marked as watched.")
        chat_id = query.message.chat_id
        bot = query.get_bot()
        undo_kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Undo", callback_data="undo")]])
        if _is_search_message(user, query.message.message_id):
            await _cleanup_search_results(bot, user)
            await bot.send_message(
                chat_id, "Marked as watched.",
                reply_markup=get_main_keyboard(user))
            await bot.send_message(
                chat_id, "Undo?",
                reply_markup=undo_kb)
        else:
            try:
                await query.message.delete()
            except Exception:
                await query.edit_message_reply_markup(reply_markup=None)
            await bot.send_message(
                chat_id, "Marked as watched.",
                reply_markup=get_main_keyboard(user))
            await bot.send_message(
                chat_id, "Undo?",
                reply_markup=undo_kb)
        # Refresh rate list only if rating came from /rate flow
        if action == "rrate" and user in _rate_list_messages:
            await _cleanup_rate_list(bot, user)
            msgs = await _send_rate_list(bot, chat_id, user)
            _rate_list_messages[user] = (chat_id, [m.message_id for m in msgs])
        return

    if action == "wl":
        wl_name = raw.split(":", 1)[1]
        mode = user_data[user].get("mode", "movie")
        wl_movies = user_data[user]["watchlists"][mode].get(wl_name, [])
        if not wl_movies:
            await query.answer(f'"{wl_name}" is empty.', show_alert=True)
            return
        await query.answer()
        api = get_api(mode)
        mt = _mode_to_type(mode)
        movies_info = []
        for mid in wl_movies:
            details = api.details(mid)
            _, _, desc, _ = extract_movie_info(details, mode=mode)
            title = details.get("title") or details.get("name") or "Unknown"
            movies_info.append((mid, title, desc))
        await send_movie_list(query.get_bot(), query.message.chat_id, f'{wl_name} watchlist:', movies_info, media_type=mt)
        return

    if action in ("exp", "col"):
        cid = int(raw.split(":", 1)[1])
        if cid not in _chunk_movies:
            await query.answer("Session expired.", show_alert=True)
            return
        expanded = action == "exp"
        chunk_movies_list, det_action, chunk_mt = _chunk_movies[cid]
        kb = build_chunk_keyboard(
            cid, chunk_movies_list, expanded=expanded, detail_action=det_action, media_type=chunk_mt)
        await query.edit_message_reply_markup(reply_markup=kb)
        await query.answer()
        return

    if action in ("det", "rdet"):
        parts = raw.split(":", 2)
        if len(parts) < 3:
            await query.answer("Invalid action.", show_alert=True)
            return
        det_mt = parts[1]
        mid = int(parts[2])
        det_mode = _type_to_mode(det_mt)
        api = get_api(det_mode)
        details = api.details(mid)
        _, poster_path, desc, _ = extract_movie_info(details, mode=det_mode)
        if action == "rdet":
            keyboard = build_rating_keyboard(
                mid, media_type=det_mt, action_prefix="rrate")
        else:
            keyboard = build_media_keyboard(mid, user, mode=det_mode)
        await query.answer()
        bot = query.get_bot()
        chat_id = query.message.chat_id
        if poster_path:
            await bot.send_photo(
                chat_id, poster_path,
                caption=esc(desc),
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=keyboard
            )
        else:
            await bot.send_message(
                chat_id, esc(desc),
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=keyboard
            )
        return

    if action == "sdet":
        parts = raw.split(":", 2)
        if len(parts) < 3:
            await query.answer("Invalid action.", show_alert=True)
            return
        mid = int(parts[2])
        try:
            details = tv.details(mid)
            num_seasons = details.get("number_of_seasons") or 1
        except Exception:
            num_seasons = 1
        _pending_season[user] = {"mid": mid,
                                 "total": num_seasons, "media_type": "tv"}
        season_kb = build_season_picker_keyboard(
            mid, "tv", num_seasons, action_prefix="supd")
        await query.answer("Update watched season:")
        bot = query.get_bot()
        chat_id = query.message.chat_id
        stored = user_data[user].get("tv_season_counts", {}).get(mid)
        current_s = stored.get("watched", "?") if stored else "?"
        await bot.send_message(
            chat_id,
            esc(
                f"Update watched season for this show (currently S{current_s}/{num_seasons}):"),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=season_kb)
        return

    if action == "supd":
        parts = raw.split(":")
        try:
            mid = int(parts[2])
            season_num = int(parts[3])
        except (ValueError, IndexError):
            await query.answer("Invalid action.", show_alert=True)
            return
        stored = user_data[user].get("tv_season_counts", {})
        if mid in stored:
            stored[mid]["watched"] = season_num
        else:
            total = season_num
            if user in _pending_season and _pending_season[user]["mid"] == mid:
                total = _pending_season[user]["total"]
            stored[mid] = {"total": total, "watched": season_num}
        _pending_season.pop(user, None)
        save_user_data()
        await query.answer(f"Updated to season {season_num}.")
        try:
            await query.message.delete()
        except Exception:
            await query.edit_message_reply_markup(reply_markup=None)
        return

    if action == "rpick":
        wl_name = raw.split(":", 1)[1]
        watchlist = None if wl_name == "*" else wl_name
        candidates, label = _collect_pick_candidates(user, watchlist)
        if not candidates:
            await query.answer("Watchlists are empty.", show_alert=True)
            return
        await query.answer()
        await _do_pick(query.get_bot(), query.message.chat_id, user,
                       candidates, label, wl_name)
        return

    if action == "sp":
        provider_index = int(raw.split(":", 1)[1])
        all_providers = get_all_movie_provider(user_data[user]["region"])
        if provider_index < 0 or provider_index >= len(all_providers):
            await query.answer("Invalid provider.", show_alert=True)
            return
        name = all_providers[provider_index]
        if name in user_data[user]["providers"]:
            user_data[user]["providers"].remove(name)
            await query.answer(f"Removed {name}.")
        else:
            user_data[user]["providers"].append(name)
            await query.answer(f"Added {name}.")
        if not user_data[user].get("onboarded", False):
            user_data[user]["onboarded"] = True
        save_user_data()
        keyboard = build_services_keyboard(user)
        await query.edit_message_reply_markup(reply_markup=keyboard)
        return

    action, media_type, movie_id, watchlist = parse_callback_data(raw)
    if movie_id is None:
        await query.answer("Invalid action.", show_alert=True)
        return

    cb_mode = _type_to_mode(media_type)

    if action == "pick":
        picker_keyboard = build_watchlist_picker_keyboard(
            movie_id, user, mode=cb_mode)
        await query.edit_message_reply_markup(reply_markup=picker_keyboard)
        await query.answer()

    elif action == "back":
        keyboard = build_media_keyboard(movie_id, user, mode=cb_mode)
        await query.edit_message_reply_markup(reply_markup=keyboard)
        await query.answer()

    elif action == "new":
        _pending_new_watchlist[user] = (movie_id, cb_mode)
        await query.answer()
        await query.get_bot().send_message(
            query.message.chat_id,
            "Enter a name for the new watchlist:",
            reply_markup=ForceReply(selective=True))

    elif action == "a":
        already_in = is_in_any_watchlist(movie_id, user, mode=cb_mode)
        if already_in:
            await query.answer(f'Already in "{already_in}" watchlist.', show_alert=True)
        else:
            if movie_id in user_data[user]["watched"][cb_mode]:
                await query.answer("Warning: you already watched this!", show_alert=True)
            else:
                await query.answer(f'Added to "{watchlist}".')
            user_data[user]["watchlists"][cb_mode][watchlist].append(movie_id)
            save_user_data()
            if _is_search_message(user, query.message.message_id):
                await _cleanup_search_results(query.get_bot(), user)
                await query.get_bot().send_message(
                    query.message.chat_id,
                    f'Added to "{watchlist}".',
                    reply_markup=get_main_keyboard(user))
            else:
                new_keyboard = build_media_keyboard(
                    movie_id, user, mode=cb_mode)
                await query.edit_message_reply_markup(reply_markup=new_keyboard)

    elif action == "rm":
        removed = False
        for _, w in user_data[user]["watchlists"][cb_mode].items():
            if movie_id in w:
                w.remove(movie_id)
                removed = True
        if removed:
            save_user_data()
            await query.answer("Removed from watchlist.")
            if _is_search_message(user, query.message.message_id):
                await _cleanup_search_results(query.get_bot(), user)
                await query.get_bot().send_message(
                    query.message.chat_id,
                    "Removed from watchlist.",
                    reply_markup=get_main_keyboard(user))
            else:
                new_keyboard = build_media_keyboard(
                    movie_id, user, mode=cb_mode)
                await query.edit_message_reply_markup(reply_markup=new_keyboard)
        else:
            await query.answer("Not in any watchlist.", show_alert=True)

    elif action == "w":
        if media_type == "tv":
            try:
                details = tv.details(movie_id)
                num_seasons = details.get("number_of_seasons") or 1
            except Exception:
                num_seasons = 1
            _pending_season[user] = {
                "mid": movie_id, "total": num_seasons, "media_type": media_type}
            season_kb = build_season_picker_keyboard(
                movie_id, media_type, num_seasons)
            await query.edit_message_reply_markup(reply_markup=season_kb)
            await query.answer("Which season did you watch up to?")
        else:
            if movie_id in user_data[user]["watched"][cb_mode] and user_data[user]["watched"][cb_mode][movie_id] is not None:
                await query.answer("Already marked as watched.", show_alert=True)
            else:
                rating_kb = build_rating_keyboard(
                    movie_id, media_type=media_type)
                await query.edit_message_reply_markup(reply_markup=rating_kb)
                await query.answer("Rate this:")

    elif action == "ws":
        parts = raw.split(":")
        try:
            season_num = int(parts[3])
        except (ValueError, IndexError):
            await query.answer("Invalid action.", show_alert=True)
            return
        if user in _pending_season:
            _pending_season[user]["season"] = season_num
        else:
            _pending_season[user] = {
                "mid": movie_id, "total": season_num, "season": season_num, "media_type": media_type}
        rating_kb = build_rating_keyboard(movie_id, media_type=media_type)
        await query.edit_message_reply_markup(reply_markup=rating_kb)
        await query.answer("Rate this:")

    else:
        await query.answer("Unknown action.", show_alert=True)


async def reply_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    text = update.message.text.strip()

    # Handle pending search
    if user in _pending_search:
        _pending_search.pop(user, None)
        if text:
            await do_search(update, text, user)
        return

    # Handle pending new watchlist name
    if user in _pending_new_watchlist:
        movie_id, nwl_mode = _pending_new_watchlist.pop(user, (None, "movie"))
        if not text:
            await send_back_text(update, "Watchlist name cannot be empty.")
            return
        if text in user_data[user]["watchlists"][nwl_mode]:
            await send_back_text(update, f'Watchlist "{text}" already exists.')
            return
        user_data[user]["watchlists"][nwl_mode][text] = []
        if movie_id is None:
            save_user_data()
            await send_back_text(update, f'Created watchlist "{text}".')
        else:
            already_in = is_in_any_watchlist(movie_id, user, mode=nwl_mode)
            if already_in:
                label = "show" if nwl_mode == "tv" else "movie"
                await send_back_text(update, f'Already in your "{already_in}" watchlist.')
            else:
                user_data[user]["watchlists"][nwl_mode][text].append(movie_id)
                save_user_data()
                await send_back_text(update, f'Created watchlist "{text}" and added it.')
        return


async def default_search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    query = update.message.text.strip()
    if query:
        await do_search(update, query, user)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)
    try:
        if isinstance(update, Update) and update.effective_chat:
            user = get_user_id(update) if update.effective_user else None
            kb = get_main_keyboard(user) if user else None
            await context.bot.send_message(
                update.effective_chat.id,
                "An unexpected error occurred. Please try again.",
                reply_markup=kb)
    except Exception:
        logger.error("Failed to send error message to user:", exc_info=True)


async def post_init(application):
    await application.bot.set_my_commands(commands=[
        BotCommand("start", "OKAAAAY LETS GO!!!"),
        BotCommand("search", "Search by keywords"),
        BotCommand("list", "Browse your watchlists"),
        BotCommand("add", "Add to your watchlist"),
        BotCommand("tadd", "Add to your trash watchlist"),
        BotCommand("watched", "Mark as watched"),
        BotCommand("remove", "Remove from all watchlists"),
        BotCommand("rate", "Rate or re-rate watched items"),
        BotCommand("services", "Manage my streaming services"),
        BotCommand("check", "Check streaming availability for your watchlist"),
        BotCommand("recommend", "Get recommendations based on your watchlist"),
        BotCommand("popular", "Show popular titles on your streaming services"),
        BotCommand("pick", "Pick a random title from your watchlists"),
        BotCommand("mode", "Switch between Movies and TV mode"),
        BotCommand("newseasons", "Check for new seasons of watched TV shows"),
        BotCommand("seasons", "View/edit watched seasons for TV shows"),
    ])
    if application.job_queue:
        application.job_queue.run_daily(
            _daily_season_check,
            time=datetime.time(hour=9, minute=0),
        )
    else:
        logger.warning("JobQueue not available. Daily season check disabled. "
                       "Install python-telegram-bot[job-queue] to enable.")


def main():
    application = Application.builder().token(
        settings["telegram_token"]).post_init(post_init).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler(['search', 's'], search_handler))
    application.add_handler(CommandHandler(['list', 'l'], list_watchlists))
    application.add_handler(CommandHandler(['add', 'a'], add_to_watchlist))
    application.add_handler(CommandHandler(
        ['tadd', 't'], add_to_trash_watchlist))
    application.add_handler(CommandHandler(['watched', 'w'], add_to_watched))
    application.add_handler(CommandHandler(
        ['remove', 'rm'], remove_from_watchlist))
    application.add_handler(CommandHandler('rate', rate_movies))
    application.add_handler(CommandHandler('services', show_my_providers))
    application.add_handler(CommandHandler(['check', 'c'], check_watchlist))
    application.add_handler(CommandHandler(['recommend', 'r'], recommend))
    application.add_handler(CommandHandler(['popular', 'pop'], popular_movies))
    application.add_handler(CommandHandler(['pick', 'p'], pick_movie))
    application.add_handler(CommandHandler(['mode', 'm'], toggle_mode))
    application.add_handler(CommandHandler(['newseasons', 'ns'], new_seasons))
    application.add_handler(CommandHandler(['seasons', 'ss'], view_seasons))
    application.add_handler(CommandHandler('clear', clear_chat))
    application.add_handler(CommandHandler('fix', fix_keyboard))
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.REPLY,
        reply_handler
    ))
    application.add_handler(MessageHandler(
        filters.Regex(
            f"^({re.escape(_MODE_SWITCH_TV)}|{re.escape(_MODE_SWITCH_MOVIE)})$"),
        toggle_mode
    ))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & ~filters.REPLY,
        default_search_handler
    ))

    application.add_error_handler(error_handler)

    application.run_polling()


if __name__ == '__main__':
    main()
