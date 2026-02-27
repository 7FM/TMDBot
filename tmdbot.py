import sys
import os
import re
import asyncio
import yaml
import logging
import random
import concurrent.futures
import multiprocessing
from collections import Counter
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton, LinkPreviewOptions, ForceReply
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, CallbackContext, ContextTypes
from telegram.constants import ParseMode
from tmdbv3api import TMDb, Movie, Search, Genre, Provider

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
search = Search()
genre = Genre()
provider = Provider()

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
            user_data[user]["watched"] = {}
            user_data[user]["watchlists"] = dict()
            user_data[user]["watchlists"]["normal"] = []
            user_data[user]["watchlists"]["trash"] = []
            user_data[user]["providers"] = []
            user_data[user]["onboarded"] = False
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


genre_dict = get_movie_genres()

# Helper functions


def get_image_url(path):
    if path:
        return f"https://image.tmdb.org/t/p/original{path}"
    return None


_provider_cache = {}
_pending_new_watchlist = {}
_pending_search = {}
_chunk_movies = {}
_chunk_id_counter = 0
_search_results = {}  # user_id -> (chat_id, [message_ids])
_search_more = {}  # user_id -> (remaining_sorted_results, query)
_rate_list_messages = {}  # user_id -> (chat_id, [message_ids])
_rec_genre_filter = {}  # user_id -> {"watchlist": str, "genres": set}
_last_watched = {}  # user_id -> {"mid": int, "watchlist": str|None, "prev_rating": int|None|"absent"}

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        [KeyboardButton("/search"), KeyboardButton("/list"), KeyboardButton("/check")],
        [KeyboardButton("/recommend"), KeyboardButton("/popular")],
        [KeyboardButton("/pick"), KeyboardButton("/clear")],
    ],
    resize_keyboard=True,
    is_persistent=True,
)


def get_all_movie_provider(region):
    if region in _provider_cache:
        return _provider_cache[region]
    movie_provider = provider.movie_providers(region=region)
    result = []
    if movie_provider:
        movie_provider = movie_provider["results"]
        result = [mp["provider_name"] for mp in movie_provider]
    _provider_cache[region] = result
    return result


# TODO JustWatch Attribution Required
def get_free_provider(id, country_code):
    details = movie.details(id, append_to_response="watch/providers")
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


def is_available_for_free(my_providers, id, country_code):
    provider = get_free_provider(id, country_code)
    return _match_providers(my_providers, provider)


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


def extract_trailer_url(m):
    if "trailers" not in m:
        m = movie.details(m["id"], append_to_response="trailers")
    if "trailers" in m and "youtube" in m["trailers"]:
        for t in m["trailers"]["youtube"]:
            if t["type"] == "Trailer":
                return f'https://www.youtube.com/watch?v={t["source"]}'

    return None


def extract_genre(m):
    global genre_dict
    genre = []
    if "genre_ids" in m:
        for g in m["genre_ids"]:
            if g not in genre_dict:
                genre_dict = get_movie_genres()
            genre.append(genre_dict[g])
    elif "genres" in m:
        for g in m["genres"]:
            genre.append(g["name"])
    return genre


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


def extract_movie_info(m, skip_trailer=False):
    title = m["title"]
    poster_path = get_image_url(m['poster_path'])
    # popularity = m["popularity"] if "popularity" in m else -1
    rating = m["vote_average"] if "vote_average" in m and m["vote_count"] > 0 else None
    release_date = m["release_date"] if "release_date" in m and m["release_date"] != "" else None
    id = m["id"]
    genre = extract_genre(m)
    trailer = None if skip_trailer else extract_trailer_url(m)
    if trailer:
        title = f'[{title}]({trailer})'
    else:
        title = f'`{title}`'
    desc = title + (" - " + release_date if release_date else "") + (" - " + (", ".join(genre)) if genre else "") + ' - ' + (str(round(rating, 1)) if rating else "?") + '/10'
    return (rating if rating else 0, poster_path, desc, id)


def sort_by_rating(movie_list):
    return sorted(movie_list, key=lambda x: x[0], reverse=True)


def is_in_any_watchlist(movie_id, user):
    for wn, w in user_data[user]["watchlists"].items():
        if movie_id in w:
            return wn
    return None

def is_valid_movie_id(movie_id):
    if not movie_id.isdigit():
        return "ID is not a number"
    try:
        movie.details(movie_id)
    except:
        return "Movie not found"
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
        last_newline = text.rfind('\n', current_position, current_position + max_chunk_size)

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


async def send_back_text(update: Update, msg):
    chunks = split_into_chunks(msg)
    sent = []
    for c in chunks:
        sent.append(await update.message.reply_text(
            esc(c), parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=MAIN_KEYBOARD))
    return sent


def parse_callback_data(data: str):
    parts = data.split(":", 2)
    action = parts[0]
    movie_id = int(parts[1])
    watchlist = parts[2] if len(parts) > 2 else None
    return action, movie_id, watchlist


def build_movie_keyboard(movie_id: int, user: int) -> InlineKeyboardMarkup:
    buttons = []
    watchlist_name = is_in_any_watchlist(movie_id, user)
    if watchlist_name:
        buttons.append(InlineKeyboardButton("Remove", callback_data=f"rm:{movie_id}"))
    else:
        buttons.append(InlineKeyboardButton("Add", callback_data=f"pick:{movie_id}"))
    buttons.append(InlineKeyboardButton("Watched", callback_data=f"w:{movie_id}"))
    return InlineKeyboardMarkup([buttons])


def build_watchlist_picker_keyboard(movie_id: int, user: int) -> InlineKeyboardMarkup:
    rows = []
    for wn in user_data[user]["watchlists"]:
        cb_data = f"a:{movie_id}:{wn}"
        if len(cb_data.encode('utf-8')) > 64:
            continue
        rows.append([InlineKeyboardButton(wn, callback_data=cb_data)])
    rows.append([
        InlineKeyboardButton("New", callback_data=f"new:{movie_id}"),
        InlineKeyboardButton("Back", callback_data=f"back:{movie_id}")
    ])
    return InlineKeyboardMarkup(rows)


def build_chunk_keyboard(chunk_id: int, movies_list, expanded: bool, detail_action: str = "det") -> InlineKeyboardMarkup:
    if expanded:
        rows = [[InlineKeyboardButton(
            m_title, callback_data=f"{detail_action}:{m_id}")]
            for m_id, m_title in movies_list]
        rows.append([InlineKeyboardButton(
            "Collapse", callback_data=f"col:{chunk_id}")])
    else:
        rows = [[InlineKeyboardButton(
            "Show movies", callback_data=f"exp:{chunk_id}")]]
    return InlineKeyboardMarkup(rows)


def build_watchlist_select_keyboard(user: int, edit_mode: bool = False) -> InlineKeyboardMarkup:
    rows = []
    for wn in user_data[user]["watchlists"]:
        if len(f"wl:{wn}".encode('utf-8')) > 64:
            continue
        count = len(user_data[user]["watchlists"][wn])
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


def build_rating_keyboard(movie_id: int, action_prefix: str = "rate") -> InlineKeyboardMarkup:
    row1 = [InlineKeyboardButton(str(i), callback_data=f"{action_prefix}:{movie_id}:{i}")
            for i in range(1, 6)]
    row2 = [InlineKeyboardButton(str(i), callback_data=f"{action_prefix}:{movie_id}:{i}")
            for i in range(6, 11)]
    row3 = [InlineKeyboardButton("Skip", callback_data=f"{action_prefix}:{movie_id}:0")]
    return InlineKeyboardMarkup([row1, row2, row3])


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
        nav.append(InlineKeyboardButton("\u25c0 Back", callback_data=f"regp:{page - 1}"))
    if end < len(REGIONS):
        nav.append(InlineKeyboardButton("Next \u25b6", callback_data=f"regp:{page + 1}"))
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


def build_genre_picker_keyboard(selected_genres: set) -> InlineKeyboardMarkup:
    rows = []
    genre_items = sorted(genre_dict.items(), key=lambda x: x[1])
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


async def send_movie_message(update: Update, caption: str, poster_path, movie_id: int, user: int):
    keyboard = build_movie_keyboard(movie_id, user)
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


async def send_movie_list(bot, chat_id, header: str, movies_info, detail_action: str = "det"):
    """Send a chunked itemized list of movies with expand/collapse buttons.

    movies_info: list of (movie_id, title, description) tuples
    detail_action: callback action prefix for movie buttons (default "det")
    Returns list of sent Message objects.
    """
    global _chunk_id_counter
    chunk_text = header + "\n"
    chunk_movies_list = []
    max_size = 4096
    sent = []
    for mid, title, desc in movies_info:
        line = f"\u2022 {desc}\n"
        if len(chunk_text) + len(line) > max_size and chunk_movies_list:
            cid = _chunk_id_counter
            _chunk_id_counter += 1
            _chunk_movies[cid] = (chunk_movies_list, detail_action)
            kb = build_chunk_keyboard(cid, chunk_movies_list, expanded=False)
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
                m_title, callback_data=f"{detail_action}:{mid}")]])
        else:
            cid = _chunk_id_counter
            _chunk_id_counter += 1
            _chunk_movies[cid] = (chunk_movies_list, detail_action)
            kb = build_chunk_keyboard(cid, chunk_movies_list, expanded=False)
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



async def add_to_watchlist_helper(watchlist, movie_id, user, update: Update):
    err_msg = is_valid_movie_id(movie_id)
    if err_msg:
        await send_back_text(update, f'The provided movie id is invalid: ' + err_msg)
        return
    movie_id = int(movie_id)
    # Check if the movie is already in the watchlist
    already_in = is_in_any_watchlist(movie_id, user)
    if already_in:
        await send_back_text(update, f'This movie is already in your "{already_in}" watchlist.')
    else:
        if movie_id in user_data[user]["watched"]:
            await send_back_text(update, "Warning: you have already seen this movie!")
        user_data[user]["watchlists"][watchlist].append(movie_id)
        save_user_data()
        await send_back_text(update, 'Movie added to watchlist.')


async def add_to_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not context.args:
        await send_back_text(update, 'Please provide the movie ID.')
        return

    watchlist = "normal"
    if len(context.args) == 2:
        watchlist = context.args[1]
    if watchlist not in user_data[user]["watchlists"]:
        await send_back_text(update, f'Info: creating new watchlist "{watchlist}"')
        user_data[user]["watchlists"][watchlist] = []

    movie_id = context.args[0]
    await add_to_watchlist_helper(watchlist, movie_id, user, update)


async def add_to_trash_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not context.args:
        await send_back_text(update, 'Please provide the movie ID.')
        return

    movie_id = context.args[0]
    await add_to_watchlist_helper("trash", movie_id, user, update)


async def add_to_watched(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not context.args:
        await send_back_text(update, 'Please provide the movie ID.')
        return

    movie_id = context.args[0]
    err_msg = is_valid_movie_id(movie_id)
    if err_msg:
        await send_back_text(update, f'The provided movie id is invalid: ' + err_msg)
        return
    movie_id = int(movie_id)
    if movie_id in user_data[user]["watched"]:
        await send_back_text(update, 'Movie was already marked as watched.')
        return
    keyboard = build_rating_keyboard(movie_id)
    await update.message.reply_text(
        "Rate this movie (1\u201310):",
        reply_markup=keyboard)


async def remove_from_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not context.args:
        await send_back_text(update, 'Please provide the movie ID.')
        return

    movie_id = context.args[0]
    err_msg = is_valid_movie_id(movie_id)
    if err_msg:
        await send_back_text(update, f'The provided movie id is invalid: ' + err_msg)
        return
    movie_id = int(movie_id)
    removed_smth = False
    for _, w in user_data[user]["watchlists"].items():
        if movie_id in w:
            w.remove(movie_id)
            removed_smth = True
    if removed_smth:
        save_user_data()
        await send_back_text(update, 'Movie removed from watchlist.')
    else:
        await send_back_text(update, 'This movie is not in your watchlist.')


async def _do_recommend(bot, chat_id, user, watchlist, genre_filter=None):
    """Core recommendation logic. genre_filter is a set of genre IDs or None for all."""
    # Sources: watchlist movies (weight 1.0) + top 20 highest-rated watched movies
    sources = [(mid, 1.0) for mid in user_data[user]["watchlists"][watchlist]]
    rated_watched = sorted(
        [(mid, r) for mid, r in user_data[user]["watched"].items()
         if r is not None and r >= 7],
        key=lambda x: -x[1])[:20]
    sources.extend((mid, rating / 10.0) for mid, rating in rated_watched)

    if not sources:
        await bot.send_message(
            chat_id,
            esc(f'Your "{watchlist}" watchlist is empty and you have no highly-rated watched movies.'),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=MAIN_KEYBOARD)
        return

    total = len(sources)

    def query_recommendations(source):
        movie_id, weight = source
        available_recommendations = []
        results = movie.recommendations(movie_id)
        movies = results["results"]
        for m in movies:
            # Genre filter: skip if movie doesn't match any selected genre
            if genre_filter:
                movie_genres = set(m.get("genre_ids", []))
                if not movie_genres.intersection(genre_filter):
                    continue
            in_watchlist = is_in_any_watchlist(m["id"], user)
            if not in_watchlist and m["id"] not in user_data[user]["watched"]:
                available, provider = is_available_for_free(
                    user_data[user]["providers"], m["id"], user_data[user]["region"])
                if available:
                    popularity, poster, desc, mid = extract_movie_info(
                        m, skip_trailer=True)
                    available_recommendations.append(
                        (popularity, poster, desc, provider, mid, m["title"], weight))
        return available_recommendations

    def do_recommend(tick):
        num_threads = multiprocessing.cpu_count()
        all_recs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for result in executor.map(query_recommendations, sources):
                all_recs.extend(result)
                tick()
        return all_recs

    available_recommendations = await _with_progress_bar(
        bot, chat_id, "Finding recommendations...", total, do_recommend)

    # Sum weights per recommended movie (weighted by source ratings)
    id_scores = {}
    for item in available_recommendations:
        mid = item[4]
        id_scores[mid] = id_scores.get(mid, 0) + item[6]

    def custom_sort_key(item):
        return (-id_scores[item[4]], -item[0])

    # Make entries unique by movie ID
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
        for _, poster_path, caption, provider, mid, title, _ in available_recommendations[:num_rec]:
            provider_str = create_available_at_str(provider)
            movies_info.append((mid, title, caption + "\n" + provider_str))
        await send_movie_list(
            bot, chat_id,
            f'Recommended movies based on your "{watchlist}" watchlist:',
            movies_info)
    else:
        await bot.send_message(
            chat_id,
            esc(f'No recommendations found based on your "{watchlist}" watchlist.'),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=MAIN_KEYBOARD)


async def recommend(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    watchlist = "normal"
    if context.args:
        watchlist = context.args[0]

    if watchlist not in user_data[user]["watchlists"]:
        await send_back_text(update, f'Watchlist "{watchlist}" not found.')
        return

    _rec_genre_filter[user] = {"watchlist": watchlist, "genres": set()}
    keyboard = build_genre_picker_keyboard(set())
    await update.message.reply_text(
        "Filter recommendations by genre (or skip for all):",
        reply_markup=keyboard)


async def check_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    def fetch_movie_details(movie_id):
        details = movie.details(
            movie_id, append_to_response="watch/providers")
        providers = _parse_providers_from_details(
            details, user_data[user]["region"])
        avail, matched = _match_providers(my_providers, providers)
        if avail:
            return (movie_id, matched, details)
        return None

    my_providers = user_data[user]["providers"]
    bot = update.get_bot()
    chat_id = update.message.chat_id
    total = sum(len(w) for w in user_data[user]["watchlists"].values())

    watchlists_snapshot = list(user_data[user]["watchlists"].items())

    def do_check(tick):
        num_threads = multiprocessing.cpu_count()
        available_movies = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for wn, w in watchlists_snapshot:
                results = []
                for r in executor.map(fetch_movie_details, w):
                    if r:
                        results.append(r)
                    tick()
                available_movies.append((wn, results))
        return available_movies

    available_movies = await _with_progress_bar(
        bot, chat_id, "Checking streaming availability...", total, do_check)

    for wn, movies in available_movies:
        if movies:
            movies_info = []
            for movie_id, provider, details in movies:
                _, _, desc, mid = extract_movie_info(
                    details, skip_trailer=True)
                provider_str = create_available_at_str(provider)
                movies_info.append((mid, details["title"], desc + "\n" + provider_str))
            await send_movie_list(
                update.get_bot(), update.message.chat_id,
                f'Movies on your {wn} watchlist available on streaming services:',
                movies_info)
        else:
            await send_back_text(update, f'None of the movies on your {wn} watchlist are available on streaming services.')


async def popular_movies(update: Update, context: CallbackContext) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    # Get currently popular movies
    status_msg = await update.message.reply_text(
        "Finding popular movies...", reply_markup=MAIN_KEYBOARD)
    target_count = 10
    page = 1
    results = movie.popular(page=page)
    total_pages = results["total_pages"]
    # Phase 1: collect candidates (skip watched)
    candidates = []
    max_candidates = target_count * 3
    while page < total_pages and len(candidates) < max_candidates:
        if page != 1:
            results = movie.popular(page=page)
        for m in results["results"]:
            if m["id"] not in user_data[user]["watched"]:
                candidates.append(m)
        page += 1

    # Phase 2: parallel availability checks
    my_providers = user_data[user]["providers"]
    user_region = user_data[user]["region"]

    def check_popular_movie(m):
        available, provider = is_available_for_free(
            my_providers, m["id"], user_region)
        if available:
            _, poster, desc, mid = extract_movie_info(m, skip_trailer=True)
            return (mid, m["title"], desc, provider)
        return None

    num_threads = multiprocessing.cpu_count()
    pop_movies = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(check_popular_movie, m) for m in candidates]
        for future in futures:
            if future.cancelled():
                continue
            result = future.result()
            if result:
                pop_movies.append(result)
                if len(pop_movies) >= target_count:
                    for f in futures:
                        f.cancel()
                    break

    try:
        await status_msg.delete()
    except Exception:
        pass

    if pop_movies:
        movies_info = []
        for mid, title, caption, provider in pop_movies[:target_count]:
            provider_str = create_available_at_str(provider)
            movies_info.append((mid, title, caption + "\n" + provider_str))
        await send_movie_list(
            update.get_bot(), update.message.chat_id,
            'Popular movies available on your streaming services:',
            movies_info)
    else:
        await send_back_text(update, 'No popular movies found on your streaming services.')


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


def _collect_pick_candidates(user, watchlist=None):
    """Collect movie IDs from one or all watchlists. Returns (candidates, label)."""
    if watchlist:
        wl = user_data[user]["watchlists"].get(watchlist, [])
        return list(wl), f'"{watchlist}"'
    all_movies = []
    for wl in user_data[user]["watchlists"].values():
        all_movies.extend(wl)
    return list(set(all_movies)), "your watchlists"


async def pick_movie(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    my_providers = user_data[user]["providers"]
    if not my_providers:
        await send_back_text(update, "Set up your streaming services first with /services.")
        return

    watchlist = context.args[0] if context.args else None
    if watchlist and watchlist not in user_data[user]["watchlists"]:
        await send_back_text(update, f'Watchlist "{watchlist}" not found.')
        return

    candidates, label = _collect_pick_candidates(user, watchlist)
    if not candidates:
        await send_back_text(update, f'{label} is empty.' if watchlist else 'All your watchlists are empty.')
        return

    user_region = user_data[user]["region"]
    random.shuffle(candidates)

    picked_mid = None
    matched_providers = None
    for mid in candidates:
        details = movie.details(mid, append_to_response="watch/providers")
        providers = _parse_providers_from_details(details, user_region)
        avail, matched = _match_providers(my_providers, providers)
        if avail:
            picked_mid = mid
            matched_providers = matched
            break

    if picked_mid is None:
        await send_back_text(update, f'No movies in {label} are available on your services.')
        return

    movie_details = movie.details(picked_mid)
    _, poster_path, desc, _ = extract_movie_info(movie_details)
    desc += "\n" + create_available_at_str(matched_providers)
    keyboard = build_movie_keyboard(picked_mid, user)
    pick_cb = f"rpick:{watchlist or '*'}"
    if len(pick_cb.encode('utf-8')) <= 64:
        rows = list(keyboard.inline_keyboard) + [
            [InlineKeyboardButton("Pick another", callback_data=pick_cb)]]
        keyboard = InlineKeyboardMarkup(rows)
    escaped = esc(desc)
    if poster_path:
        await update.message.reply_photo(
            poster_path, escaped,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard)
    else:
        await update.message.reply_text(
            escaped,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard)


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
        reply_markup=MAIN_KEYBOARD)


async def fix_keyboard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return
    await update.message.reply_text("Keyboard restored.", reply_markup=MAIN_KEYBOARD)


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
            reply_markup=MAIN_KEYBOARD
        )


async def _send_rate_list(bot, chat_id, user):
    """Build and send the rate list for a user. Returns sent messages."""
    watched = user_data[user]["watched"]
    unrated = [(mid, r) for mid, r in watched.items() if r is None]
    rated = sorted(
        [(mid, r) for mid, r in watched.items() if r is not None],
        key=lambda x: -x[1])

    movies_info = []
    for mid, r in unrated + rated:
        movie_details = movie.details(mid)
        _, _, desc, _ = extract_movie_info(movie_details, skip_trailer=True)
        rating_str = "unrated" if r is None else f"Your rating: {r}/10"
        movies_info.append((mid, movie_details["title"], f"{desc}\n{rating_str}"))

    n_unrated = len(unrated)
    n_rated = len(rated)
    if n_unrated > 0:
        header = f'{n_unrated} unrated, {n_rated} rated movie(s):'
    else:
        header = f'{n_rated} watched movie(s):'
    return await send_movie_list(
        bot, chat_id, header, movies_info, detail_action="rdet")


async def rate_movies(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not user_data[user]["watched"]:
        await send_back_text(update, "You haven't watched any movies yet!")
        return

    # Clean up previous rate list
    await _cleanup_rate_list(update.get_bot(), user)

    msgs = await _send_rate_list(
        update.get_bot(), update.message.chat_id, user)
    _rate_list_messages[user] = (update.message.chat_id, [m.message_id for m in msgs])


async def do_search(update: Update, query: str, user: int) -> None:
    results = search.movies(query)
    if results and results["total_results"] > 0:
        await _cleanup_search_results(update.get_bot(), user)
        batch = []
        msgs = await send_back_text(update, f'Search results for "{query}":')
        batch.extend(m.message_id for m in msgs)
        movies = results["results"]
        res = []
        for m in movies:
            res.append(extract_movie_info(m))
        sorted_res = sort_by_rating(res)
        show_results = min(5, len(sorted_res))
        for _, poster_path, caption, mid in sorted_res[:show_results]:
            msg = await send_movie_message(update, caption, poster_path, mid, user)
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
        _pending_new_watchlist[user] = None
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
        confirm_kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("Yes, delete", callback_data=f"dwly:{wl_name}"),
            InlineKeyboardButton("Cancel", callback_data="dwln")
        ]])
        await query.edit_message_reply_markup(reply_markup=confirm_kb)
        return

    if action == "dwly":
        wl_name = raw.split(":", 1)[1]
        if wl_name in user_data[user]["watchlists"]:
            del user_data[user]["watchlists"][wl_name]
            save_user_data()
            await query.answer(f'Deleted "{wl_name}".')
        else:
            await query.answer("Watchlist not found.", show_alert=True)
        keyboard = build_watchlist_select_keyboard(user, edit_mode=True)
        await query.edit_message_reply_markup(reply_markup=keyboard)
        return

    if action == "dwln":
        keyboard = build_watchlist_select_keyboard(user, edit_mode=True)
        await query.edit_message_reply_markup(reply_markup=keyboard)
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
        keyboard = build_genre_picker_keyboard(genres)
        await query.edit_message_reply_markup(reply_markup=keyboard)
        await query.answer()
        return

    if action == "recgo":
        if user not in _rec_genre_filter:
            await query.answer("Session expired.", show_alert=True)
            return
        state = _rec_genre_filter.pop(user)
        mode = raw.split(":", 1)[1]
        genre_filter = state["genres"] if mode == "filter" and state["genres"] else None
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
        for _, poster_path, caption, mid in next_batch:
            keyboard = build_movie_keyboard(mid, user)
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
        prev_wl = state["watchlist"]
        prev_rating = state["prev_rating"]
        # Restore previous watched state
        if prev_rating == "absent":
            user_data[user]["watched"].pop(mid, None)
        else:
            user_data[user]["watched"][mid] = prev_rating
        # Restore to watchlist if it was in one
        if prev_wl and prev_wl in user_data[user]["watchlists"]:
            user_data[user]["watchlists"][prev_wl].append(mid)
        save_user_data()
        await query.answer("Undone!")
        try:
            await query.message.delete()
        except Exception:
            await query.edit_message_reply_markup(reply_markup=None)
        await query.get_bot().send_message(
            query.message.chat_id,
            "Last watched action undone.",
            reply_markup=MAIN_KEYBOARD)
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
        mid = int(parts[1])
        rating = int(parts[2]) or None
        # Save undo state
        prev_wl = is_in_any_watchlist(mid, user)
        prev_rating = user_data[user]["watched"].get(mid, "absent")
        _last_watched[user] = {"mid": mid, "watchlist": prev_wl, "prev_rating": prev_rating}
        for _, w in user_data[user]["watchlists"].items():
            if mid in w:
                w.remove(mid)
        user_data[user]["watched"][mid] = rating
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
                reply_markup=MAIN_KEYBOARD)
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
                reply_markup=MAIN_KEYBOARD)
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
        wl_movies = user_data[user]["watchlists"].get(wl_name, [])
        if not wl_movies:
            await query.answer(f'"{wl_name}" is empty.', show_alert=True)
            return
        await query.answer()
        movies_info = []
        for mid in wl_movies:
            movie_details = movie.details(mid)
            _, _, desc, _ = extract_movie_info(movie_details)
            movies_info.append((mid, movie_details["title"], desc))
        await send_movie_list(query.get_bot(), query.message.chat_id, f'{wl_name} watchlist:', movies_info)
        return

    if action in ("exp", "col"):
        cid = int(raw.split(":", 1)[1])
        if cid not in _chunk_movies:
            await query.answer("Session expired.", show_alert=True)
            return
        expanded = action == "exp"
        chunk_movies_list, det_action = _chunk_movies[cid]
        kb = build_chunk_keyboard(cid, chunk_movies_list, expanded=expanded, detail_action=det_action)
        await query.edit_message_reply_markup(reply_markup=kb)
        await query.answer()
        return

    if action in ("det", "rdet"):
        mid = int(raw.split(":", 1)[1])
        movie_details = movie.details(mid)
        _, poster_path, desc, _ = extract_movie_info(movie_details)
        if action == "rdet":
            keyboard = build_rating_keyboard(mid, action_prefix="rrate")
        else:
            keyboard = build_movie_keyboard(mid, user)
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

    if action == "rpick":
        wl_name = raw.split(":", 1)[1]
        watchlist = None if wl_name == "*" else wl_name
        candidates, label = _collect_pick_candidates(user, watchlist)
        if not candidates:
            await query.answer("Watchlists are empty.", show_alert=True)
            return
        my_providers = user_data[user]["providers"]
        user_region = user_data[user]["region"]
        random.shuffle(candidates)
        await query.answer()
        bot = query.get_bot()
        chat_id = query.message.chat_id
        picked_mid = None
        matched_providers = None
        for mid in candidates:
            details = movie.details(mid, append_to_response="watch/providers")
            providers = _parse_providers_from_details(details, user_region)
            avail, matched = _match_providers(my_providers, providers)
            if avail:
                picked_mid = mid
                matched_providers = matched
                break
        if picked_mid is None:
            await bot.send_message(
                chat_id, esc(f'No available movies in {label}.'),
                parse_mode=ParseMode.MARKDOWN_V2, reply_markup=MAIN_KEYBOARD)
            return
        movie_details = movie.details(picked_mid)
        _, poster_path, desc, _ = extract_movie_info(movie_details)
        desc += "\n" + create_available_at_str(matched_providers)
        keyboard = build_movie_keyboard(picked_mid, user)
        pick_cb = f"rpick:{wl_name}"
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
        return

    action, movie_id, watchlist = parse_callback_data(raw)

    if action == "pick":
        picker_keyboard = build_watchlist_picker_keyboard(movie_id, user)
        await query.edit_message_reply_markup(reply_markup=picker_keyboard)
        await query.answer()

    elif action == "back":
        keyboard = build_movie_keyboard(movie_id, user)
        await query.edit_message_reply_markup(reply_markup=keyboard)
        await query.answer()

    elif action == "new":
        _pending_new_watchlist[user] = movie_id
        await query.answer()
        await query.get_bot().send_message(
            query.message.chat_id,
            "Enter a name for the new watchlist:",
            reply_markup=ForceReply(selective=True))

    elif action == "a":
        already_in = is_in_any_watchlist(movie_id, user)
        if already_in:
            await query.answer(f'Already in "{already_in}" watchlist.', show_alert=True)
        else:
            if movie_id in user_data[user]["watched"]:
                await query.answer("Warning: you already watched this!", show_alert=True)
            else:
                await query.answer(f'Added to "{watchlist}".')
            user_data[user]["watchlists"][watchlist].append(movie_id)
            save_user_data()
            if _is_search_message(user, query.message.message_id):
                await _cleanup_search_results(query.get_bot(), user)
                await query.get_bot().send_message(
                    query.message.chat_id,
                    f'Added to "{watchlist}".',
                    reply_markup=MAIN_KEYBOARD)
            else:
                new_keyboard = build_movie_keyboard(movie_id, user)
                await query.edit_message_reply_markup(reply_markup=new_keyboard)

    elif action == "rm":
        removed = False
        for _, w in user_data[user]["watchlists"].items():
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
                    reply_markup=MAIN_KEYBOARD)
            else:
                new_keyboard = build_movie_keyboard(movie_id, user)
                await query.edit_message_reply_markup(reply_markup=new_keyboard)
        else:
            await query.answer("Not in any watchlist.", show_alert=True)

    elif action == "w":
        if movie_id in user_data[user]["watched"] and user_data[user]["watched"][movie_id] is not None:
            await query.answer("Already marked as watched.", show_alert=True)
        else:
            rating_kb = build_rating_keyboard(movie_id)
            await query.edit_message_reply_markup(reply_markup=rating_kb)
            await query.answer("Rate this movie:")

    elif action == "sp":
        provider_index = movie_id  # reused field holds the index
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
        _pending_search.pop(user)
        if text:
            await do_search(update, text, user)
        return

    # Handle pending new watchlist name
    if user in _pending_new_watchlist:
        movie_id = _pending_new_watchlist.pop(user)
        if not text:
            await send_back_text(update, "Watchlist name cannot be empty.")
            return
        if text in user_data[user]["watchlists"]:
            await send_back_text(update, f'Watchlist "{text}" already exists.')
            return
        user_data[user]["watchlists"][text] = []
        if movie_id is None:
            save_user_data()
            await send_back_text(update, f'Created watchlist "{text}".')
        else:
            already_in = is_in_any_watchlist(movie_id, user)
            if already_in:
                await send_back_text(update, f'Movie is already in your "{already_in}" watchlist.')
            else:
                user_data[user]["watchlists"][text].append(movie_id)
                save_user_data()
                await send_back_text(update, f'Created watchlist "{text}" and added the movie.')
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
            await context.bot.send_message(
                update.effective_chat.id,
                "An unexpected error occurred. Please try again.",
                reply_markup=MAIN_KEYBOARD)
    except Exception:
        logger.error("Failed to send error message to user:", exc_info=True)


async def post_init(application):
    await application.bot.set_my_commands(commands=[
        BotCommand("start", "OKAAAAY LETS GO!!!"),
        BotCommand("search", "Search a movie based on given keywords"),
        BotCommand("list", "Browse your watchlists"),
        BotCommand("add", "Add movie to your watchlist"),
        BotCommand("tadd", "Add movie to your trash watchlist"),
        BotCommand("watched", "Mark a movie as watched"),
        BotCommand("remove", "Remove movie from all watchlists"),
        BotCommand("rate", "Rate or re-rate watched movies"),
        BotCommand("services", "Manage my streaming services"),
        BotCommand("check", "Check the availability of movies in your watchlist"),
        BotCommand("recommend", "Find recommendations based on your selected watchlist"),
        BotCommand("popular", "Show currently popular movies available at your streaming services"),
        BotCommand("pick", "Pick a random movie available on your services"),
    ])


def main():
    application = Application.builder().token(
        settings["telegram_token"]).post_init(post_init).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler(['search', 's'], search_handler))
    application.add_handler(CommandHandler(['list', 'l'], list_watchlists))
    application.add_handler(CommandHandler(['add', 'a'], add_to_watchlist))
    application.add_handler(CommandHandler(['tadd', 't'], add_to_trash_watchlist))
    application.add_handler(CommandHandler(['watched', 'w'], add_to_watched))
    application.add_handler(CommandHandler(['remove', 'rm'], remove_from_watchlist))
    application.add_handler(CommandHandler('rate', rate_movies))
    application.add_handler(CommandHandler('services', show_my_providers))
    application.add_handler(CommandHandler(['check', 'c'], check_watchlist))
    application.add_handler(CommandHandler(['recommend', 'r'], recommend))
    application.add_handler(CommandHandler(['popular', 'pop'], popular_movies))
    application.add_handler(CommandHandler(['pick', 'p'], pick_movie))
    application.add_handler(CommandHandler('clear', clear_chat))
    application.add_handler(CommandHandler('fix', fix_keyboard))
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.REPLY,
        reply_handler
    ))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & ~filters.REPLY,
        default_search_handler
    ))

    application.add_error_handler(error_handler)

    application.run_polling()


if __name__ == '__main__':
    main()
