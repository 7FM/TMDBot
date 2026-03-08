from telegram import (
    InlineKeyboardButton, InlineKeyboardMarkup,
    ReplyKeyboardMarkup, KeyboardButton,
)

from botlib import state
from botlib.config import settings
from tmdbot.helpers import get_all_movie_provider

# Re-export all generic keyboards from botlib
from botlib.keyboards import (  # noqa: F401
    build_media_keyboard, build_watchlist_picker_keyboard,
    build_chunk_keyboard, build_watchlist_select_keyboard,
    build_member_select_keyboard, build_rating_keyboard,
    build_category_picker_keyboard, build_recommend_category_keyboard,
)

# Re-export helpers used by other modules via keyboards
from botlib.helpers import _mode_to_type  # noqa: F401


# --- TMDb-specific keyboards below ---

_MODE_SWITCH_TV = "\U0001f4fa Switch to TV"
_MODE_SWITCH_MOVIE = "\U0001f3ac Switch to Movies"


def get_main_keyboard(user):
    mode = state.user_data.get(user, {}).get("mode", "movie")
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


def build_region_keyboard(page: int = 0) -> InlineKeyboardMarkup:
    from tmdbot.config import REGIONS, REGIONS_PER_PAGE, _flag_emoji
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
    from tmdbot.config import _flag_emoji, _region_name
    all_providers = get_all_movie_provider(state.user_data[user]["region"])
    my_providers = state.user_data[user]["providers"]
    rows = []
    for i, name in enumerate(all_providers):
        prefix = "\u2705 " if name in my_providers else "\u274c "
        rows.append([InlineKeyboardButton(
            prefix + name, callback_data=f"sp:{i}")])
    region_label = f"{_flag_emoji(state.user_data[user]['region'])} Region: {_region_name(state.user_data[user]['region'])}"
    rows.append([InlineKeyboardButton(region_label, callback_data="chreg")])
    return InlineKeyboardMarkup(rows)


def build_genre_picker_keyboard(selected_genres: set, mode="movie") -> InlineKeyboardMarkup:
    from tmdbot.config import get_genre_dict
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
