from telegram import (
    InlineKeyboardButton, InlineKeyboardMarkup,
    ReplyKeyboardMarkup, KeyboardButton,
)

from tmdbot import state
from tmdbot.config import settings, REGIONS, REGIONS_PER_PAGE, _flag_emoji, _region_name
from tmdbot.helpers import (
    get_all_movie_provider, is_in_any_watchlist,
    _user_shared_watchlists, _get_user_display_name,
    _mode_to_type,
)

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


def build_media_keyboard(media_id: int, user: int, mode=None) -> InlineKeyboardMarkup:
    if mode is None:
        mode = state.user_data[user].get("mode", "movie")
    mt = _mode_to_type(mode)
    buttons = []
    watchlist_name = is_in_any_watchlist(media_id, user, mode=mode)
    if watchlist_name:
        buttons.append(InlineKeyboardButton(
            "Remove", callback_data=f"rm:{mt}:{media_id}"))
    else:
        buttons.append(InlineKeyboardButton(
            "Add", callback_data=f"pick:{mt}:{media_id}"))
    already_watched = media_id in state.user_data[user].get(
        "watched", {}).get(mode, {})
    if not already_watched:
        buttons.append(InlineKeyboardButton(
            "Watched", callback_data=f"w:{mt}:{media_id}"))
    return InlineKeyboardMarkup([buttons])


def build_watchlist_picker_keyboard(media_id: int, user: int, mode=None) -> InlineKeyboardMarkup:
    if mode is None:
        mode = state.user_data[user].get("mode", "movie")
    mt = _mode_to_type(mode)
    rows = []
    for wn in state.user_data[user]["watchlists"][mode]:
        cb_data = f"a:{mt}:{media_id}:{wn}"
        if len(cb_data.encode('utf-8')) > 64:
            continue
        rows.append([InlineKeyboardButton(wn, callback_data=cb_data)])
    for sw_id, sw in _user_shared_watchlists(user):
        cb_data = f"sa:{mt}:{media_id}:{sw_id}"
        if len(cb_data.encode('utf-8')) > 64:
            continue
        rows.append([InlineKeyboardButton(
            f"\U0001F465 {sw['name']}", callback_data=cb_data)])
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
        mode = state.user_data[user].get("mode", "movie")
    wls = state.user_data[user]["watchlists"][mode]
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
    if not edit_mode:
        for sw_id, sw in _user_shared_watchlists(user):
            count = len(sw.get("items", {}).get(mode, []))
            rows.append([InlineKeyboardButton(
                f"\U0001F465 {sw['name']} ({count})", callback_data=f"swb:{sw_id}")])
    if edit_mode:
        for sw_id, sw in _user_shared_watchlists(user):
            if sw["owner"] == user:
                rows.append([InlineKeyboardButton(
                    f"Delete \U0001F465 {sw['name']}?", callback_data=f"sdwl:{sw_id}")])
        rows.append([InlineKeyboardButton("Back", callback_data="wlback")])
    else:
        rows.append([
            InlineKeyboardButton("New watchlist", callback_data="nwl"),
            InlineKeyboardButton("New shared", callback_data="nswl"),
            InlineKeyboardButton("Edit", callback_data="wledit")
        ])
    return InlineKeyboardMarkup(rows)


def build_member_select_keyboard(user: int, selected_members: list) -> InlineKeyboardMarkup:
    rows = []
    other_users = [u for u in settings["allowed_users"] if u != user]
    for i, uid in enumerate(other_users):
        name = _get_user_display_name(uid)
        prefix = "\u2705 " if uid in selected_members else "\u274c "
        rows.append([InlineKeyboardButton(
            f"{prefix}{name}", callback_data=f"smu:{i}")])
    rows.append([InlineKeyboardButton("Done \u2705", callback_data="smd")])
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


def build_category_picker_keyboard(user, mode=None):
    if mode is None:
        mode = state.user_data[user].get("mode", "movie")
    wl_names = list(state.user_data[user]["watchlists"][mode].keys())
    rows = []
    for i, wn in enumerate(wl_names):
        rows.append([InlineKeyboardButton(wn, callback_data=f"wcat:{i}")])
    rows.append([InlineKeyboardButton("Skip", callback_data="wcat:s")])
    return InlineKeyboardMarkup(rows)


def build_recommend_category_keyboard(user, mode=None):
    if mode is None:
        mode = state.user_data[user].get("mode", "movie")
    wl_names = list(state.user_data[user]["watchlists"][mode].keys())
    rows = []
    for i, wn in enumerate(wl_names):
        rows.append([InlineKeyboardButton(wn, callback_data=f"rwl:{i}")])
    rows.append([InlineKeyboardButton("All", callback_data="rwl:all")])
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
