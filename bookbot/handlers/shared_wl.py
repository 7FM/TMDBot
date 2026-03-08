import logging

from telegram import ForceReply, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler

from botlib import state
from botlib.config import settings
from botlib.helpers import (
    esc, _mode_to_type, _type_to_mode,
    _get_shared_wl, _next_shared_wl_id,
    _user_shared_watchlists, _get_user_display_name,
    is_in_any_watchlist, find_all_shared_watchlists,
)
from botlib.keyboards import (
    build_media_keyboard, build_member_select_keyboard,
    build_watchlist_select_keyboard,
)
from botlib.messaging import send_movie_list
from bookbot.keyboards import get_main_keyboard
from bookbot.helpers import extract_book_info
from bookbot.config import ol_work

logger = logging.getLogger(__name__)


async def handle_nswl(query, user, raw):
    """Start new shared watchlist creation."""
    await query.answer()
    mode = "book"
    state._pending_shared_wl_name[user] = {
        "media_id": None, "mode": mode}
    await query.message.reply_text(
        "Enter a name for the shared list:",
        reply_markup=ForceReply(selective=True))


async def handle_smu(query, user, raw):
    """Toggle member selection."""
    await query.answer()
    if user not in state._pending_shared_wl_members:
        return
    idx_str = raw.split(":", 1)[1] if ":" in raw else ""
    try:
        idx = int(idx_str)
    except ValueError:
        return
    other_users = [u for u in settings["allowed_users"] if u != user]
    if idx >= len(other_users):
        return
    uid = other_users[idx]
    members = state._pending_shared_wl_members[user]["members"]
    if uid in members:
        members.remove(uid)
    else:
        members.append(uid)
    keyboard = build_member_select_keyboard(user, members)
    await query.edit_message_reply_markup(reply_markup=keyboard)


async def handle_smd(query, user, raw):
    """Done selecting members — create the shared watchlist."""
    await query.answer()
    if user not in state._pending_shared_wl_members:
        return
    info = state._pending_shared_wl_members.pop(user)
    name = info["name"]
    mode = info["mode"]
    members = info["members"]
    media_id = info.get("media_id")

    if not members:
        await query.message.reply_text(
            "No members selected. Shared list not created.",
            reply_markup=get_main_keyboard(user))
        return

    all_members = [user] + members
    sw_id = _next_shared_wl_id()
    state.user_data.setdefault("shared_watchlists", {})[sw_id] = {
        "name": name,
        "owner": user,
        "members": all_members,
        "items": {mode: []},
    }
    if media_id:
        state.user_data["shared_watchlists"][sw_id]["items"][mode].append(
            media_id)
    state.save_user_data()
    if media_id:
        from botlib.hooks import run_on_add
        run_on_add(media_id, mode, user, name)

    member_names = [_get_user_display_name(m) for m in members]
    await query.message.reply_text(
        f'Created shared list "{name}" with: {", ".join(member_names)}',
        reply_markup=get_main_keyboard(user))


async def handle_swb(query, user, raw):
    """Browse shared watchlist contents."""
    await query.answer()
    sw_id_str = raw.split(":", 1)[1] if ":" in raw else ""
    try:
        sw_id = int(sw_id_str)
    except ValueError:
        return
    sw = _get_shared_wl(sw_id)
    if not sw:
        return

    mode = "book"
    mt = _mode_to_type(mode)
    items = sw.get("items", {}).get(mode, [])
    if not items:
        await query.message.reply_text(
            f'Shared list "{sw["name"]}" is empty.',
            reply_markup=get_main_keyboard(user))
        return

    infos = []
    for work_id in items:
        try:
            data = ol_work(work_id)
            info = extract_book_info(data, from_search=False)
            if info[3] is not None:
                infos.append((info[3], data.get("title", "Unknown"), info[2]))
        except Exception:
            infos.append((work_id, f"Book {work_id}", f"Book ID: {work_id}"))

    await send_movie_list(
        query.message.get_bot(), query.message.chat_id,
        f'\U0001F465 "{sw["name"]}":', infos,
        detail_action="swdet", media_type=mt)


async def handle_swdet(query, user, raw):
    """Show detail from shared watchlist context."""
    await query.answer()
    parts = raw.split(":", 2)
    if len(parts) < 3:
        return
    try:
        work_id = int(parts[2])
    except ValueError:
        return
    from bookbot.helpers import extract_book_detail
    result = extract_book_detail(work_id)
    if not result:
        await query.message.reply_text("Could not load book details.")
        return
    cover_url, desc = result
    keyboard = build_media_keyboard(work_id, user, mode="book")
    escaped = esc(desc)
    if cover_url:
        await query.message.reply_photo(
            cover_url, escaped, parse_mode="MarkdownV2", reply_markup=keyboard)
    else:
        await query.message.reply_text(
            escaped, parse_mode="MarkdownV2", reply_markup=keyboard)


async def handle_sdwl(query, user, raw):
    """Delete shared watchlist confirmation."""
    await query.answer()
    sw_id_str = raw.split(":", 1)[1] if ":" in raw else ""
    try:
        sw_id = int(sw_id_str)
    except ValueError:
        return
    sw = _get_shared_wl(sw_id)
    if not sw or sw["owner"] != user:
        await query.message.reply_text("Only the owner can delete.")
        return
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Yes", callback_data=f"sdwly:{sw_id}"),
         InlineKeyboardButton("No", callback_data="sdwln")]
    ])
    await query.edit_message_text(
        f'Delete shared list "{sw["name"]}"?', reply_markup=keyboard)


async def handle_sdwly(query, user, raw):
    """Confirm delete shared watchlist."""
    await query.answer()
    sw_id_str = raw.split(":", 1)[1] if ":" in raw else ""
    try:
        sw_id = int(sw_id_str)
    except ValueError:
        return
    sw = state.user_data.get("shared_watchlists", {}).pop(sw_id, None)
    if sw:
        state.save_user_data()
    mode = "book"
    keyboard = build_watchlist_select_keyboard(user, mode=mode)
    await query.edit_message_text("Your reading lists:", reply_markup=keyboard)


async def handle_sdwln(query, user, raw):
    """Cancel delete shared watchlist."""
    await query.answer()
    mode = "book"
    keyboard = build_watchlist_select_keyboard(user, edit_mode=True, mode=mode)
    await query.edit_message_text("Your reading lists:", reply_markup=keyboard)


def register(app, router):
    router.add('nswl', handle_nswl)
    router.add('smu', handle_smu)
    router.add('smd', handle_smd)
    router.add('swb', handle_swb)
    router.add('swdet', handle_swdet)
    router.add('sdwl', handle_sdwl)
    router.add('sdwly', handle_sdwly)
    router.add('sdwln', handle_sdwln)
