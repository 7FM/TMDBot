import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ForceReply
from telegram.constants import ParseMode

from tmdbot import state
from tmdbot.config import settings, get_api
from tmdbot.helpers import (
    esc, extract_movie_info,
    _mode_to_type, _type_to_mode,
    _get_shared_wl, _next_shared_wl_id,
    _user_shared_watchlists, _get_user_display_name,
    is_in_any_watchlist, find_all_shared_watchlists,
)
from tmdbot.keyboards import (
    get_main_keyboard, build_media_keyboard,
    build_member_select_keyboard,
    build_watchlist_select_keyboard,
)
from tmdbot.messaging import send_movie_list

logger = logging.getLogger(__name__)


async def handle_nswl(query, user, raw):
    mode = state.user_data[user].get("mode", "movie")
    state._pending_shared_wl_name[user] = {"media_id": None, "mode": mode}
    await query.answer()
    await query.get_bot().send_message(
        query.message.chat_id,
        "Enter a name for the new shared watchlist:",
        reply_markup=ForceReply(selective=True))


async def handle_smu(query, user, raw):
    index = int(raw.split(":", 1)[1])
    if user not in state._pending_shared_wl_members:
        await query.answer("Session expired.", show_alert=True)
        return
    other_users = [u for u in settings["allowed_users"] if u != user]
    if index < 0 or index >= len(other_users):
        await query.answer("Invalid user.", show_alert=True)
        return
    uid = other_users[index]
    members = state._pending_shared_wl_members[user]["members"]
    if uid in members:
        members.remove(uid)
        await query.answer(f"Removed {_get_user_display_name(uid)}.")
    else:
        members.append(uid)
        await query.answer(f"Added {_get_user_display_name(uid)}.")
    keyboard = build_member_select_keyboard(user, members)
    await query.edit_message_reply_markup(reply_markup=keyboard)


async def handle_smd(query, user, raw):
    if user not in state._pending_shared_wl_members:
        await query.answer("Session expired.", show_alert=True)
        return
    s = state._pending_shared_wl_members[user]
    members = s["members"]
    if not members:
        await query.answer("Select at least one member!", show_alert=True)
        return
    state._pending_shared_wl_members.pop(user)
    all_members = [user] + members
    sw_id = _next_shared_wl_id()
    sw_name = s["name"]
    mode = s["mode"]
    media_id = s.get("media_id")
    items = {"movie": [], "tv": []}
    if media_id is not None:
        items[mode].append(media_id)
    state.user_data["shared_watchlists"][sw_id] = {
        "name": sw_name,
        "owner": user,
        "members": all_members,
        "items": items,
    }
    state.save_user_data()
    if media_id is not None:
        from botlib.hooks import run_on_add
        run_on_add(media_id, mode, user, sw_name,
                   query.message.get_bot(), query.message.chat_id)
    await query.answer(f'Created shared watchlist "{sw_name}"!')
    try:
        await query.message.delete()
    except Exception:
        await query.edit_message_reply_markup(reply_markup=None)
    msg = f'Created shared watchlist "{sw_name}" with {len(members)} member(s).'
    if media_id is not None:
        msg += " Item added."
    bot = query.get_bot()
    chat_id = query.message.chat_id
    await bot.send_message(
        chat_id, esc(msg),
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=get_main_keyboard(user))
    creator_name = _get_user_display_name(user)
    for member_id in members:
        try:
            await bot.send_message(
                member_id,
                esc(f'{creator_name} added you to the shared watchlist "{sw_name}".'),
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=get_main_keyboard(member_id))
        except Exception:
            logger.warning(
                f"Failed to notify user {member_id} about shared watchlist creation")


async def handle_sdwl(query, user, raw):
    try:
        sw_id = int(raw.split(":", 1)[1])
    except ValueError:
        await query.answer("Invalid.", show_alert=True)
        return
    sw = _get_shared_wl(sw_id)
    if sw is None:
        await query.answer("Not found.", show_alert=True)
        return
    if sw["owner"] != user:
        await query.answer("Only the owner can delete this.", show_alert=True)
        return
    count_m = len(sw.get("items", {}).get("movie", []))
    count_t = len(sw.get("items", {}).get("tv", []))
    total = count_m + count_t
    confirm_text = f'Delete shared watchlist "{sw["name"]}"?'
    if total > 0:
        confirm_text += f' It contains {total} item{"s" if total != 1 else ""}.'
    confirm_kb = InlineKeyboardMarkup([[
        InlineKeyboardButton(
            "Yes, delete", callback_data=f"sdwly:{sw_id}"),
        InlineKeyboardButton("Cancel", callback_data="sdwln")
    ]])
    await query.edit_message_text(
        text=esc(confirm_text),
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=confirm_kb)
    await query.answer()


async def handle_sdwly(query, user, raw):
    try:
        sw_id = int(raw.split(":", 1)[1])
    except ValueError:
        await query.answer("Invalid.", show_alert=True)
        return
    sw = _get_shared_wl(sw_id)
    if sw is None:
        await query.answer("Already deleted.", show_alert=True)
        return
    if sw["owner"] != user:
        await query.answer("Only the owner can delete.", show_alert=True)
        return
    sw_name = sw["name"]
    members = sw.get("members", [])
    del state.user_data["shared_watchlists"][sw_id]
    state.save_user_data()
    await query.answer(f'Deleted "{sw_name}".')
    keyboard = build_watchlist_select_keyboard(user)
    await query.edit_message_text(
        text="Select a watchlist:",
        reply_markup=keyboard)
    owner_name = _get_user_display_name(user)
    bot = query.get_bot()
    for member_id in members:
        if member_id != user:
            try:
                await bot.send_message(
                    member_id,
                    esc(f'{owner_name} deleted the shared watchlist "{sw_name}".'),
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_markup=get_main_keyboard(member_id))
            except Exception:
                pass


async def handle_sdwln(query, user, raw):
    keyboard = build_watchlist_select_keyboard(user)
    await query.edit_message_text(
        text="Select a watchlist:",
        reply_markup=keyboard)
    await query.answer()


async def handle_swb(query, user, raw):
    try:
        sw_id = int(raw.split(":", 1)[1])
    except (ValueError, IndexError):
        await query.answer("Invalid.", show_alert=True)
        return
    sw = _get_shared_wl(sw_id)
    if sw is None:
        await query.answer("Shared watchlist not found.", show_alert=True)
        return
    if user not in sw.get("members", []):
        await query.answer("You are not a member.", show_alert=True)
        return
    mode = state.user_data[user].get("mode", "movie")
    mt = _mode_to_type(mode)
    items = sw.get("items", {}).get(mode, [])
    if not items:
        await query.answer(f'"{sw["name"]}" has no {mode} items.', show_alert=True)
        return
    await query.answer()
    api = get_api(mode)
    movies_info = []
    for mid in items:
        try:
            details = api.details(mid)
            _, _, desc, _ = extract_movie_info(details, mode=mode)
            title = details.get("title") or details.get(
                "name") or "Unknown"
            movies_info.append((mid, title, desc))
        except Exception:
            continue
    await send_movie_list(
        query.get_bot(), query.message.chat_id,
        f'\U0001F465 {sw["name"]}:',
        movies_info, detail_action="swdet", media_type=mt)


async def handle_swdet(query, user, raw):
    parts = raw.split(":", 2)
    if len(parts) < 3:
        await query.answer("Invalid.", show_alert=True)
        return
    det_mt = parts[1]
    mid = int(parts[2])
    det_mode = _type_to_mode(det_mt)
    api = get_api(det_mode)
    try:
        details = api.details(mid, append_to_response="watch/providers")
    except Exception:
        await query.answer("Failed to load details.", show_alert=True)
        return
    _, poster_path, desc, _ = extract_movie_info(details, mode=det_mode)
    shared_wls = find_all_shared_watchlists(mid, user, mode=det_mode)
    buttons = []
    for sid, sname in shared_wls:
        cb = f"srm:{det_mt}:{mid}:{sid}"
        if len(cb.encode('utf-8')) <= 64:
            buttons.append([InlineKeyboardButton(
                f"Remove from \U0001F465 {sname}", callback_data=cb)])
    watchlist_name = is_in_any_watchlist(mid, user, mode=det_mode)
    if not watchlist_name:
        buttons.append([InlineKeyboardButton(
            "Add to personal", callback_data=f"pick:{det_mt}:{mid}")])
    already_watched = mid in state.user_data[user].get(
        "watched", {}).get(det_mode, {})
    if not already_watched:
        buttons.append([InlineKeyboardButton(
            "Watched", callback_data=f"w:{det_mt}:{mid}")])
    keyboard = InlineKeyboardMarkup(buttons)
    await query.answer()
    bot = query.get_bot()
    chat_id = query.message.chat_id
    if poster_path:
        await bot.send_photo(chat_id, poster_path, caption=esc(desc),
                             parse_mode=ParseMode.MARKDOWN_V2, reply_markup=keyboard)
    else:
        await bot.send_message(chat_id, esc(desc),
                               parse_mode=ParseMode.MARKDOWN_V2, reply_markup=keyboard)


def register(app, router):
    router.add('nswl', handle_nswl)
    router.add('smu', handle_smu)
    router.add('smd', handle_smd)
    router.add('sdwl', handle_sdwl)
    router.add('sdwly', handle_sdwly)
    router.add('sdwln', handle_sdwln)
    router.add('swb', handle_swb)
    router.add('swdet', handle_swdet)
