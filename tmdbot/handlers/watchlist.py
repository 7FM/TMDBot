import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ForceReply
from telegram.constants import ParseMode
from telegram.ext import CommandHandler

from tmdbot import state
from tmdbot.config import settings, get_api
from tmdbot.base import BaseCommand
from tmdbot.helpers import (
    get_user_id, esc, extract_movie_info, is_valid_media_id,
    is_in_any_watchlist, find_all_watchlists,
    _mode_to_type, _type_to_mode,
    _get_shared_wl, _user_shared_watchlists,
    _get_user_display_name, find_all_shared_watchlists,
)
from tmdbot.keyboards import (
    get_main_keyboard, build_media_keyboard,
    build_watchlist_picker_keyboard,
    build_watchlist_select_keyboard,
)
from tmdbot.messaging import (
    send_back_text, send_movie_list,
    _cleanup_search_results, _is_search_message,
    _notify_shared_wl_members,
)

logger = logging.getLogger(__name__)


class ListCommand(BaseCommand):
    async def execute(self, update, context, user):
        keyboard = build_watchlist_select_keyboard(user)
        await update.message.reply_text(
            "Select a watchlist:",
            reply_markup=keyboard
        )


class AddCommand(BaseCommand):
    async def execute(self, update, context, user):
        if not context.args:
            await send_back_text(update, 'Please provide the ID.')
            return
        mode = state.user_data[user].get("mode", "movie")
        watchlist = "normal"
        if len(context.args) == 2:
            watchlist = context.args[1]
        if watchlist not in state.user_data[user]["watchlists"][mode]:
            await send_back_text(update, f'Info: creating new watchlist "{watchlist}"')
            state.user_data[user]["watchlists"][mode][watchlist] = []
        media_id = context.args[0]
        await _add_to_watchlist_helper(watchlist, media_id, user, update)


class TrashAddCommand(BaseCommand):
    async def execute(self, update, context, user):
        if not context.args:
            await send_back_text(update, 'Please provide the ID.')
            return
        media_id = context.args[0]
        await _add_to_watchlist_helper("trash", media_id, user, update)


class RemoveCommand(BaseCommand):
    async def execute(self, update, context, user):
        if not context.args:
            await send_back_text(update, 'Please provide the ID.')
            return
        mode = state.user_data[user].get("mode", "movie")
        media_id = context.args[0]
        err_msg = is_valid_media_id(media_id, mode)
        if err_msg:
            await send_back_text(update, f'The provided ID is invalid: ' + err_msg)
            return
        media_id = int(media_id)
        removed_smth = False
        for _, w in state.user_data[user]["watchlists"][mode].items():
            if media_id in w:
                w.remove(media_id)
                removed_smth = True
        if removed_smth:
            state.save_user_data()
            await send_back_text(update, 'Removed from watchlist.')
        else:
            await send_back_text(update, 'Not in any watchlist.')


async def _add_to_watchlist_helper(watchlist, media_id, user, update):
    mode = state.user_data[user].get("mode", "movie")
    err_msg = is_valid_media_id(media_id, mode)
    if err_msg:
        await send_back_text(update, f'The provided ID is invalid: ' + err_msg)
        return
    media_id = int(media_id)
    already_in = is_in_any_watchlist(media_id, user, mode=mode)
    if already_in:
        await send_back_text(update, f'Already in your "{already_in}" watchlist.')
    else:
        if media_id in state.user_data[user]["watched"][mode]:
            prev_rating = state.user_data[user]["watched"][mode][media_id]
            if isinstance(prev_rating, (int, float)) and prev_rating > 0:
                await send_back_text(update, f"Warning: you already watched this (rated {prev_rating}/10)!")
            else:
                await send_back_text(update, "Warning: you have already watched this!")
        state.user_data[user]["watchlists"][mode][watchlist].append(media_id)
        state.save_user_data()
        await send_back_text(update, 'Added to watchlist.')


# Callback handlers

async def handle_nwl(query, user, raw):
    mode = state.user_data[user].get("mode", "movie")
    state._pending_new_watchlist[user] = (None, mode)
    await query.answer()
    await query.get_bot().send_message(
        query.message.chat_id,
        "Enter a name for the new watchlist:",
        reply_markup=ForceReply(selective=True))


async def handle_wl(query, user, raw):
    wl_name = raw.split(":", 1)[1]
    mode = state.user_data[user].get("mode", "movie")
    wl_movies = state.user_data[user]["watchlists"][mode].get(wl_name, [])
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


async def handle_wledit(query, user, raw):
    keyboard = build_watchlist_select_keyboard(user, edit_mode=True)
    await query.edit_message_reply_markup(reply_markup=keyboard)
    await query.answer()


async def handle_wlback(query, user, raw):
    keyboard = build_watchlist_select_keyboard(user)
    await query.edit_message_reply_markup(reply_markup=keyboard)
    await query.answer()


async def handle_dwl(query, user, raw):
    wl_name = raw.split(":", 1)[1]
    await query.answer()
    mode = state.user_data[user].get("mode", "movie")
    count = len(state.user_data[user]["watchlists"][mode].get(wl_name, []))
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


async def handle_dwly(query, user, raw):
    wl_name = raw.split(":", 1)[1]
    mode = state.user_data[user].get("mode", "movie")
    if wl_name in state.user_data[user]["watchlists"][mode]:
        del state.user_data[user]["watchlists"][mode][wl_name]
        state.save_user_data()
        await query.answer(f'Deleted "{wl_name}".')
    else:
        await query.answer("Watchlist not found.", show_alert=True)
    keyboard = build_watchlist_select_keyboard(user, edit_mode=True)
    await query.edit_message_text(
        text="Select a watchlist:",
        reply_markup=keyboard)


async def handle_dwln(query, user, raw):
    keyboard = build_watchlist_select_keyboard(user, edit_mode=True)
    await query.edit_message_text(
        text="Select a watchlist:",
        reply_markup=keyboard)
    await query.answer()


async def _handle_fallback(query, user, raw):
    """Handle parse_callback_data-based actions: pick, back, a, rm, new, sa, srm, w, ws."""
    from tmdbot.helpers import parse_callback_data
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
        state._pending_new_watchlist[user] = (movie_id, cb_mode)
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
            if movie_id in state.user_data[user]["watched"][cb_mode]:
                prev_rating = state.user_data[user]["watched"][cb_mode][movie_id]
                if isinstance(prev_rating, (int, float)) and prev_rating > 0:
                    await query.answer(f"Warning: you already watched this (rated {prev_rating}/10)!", show_alert=True)
                else:
                    await query.answer("Warning: you already watched this!", show_alert=True)
            else:
                await query.answer(f'Added to "{watchlist}".')
            state.user_data[user]["watchlists"][cb_mode][watchlist].append(movie_id)
            state.save_user_data()
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

    elif action == "sa":
        await _handle_sa(query, user, movie_id, watchlist, cb_mode, media_type)

    elif action == "srm":
        await _handle_srm(query, user, movie_id, watchlist, cb_mode, media_type)

    elif action == "rm":
        removed = False
        for _, w in state.user_data[user]["watchlists"][cb_mode].items():
            if movie_id in w:
                w.remove(movie_id)
                removed = True
        if removed:
            state.save_user_data()
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
        from tmdbot.handlers.watched import handle_w_action
        await handle_w_action(query, user, movie_id, media_type, cb_mode)

    elif action == "ws":
        from tmdbot.handlers.watched import handle_ws_action
        await handle_ws_action(query, user, raw, movie_id, media_type)

    else:
        await query.answer("Unknown action.", show_alert=True)


async def _handle_sa(query, user, movie_id, watchlist, cb_mode, media_type):
    try:
        sw_id = int(watchlist)
    except (ValueError, TypeError):
        await query.answer("Invalid.", show_alert=True)
        return
    sw = _get_shared_wl(sw_id)
    if sw is None:
        await query.answer("Shared watchlist not found.", show_alert=True)
        return
    if user not in sw.get("members", []):
        await query.answer("You are not a member.", show_alert=True)
        return
    items = sw.get("items", {}).get(cb_mode, [])
    if movie_id in items:
        await query.answer(f'Already in "{sw["name"]}".', show_alert=True)
        return
    if movie_id in state.user_data[user]["watched"][cb_mode]:
        prev_rating = state.user_data[user]["watched"][cb_mode][movie_id]
        if isinstance(prev_rating, (int, float)) and prev_rating > 0:
            await query.answer(f"Warning: you already watched this (rated {prev_rating}/10)!", show_alert=True)
        else:
            await query.answer("Warning: you already watched this!", show_alert=True)
    else:
        await query.answer(f'Added to "{sw["name"]}".')
    sw["items"].setdefault(cb_mode, []).append(movie_id)
    state.save_user_data()
    bot = query.get_bot()
    if _is_search_message(user, query.message.message_id):
        await _cleanup_search_results(bot, user)
        await bot.send_message(
            query.message.chat_id,
            esc(f'Added to shared watchlist "{sw["name"]}".'),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=get_main_keyboard(user))
    else:
        new_keyboard = build_media_keyboard(movie_id, user, mode=cb_mode)
        await query.edit_message_reply_markup(reply_markup=new_keyboard)
    try:
        api = get_api(cb_mode)
        details = api.details(movie_id)
        title = details.get("title") or details.get("name") or "Unknown"
    except Exception:
        title = str(movie_id)
    await _notify_shared_wl_members(
        bot, sw, user,
        f'{_get_user_display_name(user)} added "{title}" to "{sw["name"]}".')


async def _handle_srm(query, user, movie_id, watchlist, cb_mode, media_type):
    try:
        sw_id = int(watchlist)
    except (ValueError, TypeError):
        await query.answer("Invalid.", show_alert=True)
        return
    sw = _get_shared_wl(sw_id)
    if sw is None:
        await query.answer("Shared watchlist not found.", show_alert=True)
        return
    if user not in sw.get("members", []):
        await query.answer("You are not a member.", show_alert=True)
        return
    items = sw.get("items", {}).get(cb_mode, [])
    if movie_id not in items:
        await query.answer("Item not in this watchlist.", show_alert=True)
        return
    items.remove(movie_id)
    state.save_user_data()
    await query.answer(f'Removed from "{sw["name"]}".')
    try:
        await query.message.delete()
    except Exception:
        await query.edit_message_reply_markup(reply_markup=None)
    try:
        api = get_api(cb_mode)
        details = api.details(movie_id)
        title = details.get("title") or details.get("name") or "Unknown"
    except Exception:
        title = str(movie_id)
    mt = _mode_to_type(cb_mode)
    actor_name = _get_user_display_name(user)
    for member_id in sw.get("members", []):
        if member_id == user:
            continue
        already_watched = movie_id in state.user_data.get(
            member_id, {}).get("watched", {}).get(cb_mode, {})
        try:
            if already_watched:
                await query.get_bot().send_message(
                    member_id,
                    esc(
                        f'{actor_name} removed "{title}" from "{sw["name"]}".'),
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_markup=get_main_keyboard(member_id))
            else:
                watched_kb = InlineKeyboardMarkup([[
                    InlineKeyboardButton(
                        "Watched", callback_data=f"w:{mt}:{movie_id}")
                ]])
                await query.get_bot().send_message(
                    member_id,
                    esc(
                        f'{actor_name} removed "{title}" from "{sw["name"]}". Mark as watched?'),
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_markup=watched_kb)
        except Exception:
            logger.warning(f"Failed to notify user {member_id}")


def register(app, router):
    app.add_handler(CommandHandler(['list', 'l'], ListCommand()))
    app.add_handler(CommandHandler(['add', 'a'], AddCommand()))
    app.add_handler(CommandHandler(['tadd', 't'], TrashAddCommand()))
    app.add_handler(CommandHandler(['remove', 'rm'], RemoveCommand()))
    router.add('nwl', handle_nwl)
    router.add('wl', handle_wl)
    router.add('wledit', handle_wledit)
    router.add('wlback', handle_wlback)
    router.add('dwl', handle_dwl)
    router.add('dwly', handle_dwly)
    router.add('dwln', handle_dwln)
    router.set_fallback(_handle_fallback)
