import logging

from telegram import ForceReply, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import CommandHandler

from botlib import state
from botlib.base import BaseCommand
from botlib.helpers import (
    esc, parse_callback_data, _mode_to_type, _type_to_mode,
    is_in_any_watchlist, find_all_watchlists, get_watched_rating,
    _get_shared_wl, _user_shared_watchlists, _get_user_display_name,
    find_all_shared_watchlists,
)
from botlib.keyboards import (
    build_media_keyboard, build_watchlist_picker_keyboard,
    build_watchlist_select_keyboard,
)
from botlib.messaging import (
    send_back_text, send_movie_list,
    _cleanup_search_results, _is_search_message, _notify_shared_wl_members,
)
from bookbot.keyboards import get_main_keyboard
from bookbot.helpers import extract_book_info, extract_book_detail
from bookbot.config import ol_work

logger = logging.getLogger(__name__)


class ListCommand(BaseCommand):
    async def execute(self, update, context, user):
        mode = "book"
        keyboard = build_watchlist_select_keyboard(user, mode=mode)
        await update.message.reply_text("Your reading lists:", reply_markup=keyboard)


class AddCommand(BaseCommand):
    async def execute(self, update, context, user):
        if not context.args:
            await send_back_text(update, "Usage: /add <work_id>")
            return
        work_id_str = context.args[0]
        if not work_id_str.isdigit():
            await send_back_text(update, "Invalid book ID.")
            return
        work_id = int(work_id_str)
        mode = "book"
        wl = is_in_any_watchlist(work_id, user, mode=mode)
        if wl:
            await send_back_text(update, f'Already in your "{wl}" list.')
            return
        mt = _mode_to_type(mode)
        keyboard = build_watchlist_picker_keyboard(work_id, user, mode=mode)
        await update.message.reply_text(
            "Add to which list?", reply_markup=keyboard)


class RemoveCommand(BaseCommand):
    async def execute(self, update, context, user):
        if not context.args:
            await send_back_text(update, "Usage: /remove <work_id>")
            return
        work_id_str = context.args[0]
        if not work_id_str.isdigit():
            await send_back_text(update, "Invalid book ID.")
            return
        work_id = int(work_id_str)
        mode = "book"
        removed = False
        for wn, wl in state.user_data[user]["watchlists"][mode].items():
            if work_id in wl:
                wl.remove(work_id)
                removed = True
        if removed:
            state.save_user_data()
            await send_back_text(update, "Removed from your reading list.")
        else:
            await send_back_text(update, "Not in any reading list.")


async def _show_watchlist_contents(bot, chat_id, user, wl_name, mode="book"):
    """Show contents of a watchlist."""
    items = state.user_data[user]["watchlists"][mode].get(wl_name, [])
    if not items:
        await bot.send_message(chat_id, f'"{wl_name}" is empty.')
        return
    mt = _mode_to_type(mode)
    infos = []
    for work_id in items:
        try:
            data = ol_work(work_id)
            info = extract_book_info(data, from_search=False)
            if info[3] is not None:
                infos.append((info[3], data.get("title", "Unknown"), info[2]))
        except Exception:
            infos.append((work_id, f"Book {work_id}", f"Book ID: {work_id}"))

    await send_movie_list(bot, chat_id, f'"{wl_name}":', infos,
                          detail_action="det", media_type=mt)


async def handle_wl(query, user, raw):
    """Browse a watchlist."""
    await query.answer()
    wl_name = raw.split(":", 1)[1] if ":" in raw else ""
    mode = "book"
    await _show_watchlist_contents(
        query.message.get_bot(), query.message.chat_id, user, wl_name, mode=mode)


async def handle_fallback(query, user, raw):
    """Handle parse_callback_data-based actions (pick, back, a, rm, new, w)."""
    action, media_type, media_id, watchlist = parse_callback_data(raw)
    if action is None:
        await query.answer("Unknown action.", show_alert=True)
        return

    mode = _type_to_mode(media_type)
    bot = query.message.get_bot()
    chat_id = query.message.chat_id

    if action == "pick":
        await query.answer()
        keyboard = build_watchlist_picker_keyboard(media_id, user, mode=mode)
        if _is_search_message(user, query.message.message_id):
            await query.edit_message_reply_markup(reply_markup=keyboard)
        else:
            await query.message.reply_text(
                "Add to which list?", reply_markup=keyboard)

    elif action == "back":
        await query.answer()
        keyboard = build_media_keyboard(media_id, user, mode=mode)
        await query.edit_message_reply_markup(reply_markup=keyboard)

    elif action == "a":
        await query.answer()
        if not watchlist:
            return
        wl = state.user_data[user]["watchlists"][mode]
        if watchlist not in wl:
            return
        already = is_in_any_watchlist(media_id, user, mode=mode)
        if already:
            await query.message.reply_text(
                f'Already in your "{already}" list.',
                reply_markup=get_main_keyboard(user))
        else:
            # Check if already read
            watched_entry = state.user_data[user].get(
                "watched", {}).get(mode, {}).get(media_id)
            if watched_entry:
                r = get_watched_rating(watched_entry)
                rating_info = f" (rated {r}/10)" if r else ""
                await query.message.reply_text(
                    f'Already read{rating_info}. Adding anyway.',
                    reply_markup=get_main_keyboard(user))
            wl[watchlist].append(media_id)
            state.save_user_data()
            await query.message.reply_text(
                f'Added to "{watchlist}".',
                reply_markup=get_main_keyboard(user))
        await _cleanup_search_results(bot, user)

    elif action == "rm":
        await query.answer()
        removed_from = []
        for wn, wl in state.user_data[user]["watchlists"][mode].items():
            if media_id in wl:
                wl.remove(media_id)
                removed_from.append(wn)
        if removed_from:
            state.save_user_data()
            await query.message.reply_text(
                f'Removed from: {", ".join(removed_from)}.',
                reply_markup=get_main_keyboard(user))
        else:
            await query.message.reply_text(
                "Not in any reading list.",
                reply_markup=get_main_keyboard(user))
        await _cleanup_search_results(bot, user)

    elif action == "new":
        await query.answer()
        state._pending_new_watchlist[user] = (media_id, mode)
        await query.message.reply_text(
            "Enter a name for the new list:",
            reply_markup=ForceReply(selective=True))

    elif action == "w":
        # Mark as read — delegate to read handler
        from bookbot.handlers.read import handle_w_action
        await handle_w_action(query, user, raw)

    elif action == "sa":
        # Add to shared watchlist
        await query.answer()
        sw_id = int(watchlist) if watchlist else None
        if sw_id is None:
            return
        sw = _get_shared_wl(sw_id)
        if not sw:
            return
        items = sw.get("items", {}).get(mode, [])
        if media_id in items:
            await query.message.reply_text(
                f'Already in shared list "{sw["name"]}".',
                reply_markup=get_main_keyboard(user))
        else:
            if mode not in sw.get("items", {}):
                sw.setdefault("items", {})[mode] = []
            sw["items"][mode].append(media_id)
            state.save_user_data()
            display_name = _get_user_display_name(user)
            await query.message.reply_text(
                f'Added to shared list "{sw["name"]}".',
                reply_markup=get_main_keyboard(user))
            await _notify_shared_wl_members(
                bot, sw, user,
                f'{display_name} added a book to "{sw["name"]}".')
        await _cleanup_search_results(bot, user)

    elif action == "srm":
        # Remove from shared watchlist
        await query.answer()
        sw_id = int(watchlist) if watchlist else None
        if sw_id is None:
            return
        sw = _get_shared_wl(sw_id)
        if not sw:
            return
        items = sw.get("items", {}).get(mode, [])
        if media_id in items:
            items.remove(media_id)
            state.save_user_data()
            display_name = _get_user_display_name(user)
            await query.message.reply_text(
                f'Removed from shared list "{sw["name"]}".',
                reply_markup=get_main_keyboard(user))
            await _notify_shared_wl_members(
                bot, sw, user,
                f'{display_name} removed a book from "{sw["name"]}".')
        await _cleanup_search_results(bot, user)

    else:
        await query.answer("Unknown action.", show_alert=True)


async def handle_nwl(query, user, raw):
    """New watchlist from list view."""
    await query.answer()
    mode = "book"
    state._pending_new_watchlist[user] = (None, mode)
    await query.message.reply_text(
        "Enter a name for the new list:",
        reply_markup=ForceReply(selective=True))


async def handle_wledit(query, user, raw):
    await query.answer()
    mode = "book"
    keyboard = build_watchlist_select_keyboard(user, edit_mode=True, mode=mode)
    await query.edit_message_reply_markup(reply_markup=keyboard)


async def handle_wlback(query, user, raw):
    await query.answer()
    mode = "book"
    keyboard = build_watchlist_select_keyboard(
        user, edit_mode=False, mode=mode)
    await query.edit_message_reply_markup(reply_markup=keyboard)


async def handle_dwl(query, user, raw):
    """Delete watchlist confirmation."""
    await query.answer()
    wl_name = raw.split(":", 1)[1] if ":" in raw else ""
    mode = "book"
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("Yes, delete", callback_data=f"dwly:{wl_name}"),
         InlineKeyboardButton("No", callback_data="dwln")]
    ])
    await query.edit_message_text(
        f'Delete list "{wl_name}"?', reply_markup=keyboard)


async def handle_dwly(query, user, raw):
    """Confirm delete watchlist."""
    await query.answer()
    wl_name = raw.split(":", 1)[1] if ":" in raw else ""
    mode = "book"
    wls = state.user_data[user]["watchlists"][mode]
    if wl_name in wls:
        del wls[wl_name]
        state.save_user_data()
    keyboard = build_watchlist_select_keyboard(user, mode=mode)
    await query.edit_message_text("Your reading lists:", reply_markup=keyboard)


async def handle_dwln(query, user, raw):
    """Cancel delete watchlist."""
    await query.answer()
    mode = "book"
    keyboard = build_watchlist_select_keyboard(user, edit_mode=True, mode=mode)
    await query.edit_message_text("Your reading lists:", reply_markup=keyboard)


def register(app, router):
    app.add_handler(CommandHandler(['list', 'l'], ListCommand()))
    app.add_handler(CommandHandler('add', AddCommand()))
    app.add_handler(CommandHandler('remove', RemoveCommand()))
    router.add('wl', handle_wl)
    router.add('nwl', handle_nwl)
    router.add('wledit', handle_wledit)
    router.add('wlback', handle_wlback)
    router.add('dwl', handle_dwl)
    router.add('dwly', handle_dwly)
    router.add('dwln', handle_dwln)
    router.set_fallback(handle_fallback)
