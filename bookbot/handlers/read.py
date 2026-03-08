import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler

from botlib import state
from botlib.base import BaseCommand
from botlib.helpers import (
    esc, _type_to_mode, _mode_to_type,
    is_in_any_watchlist, find_all_watchlists,
    find_all_shared_watchlists, get_watched_rating,
    get_watched_category, _get_shared_wl, _get_user_display_name,
)
from botlib.keyboards import (
    build_rating_keyboard, build_category_picker_keyboard,
)
from botlib.messaging import (
    send_back_text, send_movie_list,
    _cleanup_search_results, _cleanup_rate_list,
    _is_search_message, _notify_shared_wl_members, _with_progress_bar,
)
from bookbot.keyboards import get_main_keyboard
from bookbot.helpers import extract_book_info
from bookbot.config import ol_work

logger = logging.getLogger(__name__)


class ReadCommand(BaseCommand):
    """Mark a book as read by ID."""

    async def execute(self, update, context, user):
        if not context.args:
            await send_back_text(update, "Usage: /read <work_id>")
            return
        work_id_str = context.args[0]
        if not work_id_str.isdigit():
            await send_back_text(update, "Invalid book ID.")
            return
        work_id = int(work_id_str)
        mode = "book"
        mt = _mode_to_type(mode)

        prev_wls = find_all_watchlists(work_id, user, mode=mode)
        if prev_wls:
            category = prev_wls[0]
        else:
            category = None

        keyboard = build_rating_keyboard(work_id, media_type=mt)
        await update.message.reply_text(
            "Rate this book (1-10, or skip):", reply_markup=keyboard)


class RateCommand(BaseCommand):
    """Rate or re-rate read books."""

    async def execute(self, update, context, user):
        mode = "book"
        watched = state.user_data[user].get("watched", {}).get(mode, {})
        if not watched:
            await send_back_text(update, "You haven't read any books yet.")
            return
        await _send_rate_list(update, user)


async def _send_rate_list(update, user):
    """Show list of read books for rating."""
    mode = "book"
    mt = _mode_to_type(mode)
    watched = state.user_data[user].get("watched", {}).get(mode, {})
    items = list(watched.items())
    if not items:
        await send_back_text(update, "No read books to rate.")
        return

    bot = update.get_bot()
    chat_id = update.message.chat_id

    import concurrent.futures
    import multiprocessing

    def fetch_all(tick):
        results = []
        max_workers = min(len(items), multiprocessing.cpu_count() * 2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for mid, entry in items:
                futures[pool.submit(ol_work, mid)] = (mid, entry)
            for f in concurrent.futures.as_completed(futures):
                mid, entry = futures[f]
                try:
                    data = f.result()
                    info = extract_book_info(data, from_search=False)
                    title = data.get("title", f"Book {mid}")
                    r = get_watched_rating(entry)
                    cat = get_watched_category(entry)
                    rating_str = f" [{r}/10]" if r else " [unrated]"
                    cat_str = f" [{cat}]" if cat else ""
                    results.append(
                        (mid, title, info[2] + rating_str + cat_str))
                except Exception:
                    results.append((mid, f"Book {mid}", f"Book {mid}"))
                tick()
        return results

    await _cleanup_rate_list(bot, user)
    infos = await _with_progress_bar(
        bot, chat_id, "Loading your books...", len(items), fetch_all)

    sent = await send_movie_list(
        bot, chat_id, "Your read books (tap to rate):",
        infos, detail_action="rdet", media_type=mt)
    state._rate_list_messages[user] = (
        chat_id, [m.message_id for m in sent])


async def handle_w_action(query, user, raw):
    """Handle 'w' (mark as read) callback."""
    await query.answer()
    action, media_type, media_id, watchlist = (
        raw.split(":")[0], raw.split(":")[1],
        int(raw.split(":")[2]), None)
    mode = _type_to_mode(media_type)
    bot = query.message.get_bot()

    prev_wls = find_all_watchlists(media_id, user, mode=mode)
    if prev_wls:
        category = prev_wls[0]
        _continue_read_flow(query, user, media_id, media_type, mode, category)
        await _do_continue_read(query, user, media_id, media_type, mode, category)
    else:
        # Not in any watchlist — ask for category
        state._pending_watched_category[user] = {
            "mid": media_id,
            "media_type": media_type,
            "mode": mode,
        }
        keyboard = build_category_picker_keyboard(user, mode=mode)
        await query.message.reply_text(
            "Which category does this book belong to?",
            reply_markup=keyboard)
        await _cleanup_search_results(bot, user)


async def _do_continue_read(query, user, media_id, media_type, mode, category):
    """Show rating keyboard."""
    bot = query.message.get_bot()
    keyboard = build_rating_keyboard(media_id, media_type=media_type)
    await query.message.reply_text(
        "Rate this book (1-10, or skip):", reply_markup=keyboard)
    await _cleanup_search_results(bot, user)


def _continue_read_flow(query, user, media_id, media_type, mode, category):
    """Store category in pending state for the rating handler."""
    state._pending_watched_category[user] = {
        "mid": media_id,
        "media_type": media_type,
        "mode": mode,
        "category": category,
    }


async def handle_wcat(query, user, raw):
    """Category picker callback."""
    await query.answer()
    if user not in state._pending_watched_category:
        return

    pending = state._pending_watched_category[user]
    mode = pending["mode"]
    media_id = pending["mid"]
    media_type = pending["media_type"]

    idx_str = raw.split(":", 1)[1] if ":" in raw else ""
    if idx_str == "s":
        category = None
    else:
        try:
            idx = int(idx_str)
            wl_names = list(state.user_data[user]["watchlists"][mode].keys())
            category = wl_names[idx] if idx < len(wl_names) else None
        except (ValueError, IndexError):
            category = None

    if pending.get("change_only"):
        # Just changing category of already-read item
        state._pending_watched_category.pop(user)
        watched = state.user_data[user].get("watched", {}).get(mode, {})
        entry = watched.get(media_id)
        if isinstance(entry, dict):
            entry["category"] = category
            state.save_user_data()
        await query.message.reply_text(
            f"Category updated to: {category or 'none'}",
            reply_markup=get_main_keyboard(user))
        return

    pending["category"] = category
    keyboard = build_rating_keyboard(media_id, media_type=media_type)
    await query.message.reply_text(
        "Rate this book (1-10, or skip):", reply_markup=keyboard)


async def handle_ccat(query, user, raw):
    """Initiate change-category for already-read item."""
    await query.answer()
    parts = raw.split(":", 2)
    if len(parts) < 3:
        return
    media_type = parts[1]
    try:
        media_id = int(parts[2])
    except ValueError:
        return
    mode = _type_to_mode(media_type)
    state._pending_watched_category[user] = {
        "mid": media_id,
        "media_type": media_type,
        "mode": mode,
        "change_only": True,
    }
    keyboard = build_category_picker_keyboard(user, mode=mode)
    await query.message.reply_text(
        "Select new category:", reply_markup=keyboard)


async def handle_rate(query, user, raw):
    """Handle rating callback (from search/watchlist context)."""
    await query.answer()
    parts = raw.split(":")
    if len(parts) < 4:
        return
    media_type = parts[1]
    try:
        mid = int(parts[2])
        rating = int(parts[3])
    except ValueError:
        return
    rate_mode = _type_to_mode(media_type)
    if rating == 0:
        rating = None

    # Determine category
    prev_wls = find_all_watchlists(mid, user, mode=rate_mode)
    prev_entry = state.user_data[user].get(
        "watched", {}).get(rate_mode, {}).get(mid, "absent")
    if prev_wls:
        category = prev_wls[0]
    elif user in state._pending_watched_category:
        category = state._pending_watched_category.pop(user).get("category")
    else:
        category = get_watched_category(
            prev_entry) if prev_entry != "absent" else None

    # Remove from personal watchlists
    for wn, wl in state.user_data[user]["watchlists"][rate_mode].items():
        if mid in wl:
            wl.remove(mid)

    state.user_data[user].setdefault("watched", {}).setdefault(rate_mode, {})
    state.user_data[user]["watched"][rate_mode][mid] = {
        "rating": rating, "category": category}

    # Store undo state
    state._last_watched[user] = {
        "mid": mid,
        "mode": rate_mode,
        "media_type": media_type,
        "watchlists": prev_wls,
        "prev_entry": prev_entry,
    }

    state.save_user_data()
    bot = query.message.get_bot()
    chat_id = query.message.chat_id

    rating_str = f" ({rating}/10)" if rating else ""
    await bot.send_message(
        chat_id, esc(f"Marked as read{rating_str}."),
        reply_markup=get_main_keyboard(user),
        parse_mode="MarkdownV2")

    # Send undo button
    undo_kb = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Undo?", callback_data="undo")]])
    await bot.send_message(chat_id, "Undo?", reply_markup=undo_kb)

    await _cleanup_search_results(bot, user)

    # Notify shared watchlist members if applicable
    shared_wls = find_all_shared_watchlists(mid, user, mode=rate_mode)
    if shared_wls:
        display_name = _get_user_display_name(user)
        for sw_id, sw_name in shared_wls:
            sw = _get_shared_wl(sw_id)
            if sw:
                await _notify_shared_wl_members(
                    bot, sw, user,
                    f'{display_name} finished reading a book from "{sw_name}"{rating_str}.')


async def handle_rrate(query, user, raw):
    """Handle rating callback from /rate list (refreshes list)."""
    await handle_rate(query, user, raw.replace("rrate:", "rate:", 1))
    # Refresh rate list
    bot = query.message.get_bot()
    await _cleanup_rate_list(bot, user)


async def handle_undo(query, user, raw):
    """Undo last 'mark as read'."""
    await query.answer()
    if user not in state._last_watched:
        await query.message.reply_text("Nothing to undo.",
                                       reply_markup=get_main_keyboard(user))
        return

    info = state._last_watched.pop(user)
    mid = info["mid"]
    mode = info["mode"]
    prev_entry = info.get("prev_entry", "absent")
    prev_wls = info.get("watchlists", [])

    # Restore watched state
    if prev_entry == "absent":
        state.user_data[user]["watched"][mode].pop(mid, None)
    else:
        state.user_data[user]["watched"][mode][mid] = prev_entry

    # Restore watchlist placement
    for wn in prev_wls:
        if wn in state.user_data[user]["watchlists"][mode]:
            if mid not in state.user_data[user]["watchlists"][mode][wn]:
                state.user_data[user]["watchlists"][mode][wn].append(mid)

    state.save_user_data()

    try:
        await query.edit_message_text("Undone!")
    except Exception:
        pass
    await query.message.reply_text("Undone! Book restored.",
                                   reply_markup=get_main_keyboard(user))


def register(app, router):
    app.add_handler(CommandHandler(['read', 'watched'], ReadCommand()))
    app.add_handler(CommandHandler('rate', RateCommand()))
    router.add('rate', handle_rate)
    router.add('rrate', handle_rrate)
    router.add('undo', handle_undo)
    router.add('wcat', handle_wcat)
    router.add('ccat', handle_ccat)
