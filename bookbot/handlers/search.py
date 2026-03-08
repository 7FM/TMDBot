import logging

from telegram import ForceReply, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import CommandHandler

from botlib import state
from botlib.base import BaseCommand
from botlib.helpers import esc, _mode_to_type
from botlib.keyboards import build_media_keyboard, build_chunk_keyboard, build_rating_keyboard
from botlib.messaging import (
    send_back_text, send_movie_message, send_movie_list,
    _cleanup_search_results, _is_search_message,
)
from bookbot.config import ol_search
from bookbot.helpers import extract_book_info, extract_book_detail, sort_by_rating

logger = logging.getLogger(__name__)


class SearchCommand(BaseCommand):
    async def execute(self, update, context, user):
        query = " ".join(context.args) if context.args else None
        if not query:
            state._pending_search[user] = True
            await update.message.reply_text(
                "What book are you looking for?",
                reply_markup=ForceReply(selective=True))
            return
        await do_search(update, query, user)


async def do_search(update, query, user):
    await _cleanup_search_results(update.get_bot(), user)
    try:
        results = ol_search(query, limit=25)
    except Exception as e:
        logger.error("OL search failed: %s", e)
        await send_back_text(update, "Search failed. Please try again.")
        return

    if not results:
        await send_back_text(update, f'No results for "{query}".')
        return

    mode = "book"
    watched = state.user_data[user].get("watched", {}).get(mode, {})
    infos = []
    for doc in results:
        info = extract_book_info(doc, from_search=True)
        if info[3] is not None and info[3] not in watched:
            infos.append(info)

    if not infos:
        await send_back_text(update, f'No new results for "{query}" (all already read).')
        return

    sorted_infos = sort_by_rating(infos)
    first_5 = sorted_infos[:5]
    remaining = sorted_infos[5:]

    if remaining:
        state._search_more[user] = (remaining, query)

    msg_ids = []
    for rating, cover_url, desc, work_id in first_5:
        msg = await send_movie_message(
            update, desc, cover_url, work_id, user, mode=mode)
        msg_ids.append(msg.message_id)

    if remaining:
        btn = InlineKeyboardButton(
            f"Show more ({len(remaining)} remaining)", callback_data="smore")
        more_msg = await update.message.reply_text(
            "More results available:",
            reply_markup=InlineKeyboardMarkup([[btn]]))
        msg_ids.append(more_msg.message_id)

    state._search_results[user] = (update.message.chat_id, msg_ids)


async def handle_smore(query, user, raw):
    await query.answer()
    if user not in state._search_more:
        return

    remaining, search_query = state._search_more[user]
    next_5 = remaining[:5]
    rest = remaining[5:]
    bot = query.message.get_bot()
    chat_id = query.message.chat_id
    mode = "book"

    # Get current tracked message ids
    old_chat_id, old_msg_ids = state._search_results.get(user, (chat_id, []))

    msg_ids = list(old_msg_ids)
    # Delete the "show more" button message
    try:
        await bot.delete_message(chat_id, query.message.message_id)
    except Exception:
        pass
    if query.message.message_id in msg_ids:
        msg_ids.remove(query.message.message_id)

    for rating, cover_url, desc, work_id in next_5:
        keyboard = build_media_keyboard(work_id, user, mode=mode)
        escaped = esc(desc)
        if cover_url:
            msg = await bot.send_photo(
                chat_id, cover_url, escaped,
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=keyboard)
        else:
            msg = await bot.send_message(
                chat_id, escaped,
                parse_mode=ParseMode.MARKDOWN_V2,
                reply_markup=keyboard)
        msg_ids.append(msg.message_id)

    if rest:
        state._search_more[user] = (rest, search_query)
        btn = InlineKeyboardButton(
            f"Show more ({len(rest)} remaining)", callback_data="smore")
        more_msg = await bot.send_message(
            chat_id, "More results available:",
            reply_markup=InlineKeyboardMarkup([[btn]]))
        msg_ids.append(more_msg.message_id)
    else:
        state._search_more.pop(user, None)

    state._search_results[user] = (chat_id, msg_ids)


async def handle_detail(query, user, raw):
    """Show detail card for a book."""
    await query.answer()
    parts = raw.split(":", 2)
    if len(parts) < 3:
        return
    media_type = parts[1]
    try:
        work_id = int(parts[2])
    except ValueError:
        return

    result = extract_book_detail(work_id)
    if not result:
        await query.message.reply_text("Could not load book details.")
        return

    cover_url, desc = result
    keyboard = build_media_keyboard(work_id, user, mode="book")
    escaped = esc(desc)
    if cover_url:
        await query.message.reply_photo(
            cover_url, escaped,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard)
    else:
        await query.message.reply_text(
            escaped,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard)


async def handle_rdet(query, user, raw):
    """Show detail card with rating keyboard (from /rate)."""
    await query.answer()
    parts = raw.split(":", 2)
    if len(parts) < 3:
        return
    det_mt = parts[1]
    try:
        mid = int(parts[2])
    except ValueError:
        return

    result = extract_book_detail(mid)
    if not result:
        await query.message.reply_text("Could not load book details.")
        return

    cover_url, desc = result
    rating_kb = build_rating_keyboard(
        mid, media_type=det_mt, action_prefix="rrate")
    rows = list(rating_kb.inline_keyboard) + [
        [InlineKeyboardButton(
            "Change category", callback_data=f"ccat:{det_mt}:{mid}")]]
    keyboard = InlineKeyboardMarkup(rows)
    escaped = esc(desc)
    if cover_url:
        await query.message.reply_photo(
            cover_url, escaped,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard)
    else:
        await query.message.reply_text(
            escaped,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard)


async def handle_expand(query, user, raw):
    """Expand a chunk list."""
    await query.answer()
    try:
        chunk_id = int(raw.split(":")[1])
    except (IndexError, ValueError):
        return
    if chunk_id not in state._chunk_movies:
        return
    movies_list, detail_action, media_type = state._chunk_movies[chunk_id]
    kb = build_chunk_keyboard(chunk_id, movies_list, expanded=True,
                              detail_action=detail_action, media_type=media_type)
    await query.edit_message_reply_markup(reply_markup=kb)


async def handle_collapse(query, user, raw):
    """Collapse a chunk list."""
    await query.answer()
    try:
        chunk_id = int(raw.split(":")[1])
    except (IndexError, ValueError):
        return
    if chunk_id not in state._chunk_movies:
        return
    movies_list, detail_action, media_type = state._chunk_movies[chunk_id]
    kb = build_chunk_keyboard(chunk_id, movies_list, expanded=False,
                              detail_action=detail_action, media_type=media_type)
    await query.edit_message_reply_markup(reply_markup=kb)


async def default_search_handler(update, context):
    """Handle plain text (non-command, non-reply) as search."""
    from botlib.helpers import get_user_id, check_user_invalid
    from botlib.messaging import unauthorized_msg
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return
    if not state.user_data.get(user, {}).get("onboarded", False):
        return
    text = update.message.text.strip()
    if text:
        await do_search(update, text, user)


def register(app, router):
    app.add_handler(CommandHandler(['search', 's'], SearchCommand()))
    router.add('smore', handle_smore)
    router.add('det', handle_detail)
    router.add('rdet', handle_rdet)
    router.add('exp', handle_expand)
    router.add('col', handle_collapse)
