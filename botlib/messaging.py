import asyncio
import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, LinkPreviewOptions
from telegram.constants import ParseMode

from botlib import state
from botlib.helpers import esc, split_into_chunks, get_user_id
from botlib.keyboards import build_media_keyboard, build_chunk_keyboard

logger = logging.getLogger(__name__)

# Registry for domain-specific main keyboard function
_get_main_keyboard = None


def register_main_keyboard_fn(fn):
    global _get_main_keyboard
    _get_main_keyboard = fn


async def send_back_text(update: Update, msg, user=None):
    if user is None:
        user = get_user_id(update)
    kb = _get_main_keyboard(user) if _get_main_keyboard else None
    chunks = split_into_chunks(msg)
    sent = []
    for c in chunks:
        sent.append(await update.message.reply_text(
            esc(c), parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=kb))
    return sent


async def unauthorized_msg(update: Update) -> None:
    user_id = get_user_id(update)
    await send_back_text(update, f'*Unauthorized user detected!*\nPlease contact the bot admin to whitelist your user id = `{user_id}`.\nOtherwise, consider hosting your own bot instance. The source code is publicly available at [GitHub](https://github.com/7FM/TMDBot).')


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


async def send_movie_list(bot, chat_id, header: str, movies_info, detail_action: str = "det", media_type: str = "m"):
    """Send a chunked itemized list with expand/collapse buttons.

    movies_info: list of (media_id, title, description) tuples
    detail_action: callback action prefix for detail buttons (default "det")
    media_type: "m" or "tv" for callback data
    Returns list of sent Message objects.
    """
    if len(state._chunk_movies) > state._CHUNK_MOVIES_MAX:
        oldest_keys = sorted(state._chunk_movies.keys())[
            :len(state._chunk_movies) - state._CHUNK_MOVIES_MAX]
        for k in oldest_keys:
            del state._chunk_movies[k]
    chunk_text = header + "\n"
    chunk_movies_list = []
    max_size = 4096
    sent = []
    for mid, title, desc in movies_info:
        line = f"\u2022 {desc}\n"
        if len(chunk_text) + len(line) > max_size and chunk_movies_list:
            cid = state.next_chunk_id()
            state._chunk_movies[cid] = (
                chunk_movies_list, detail_action, media_type)
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
            cid = state.next_chunk_id()
            state._chunk_movies[cid] = (
                chunk_movies_list, detail_action, media_type)
            kb = build_chunk_keyboard(
                cid, chunk_movies_list, expanded=False, media_type=media_type)
        sent.append(await bot.send_message(
            chat_id=chat_id,
            text=esc(chunk_text),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=kb,
            link_preview_options=LinkPreviewOptions(is_disabled=True)))
    return sent


def _is_search_message(user, message_id):
    if user not in state._search_results:
        return False
    _, msg_ids = state._search_results[user]
    return message_id in msg_ids


async def _cleanup_search_results(bot, user):
    state._search_more.pop(user, None)
    state._pending_search.pop(user, None)
    if user not in state._search_results:
        return
    chat_id, msg_ids = state._search_results.pop(user)
    for mid in msg_ids:
        try:
            await bot.delete_message(chat_id, mid)
        except Exception:
            pass


async def _cleanup_rate_list(bot, user):
    if user not in state._rate_list_messages:
        return
    chat_id, msg_ids = state._rate_list_messages.pop(user)
    for mid in msg_ids:
        try:
            await bot.delete_message(chat_id, mid)
        except Exception:
            pass


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
    except Exception:
        pass
    # Delete progress message
    try:
        await bot.delete_message(chat_id, msg_id)
    except Exception:
        pass
    return result


async def _notify_shared_wl_members(bot, sw, acting_user, message_text, keyboard=None):
    for member_id in sw.get("members", []):
        if member_id != acting_user:
            try:
                kb = keyboard or (_get_main_keyboard(
                    member_id) if _get_main_keyboard else None)
                await bot.send_message(
                    member_id, esc(message_text),
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_markup=kb)
            except Exception:
                logger.warning(f"Failed to notify user {member_id}")
