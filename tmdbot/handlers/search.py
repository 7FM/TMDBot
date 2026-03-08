from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ForceReply
from telegram.constants import ParseMode
from telegram.ext import CommandHandler

from tmdbot import state
from tmdbot.config import get_api
from tmdbot.base import BaseCommand
from tmdbot.helpers import (
    extract_movie_info, sort_by_rating, esc,
    _mode_to_type, _type_to_mode,
)
from tmdbot.keyboards import build_media_keyboard, build_chunk_keyboard
from tmdbot.messaging import (
    send_back_text, send_movie_message, send_movie_list,
    _cleanup_search_results, _is_search_message,
)


class SearchCommand(BaseCommand):
    async def execute(self, update, context, user):
        if not context.args:
            state._pending_search[user] = True
            mode = state.user_data[user].get("mode", "movie")
            label = "movie" if mode == "movie" else "TV show"
            await update.message.reply_text(
                f"Enter a {label} title to search:",
                reply_markup=ForceReply(selective=True))
            return
        query = ' '.join(context.args)
        await do_search(update, query, user)


async def do_search(update, query, user):
    from tmdbot.config import search
    mode = state.user_data[user].get("mode", "movie")
    if mode == "movie":
        results = search.movies(query)
    else:
        results = search.tv_shows(query)
    if results and results["total_results"] > 0:
        await _cleanup_search_results(update.get_bot(), user)
        batch = []
        msgs = await send_back_text(update, f'Search results for "{query}":')
        batch.extend(m.message_id for m in msgs)
        items = results["results"]
        res = []
        for m in items:
            res.append(extract_movie_info(m, mode=mode))
        sorted_res = sort_by_rating(res)
        show_results = min(5, len(sorted_res))
        for _, poster_path, caption, mid in sorted_res[:show_results]:
            msg = await send_movie_message(update, caption, poster_path, mid, user, mode=mode)
            batch.append(msg.message_id)
        remaining = sorted_res[show_results:]
        if remaining:
            state._search_more[user] = (remaining, query)
            btn = InlineKeyboardButton(
                f"Show more ({len(remaining)} remaining)",
                callback_data="smore")
            msg = await update.message.reply_text(
                "More results available:",
                reply_markup=InlineKeyboardMarkup([[btn]]))
            batch.append(msg.message_id)
        state._search_results[user] = (update.message.chat_id, batch)
    else:
        await send_back_text(update, 'No results found.')


async def handle_smore(query, user, raw):
    if user not in state._search_more:
        await query.answer("No more results.", show_alert=True)
        return
    remaining, search_query = state._search_more[user]
    try:
        await query.message.delete()
    except Exception:
        pass
    if user in state._search_results:
        chat_id, msg_ids = state._search_results[user]
        if query.message.message_id in msg_ids:
            msg_ids.remove(query.message.message_id)
    next_batch = remaining[:5]
    new_remaining = remaining[5:]
    await query.answer()
    bot = query.get_bot()
    chat_id = query.message.chat_id
    batch = state._search_results[user][1] if user in state._search_results else []
    mode = state.user_data[user].get("mode", "movie")
    for _, poster_path, caption, mid in next_batch:
        keyboard = build_media_keyboard(mid, user, mode=mode)
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
        state._search_more[user] = (new_remaining, search_query)
        btn = InlineKeyboardButton(
            f"Show more ({len(new_remaining)} remaining)",
            callback_data="smore")
        msg = await bot.send_message(
            chat_id, "More results available:",
            reply_markup=InlineKeyboardMarkup([[btn]]))
        batch.append(msg.message_id)
    else:
        state._search_more.pop(user, None)
    state._search_results[user] = (chat_id, batch)


async def handle_det(query, user, raw):
    parts = raw.split(":", 2)
    if len(parts) < 3:
        await query.answer("Invalid action.", show_alert=True)
        return
    det_mt = parts[1]
    mid = int(parts[2])
    det_mode = _type_to_mode(det_mt)
    api = get_api(det_mode)
    details = api.details(mid)
    _, poster_path, desc, _ = extract_movie_info(details, mode=det_mode)
    action = parts[0]
    if action == "rdet":
        from tmdbot.keyboards import build_rating_keyboard
        rating_kb = build_rating_keyboard(
            mid, media_type=det_mt, action_prefix="rrate")
        rows = list(rating_kb.inline_keyboard) + [
            [InlineKeyboardButton("Change category", callback_data=f"ccat:{det_mt}:{mid}")]]
        keyboard = InlineKeyboardMarkup(rows)
    else:
        keyboard = build_media_keyboard(mid, user, mode=det_mode)
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


async def handle_exp_col(query, user, raw):
    action = raw.split(":", 1)[0]
    cid = int(raw.split(":", 1)[1])
    if cid not in state._chunk_movies:
        await query.answer("Session expired.", show_alert=True)
        return
    expanded = action == "exp"
    chunk_movies_list, det_action, chunk_mt = state._chunk_movies[cid]
    kb = build_chunk_keyboard(
        cid, chunk_movies_list, expanded=expanded, detail_action=det_action, media_type=chunk_mt)
    await query.edit_message_reply_markup(reply_markup=kb)
    await query.answer()


async def default_search_handler(update, context):
    from tmdbot.helpers import get_user_id, check_user_invalid
    user = get_user_id(update)
    if user not in state.user_data:
        return
    from tmdbot.config import settings
    if user not in settings['allowed_users']:
        return
    if not state.user_data.get(user, {}).get("onboarded", False):
        return
    text = update.message.text.strip()
    if text:
        await do_search(update, text, user)


def register(app, router):
    app.add_handler(CommandHandler(['search', 's'], SearchCommand()))
    router.add('smore', handle_smore)
    router.add('det', handle_det)
    router.add('rdet', handle_det)
    router.add('exp', handle_exp_col)
    router.add('col', handle_exp_col)
