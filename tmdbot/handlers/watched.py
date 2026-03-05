from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import CommandHandler

from tmdbot import state
from tmdbot.config import get_api, tv
from tmdbot.base import BaseCommand
from tmdbot.helpers import (
    esc, extract_movie_info, is_valid_media_id,
    is_in_any_watchlist, find_all_watchlists,
    find_all_shared_watchlists,
    _mode_to_type, _type_to_mode,
    _get_shared_wl, _get_user_display_name,
    _count_released_seasons,
)
from tmdbot.keyboards import (
    get_main_keyboard, build_media_keyboard,
    build_rating_keyboard, build_season_picker_keyboard,
)
from tmdbot.messaging import (
    send_back_text, send_movie_list,
    _cleanup_search_results, _cleanup_rate_list,
    _is_search_message, _notify_shared_wl_members,
)


class WatchedCommand(BaseCommand):
    async def execute(self, update, context, user):
        if not context.args:
            await send_back_text(update, 'Please provide the ID.')
            return
        mode = state.user_data[user].get("mode", "movie")
        mt = _mode_to_type(mode)
        media_id = context.args[0]
        err_msg = is_valid_media_id(media_id, mode)
        if err_msg:
            await send_back_text(update, f'The provided ID is invalid: ' + err_msg)
            return
        media_id = int(media_id)
        if mode == "tv":
            try:
                details = tv.details(media_id)
                num_seasons = _count_released_seasons(details)
            except Exception:
                num_seasons = 1
            state._pending_season[user] = {"mid": media_id,
                                           "total": num_seasons, "media_type": mt}
            keyboard = build_season_picker_keyboard(media_id, mt, num_seasons)
            await update.message.reply_text(
                "Which season did you watch up to?",
                reply_markup=keyboard)
        else:
            if media_id in state.user_data[user]["watched"][mode]:
                await send_back_text(update, 'Already marked as watched.')
                return
            keyboard = build_rating_keyboard(media_id, mt)
            await update.message.reply_text(
                "Rate this (1\u201310):",
                reply_markup=keyboard)


class RateCommand(BaseCommand):
    async def execute(self, update, context, user):
        mode = state.user_data[user].get("mode", "movie")
        if not state.user_data[user]["watched"][mode]:
            await send_back_text(update, "You haven't watched anything in this mode yet!")
            return
        await _cleanup_rate_list(update.get_bot(), user)
        msgs = await _send_rate_list(
            update.get_bot(), update.message.chat_id, user)
        state._rate_list_messages[user] = (update.message.chat_id, [
            m.message_id for m in msgs])


async def _send_rate_list(bot, chat_id, user):
    """Build and send the rate list for a user. Returns sent messages."""
    mode = state.user_data[user].get("mode", "movie")
    mt = _mode_to_type(mode)
    api = get_api(mode)
    watched = state.user_data[user]["watched"][mode]
    unrated = [(mid, r) for mid, r in watched.items() if r is None]
    rated = sorted(
        [(mid, r) for mid, r in watched.items() if r is not None],
        key=lambda x: -x[1])
    movies_info = []
    for mid, r in unrated + rated:
        details = api.details(mid)
        _, _, desc, _ = extract_movie_info(
            details, skip_trailer=True, mode=mode)
        title = details.get("title") or details.get("name") or "Unknown"
        rating_str = "unrated" if r is None else f"Your rating: {r}/10"
        movies_info.append((mid, title, f"{desc}\n{rating_str}"))
    n_unrated = len(unrated)
    n_rated = len(rated)
    label = "movie" if mode == "movie" else "show"
    if n_unrated > 0:
        header = f'{n_unrated} unrated, {n_rated} rated {label}(s):'
    else:
        header = f'{n_rated} watched {label}(s):'
    return await send_movie_list(
        bot, chat_id, header, movies_info, detail_action="rdet", media_type=mt)


# Callback handlers

async def handle_rate(query, user, raw):
    """Handle rate and rrate actions."""
    parts = raw.split(":")
    action = parts[0]
    try:
        mt = parts[1]
        mid = int(parts[2])
        rating = int(parts[3]) or None
    except (ValueError, IndexError):
        await query.answer("Invalid action.", show_alert=True)
        return
    rate_mode = _type_to_mode(mt)
    # Save undo state
    prev_wls = find_all_watchlists(mid, user, mode=rate_mode)
    prev_rating = state.user_data[user]["watched"][rate_mode].get(mid, "absent")
    prev_season_data = state.user_data[user]["tv_season_counts"].get(
        mid, "absent") if rate_mode == "tv" else None
    state._last_watched[user] = {"mid": mid, "watchlists": prev_wls,
                                 "prev_rating": prev_rating, "prev_season_data": prev_season_data, "mode": rate_mode}
    for wn in prev_wls:
        state.user_data[user]["watchlists"][rate_mode][wn].remove(mid)
    state.user_data[user]["watched"][rate_mode][mid] = rating
    # Save season tracking data for TV shows
    if rate_mode == "tv" and user in state._pending_season and state._pending_season[user]["mid"] == mid:
        pending = state._pending_season.pop(user)
        state.user_data[user]["tv_season_counts"][mid] = {
            "total": pending["total"],
            "watched": pending.get("season", pending["total"]),
        }
    state.save_user_data()
    if rating:
        await query.answer(f"Rated {rating}/10 and marked as watched.")
    else:
        await query.answer("Marked as watched.")
    chat_id = query.message.chat_id
    bot = query.get_bot()
    undo_kb = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Undo", callback_data="undo")]])
    if _is_search_message(user, query.message.message_id):
        await _cleanup_search_results(bot, user)
        await bot.send_message(
            chat_id, "Marked as watched.",
            reply_markup=get_main_keyboard(user))
        await bot.send_message(
            chat_id, "Undo?",
            reply_markup=undo_kb)
    else:
        try:
            await query.message.delete()
        except Exception:
            await query.edit_message_reply_markup(reply_markup=None)
        await bot.send_message(
            chat_id, "Marked as watched.",
            reply_markup=get_main_keyboard(user))
        await bot.send_message(
            chat_id, "Undo?",
            reply_markup=undo_kb)
    # Refresh rate list only if rating came from /rate flow
    if action == "rrate" and user in state._rate_list_messages:
        await _cleanup_rate_list(bot, user)
        msgs = await _send_rate_list(bot, chat_id, user)
        state._rate_list_messages[user] = (chat_id, [m.message_id for m in msgs])
    # Notify shared watchlist members about watched
    shared_wls = find_all_shared_watchlists(mid, user, mode=rate_mode)
    if shared_wls:
        try:
            api_tmp = get_api(rate_mode)
            details_tmp = api_tmp.details(mid)
            watch_title = details_tmp.get(
                "title") or details_tmp.get("name") or str(mid)
        except Exception:
            watch_title = str(mid)
        for sw_id, sw_name in shared_wls:
            sw = _get_shared_wl(sw_id)
            if sw:
                rating_text = f" ({rating}/10)" if rating else ""
                await _notify_shared_wl_members(
                    bot, sw, user,
                    f'{_get_user_display_name(user)} watched "{watch_title}"{rating_text} from "{sw_name}".')


async def handle_undo(query, user, raw):
    if user not in state._last_watched:
        await query.answer("Nothing to undo.", show_alert=True)
        return
    s = state._last_watched.pop(user)
    mid = s["mid"]
    undo_mode = s.get("mode", "movie")
    prev_wls = s.get("watchlists", [])
    if not prev_wls and s.get("watchlist"):
        prev_wls = [s["watchlist"]]
    prev_rating = s["prev_rating"]
    if prev_rating == "absent":
        state.user_data[user]["watched"][undo_mode].pop(mid, None)
    else:
        state.user_data[user]["watched"][undo_mode][mid] = prev_rating
    for wn in prev_wls:
        if wn in state.user_data[user]["watchlists"][undo_mode]:
            state.user_data[user]["watchlists"][undo_mode][wn].append(mid)
    prev_season_data = s.get("prev_season_data")
    if undo_mode == "tv" and prev_season_data is not None:
        if prev_season_data == "absent":
            state.user_data[user]["tv_season_counts"].pop(mid, None)
        else:
            state.user_data[user]["tv_season_counts"][mid] = prev_season_data
    state.save_user_data()
    await query.answer("Undone!")
    try:
        await query.message.delete()
    except Exception:
        await query.edit_message_reply_markup(reply_markup=None)
    await query.get_bot().send_message(
        query.message.chat_id,
        "Last watched action undone.",
        reply_markup=get_main_keyboard(user))
    if user in state._rate_list_messages:
        bot = query.get_bot()
        chat_id = query.message.chat_id
        await _cleanup_rate_list(bot, user)
        msgs = await _send_rate_list(bot, chat_id, user)
        state._rate_list_messages[user] = (chat_id, [m.message_id for m in msgs])


async def handle_w_action(query, user, movie_id, media_type, cb_mode):
    """Handle 'w' (watched) action from fallback dispatcher."""
    if media_type == "tv":
        try:
            details = tv.details(movie_id)
            num_seasons = _count_released_seasons(details)
        except Exception:
            num_seasons = 1
        state._pending_season[user] = {
            "mid": movie_id, "total": num_seasons, "media_type": media_type}
        season_kb = build_season_picker_keyboard(
            movie_id, media_type, num_seasons)
        await query.edit_message_reply_markup(reply_markup=season_kb)
        await query.answer("Which season did you watch up to?")
    else:
        if movie_id in state.user_data[user]["watched"][cb_mode] and state.user_data[user]["watched"][cb_mode][movie_id] is not None:
            await query.answer("Already marked as watched.", show_alert=True)
        else:
            rating_kb = build_rating_keyboard(
                movie_id, media_type=media_type)
            await query.edit_message_reply_markup(reply_markup=rating_kb)
            await query.answer("Rate this:")


async def handle_ws_action(query, user, raw, movie_id, media_type):
    """Handle 'ws' (season picked) action from fallback dispatcher."""
    parts = raw.split(":")
    try:
        season_num = int(parts[3])
    except (ValueError, IndexError):
        await query.answer("Invalid action.", show_alert=True)
        return
    if user in state._pending_season:
        state._pending_season[user]["season"] = season_num
    else:
        state._pending_season[user] = {
            "mid": movie_id, "total": season_num, "season": season_num, "media_type": media_type}
    rating_kb = build_rating_keyboard(movie_id, media_type=media_type)
    await query.edit_message_reply_markup(reply_markup=rating_kb)
    await query.answer("Rate this:")


def register(app, router):
    app.add_handler(CommandHandler(['watched', 'w'], WatchedCommand()))
    app.add_handler(CommandHandler('rate', RateCommand()))
    router.add('rate', handle_rate)
    router.add('rrate', handle_rate)
    router.add('undo', handle_undo)
