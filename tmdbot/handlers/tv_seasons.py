import asyncio
import concurrent.futures
import multiprocessing
import datetime

from telegram.constants import ParseMode
from telegram.ext import CommandHandler

from tmdbot import state
from tmdbot.config import settings, tv
from tmdbot.base import BaseCommand
from tmdbot.helpers import (
    esc, extract_movie_info,
    get_watched_rating,
    _count_released_seasons,
)
from tmdbot.keyboards import build_season_picker_keyboard
from tmdbot.messaging import (
    send_back_text, send_movie_list,
    _with_progress_bar,
)


class NewSeasonsCommand(BaseCommand):
    async def execute(self, update, context, user):
        watched_tv = state.user_data[user].get("watched", {}).get("tv", {})
        if not watched_tv:
            await send_back_text(update, "You have no watched TV shows.")
            return
        bot = update.get_bot()
        chat_id = update.message.chat_id
        total = len(watched_tv)

        def do_check(tick):
            return _check_new_seasons_for_user(user, tick)

        new_list, newly_recorded = await _with_progress_bar(
            bot, chat_id, "Checking for new seasons\u2026", total, do_check)
        state.save_user_data()
        parts = []
        if new_list:
            movies_info = []
            for mid, title, old_total, new_total, watched_season, details in new_list:
                _, _, desc, _ = extract_movie_info(
                    details, skip_trailer=True, mode="tv")
                diff = new_total - old_total
                season_word = "season" if diff == 1 else "seasons"
                extra = f"\u2728 {diff} new {season_word} (you watched S{watched_season}, now has {new_total} seasons)"
                movies_info.append((mid, title, desc + "\n" + extra))
            await send_movie_list(bot, chat_id,
                                  f"New seasons available for {len(new_list)} show(s):",
                                  movies_info, media_type="tv")
        else:
            parts.append("No new seasons detected for your watched shows.")
        if newly_recorded:
            parts.append(f"Recorded season data for {newly_recorded} new show(s).")
        if parts:
            await send_back_text(update, " ".join(parts))


class ViewSeasonsCommand(BaseCommand):
    async def execute(self, update, context, user):
        watched_tv = state.user_data[user].get("watched", {}).get("tv", {})
        if not watched_tv:
            await send_back_text(update, "You have no watched TV shows.")
            return
        stored = state.user_data[user].get("tv_season_counts", {})
        movies_info = []
        for mid, entry in watched_tv.items():
            try:
                details = tv.details(mid)
            except Exception:
                continue
            title = details.get("name") or details.get("title") or "Unknown"
            _, _, desc, _ = extract_movie_info(
                details, skip_trailer=True, mode="tv")
            season_data = stored.get(mid)
            if season_data:
                watched_s = season_data.get("watched", "?")
                total_s = season_data.get("total", "?")
                season_str = f"Watched: S{watched_s}/{total_s}"
            else:
                total_s = _count_released_seasons(details) or "?"
                season_str = f"Watched: ?/{total_s}"
            rating = get_watched_rating(entry)
            rating_str = f"{rating}/10" if rating else "unrated"
            movies_info.append(
                (mid, title, f"{desc}\n{season_str} - {rating_str}"))
        await send_movie_list(
            update.get_bot(), update.message.chat_id,
            f"Season tracking for {len(movies_info)} TV show(s):",
            movies_info, detail_action="sdet", media_type="tv")


def _check_new_seasons_for_user(user, tick=None):
    """Check watched TV shows for new seasons."""
    watched_tv = state.user_data[user].get("watched", {}).get("tv", {})
    stored = state.user_data[user].get("tv_season_counts", {})
    new_list = []
    newly_recorded = 0
    num_threads = min(multiprocessing.cpu_count(), 8)

    def fetch_show(mid):
        try:
            details = tv.details(mid)
            if tick:
                tick()
            return (mid, details)
        except Exception:
            if tick:
                tick()
            return (mid, None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for mid, details in executor.map(fetch_show, list(watched_tv.keys())):
            if details is None:
                continue
            title = details.get("name") or details.get("title") or "Unknown"
            current_total = _count_released_seasons(details)
            if mid in stored:
                old_total = stored[mid].get("total", 0)
                watched_season = stored[mid].get("watched", old_total)
                if current_total > old_total:
                    new_list.append(
                        (mid, title, old_total, current_total, watched_season, details))
                stored[mid]["total"] = current_total
            else:
                stored[mid] = {"total": current_total,
                               "watched": current_total}
                newly_recorded += 1
    state.user_data[user]["tv_season_counts"] = stored
    return new_list, newly_recorded


async def _daily_season_check(context):
    """Daily job: check all users' watched TV shows for new seasons."""
    for user in settings["allowed_users"]:
        watched_tv = state.user_data[user].get("watched", {}).get("tv", {})
        if not watched_tv:
            continue
        new_list, _ = await asyncio.to_thread(
            _check_new_seasons_for_user, user)
        if not new_list:
            continue
        state.save_user_data()
        movies_info = []
        for mid, title, old_total, new_total, watched_season, details in new_list:
            _, _, desc, _ = extract_movie_info(
                details, skip_trailer=True, mode="tv")
            diff = new_total - old_total
            season_word = "season" if diff == 1 else "seasons"
            extra = f"\u2728 {diff} new {season_word} (you watched S{watched_season}, now has {new_total} seasons)"
            movies_info.append((mid, title, desc + "\n" + extra))
        await send_movie_list(context.bot, user,
                              f"New seasons available for {len(new_list)} show(s):",
                              movies_info, media_type="tv")


# Callback handlers

async def handle_sdet(query, user, raw):
    parts = raw.split(":", 2)
    if len(parts) < 3:
        await query.answer("Invalid action.", show_alert=True)
        return
    mid = int(parts[2])
    try:
        details = tv.details(mid)
        num_seasons = _count_released_seasons(details)
    except Exception:
        num_seasons = 1
    state._pending_season[user] = {"mid": mid,
                                   "total": num_seasons, "media_type": "tv"}
    season_kb = build_season_picker_keyboard(
        mid, "tv", num_seasons, action_prefix="supd")
    await query.answer("Update watched season:")
    bot = query.get_bot()
    chat_id = query.message.chat_id
    stored = state.user_data[user].get("tv_season_counts", {}).get(mid)
    current_s = stored.get("watched", "?") if stored else "?"
    await bot.send_message(
        chat_id,
        esc(
            f"Update watched season for this show (currently S{current_s}/{num_seasons}):"),
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=season_kb)


async def handle_supd(query, user, raw):
    parts = raw.split(":")
    try:
        mid = int(parts[2])
        season_num = int(parts[3])
    except (ValueError, IndexError):
        await query.answer("Invalid action.", show_alert=True)
        return
    stored = state.user_data[user].get("tv_season_counts", {})
    if mid in stored:
        stored[mid]["watched"] = season_num
    else:
        total = season_num
        if user in state._pending_season and state._pending_season[user]["mid"] == mid:
            total = state._pending_season[user]["total"]
        stored[mid] = {"total": total, "watched": season_num}
    state._pending_season.pop(user, None)
    state.save_user_data()
    await query.answer(f"Updated to season {season_num}.")
    try:
        await query.message.delete()
    except Exception:
        await query.edit_message_reply_markup(reply_markup=None)


def register(app, router):
    app.add_handler(CommandHandler(['newseasons', 'ns'], NewSeasonsCommand()))
    app.add_handler(CommandHandler(['seasons', 'ss'], ViewSeasonsCommand()))
    router.add('sdet', handle_sdet)
    router.add('supd', handle_supd)
