import random
import concurrent.futures
import multiprocessing

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import CommandHandler

from tmdbot import state
from tmdbot.config import get_api
from tmdbot.base import BaseCommand
from tmdbot.helpers import (
    extract_movie_info, esc,
    is_in_any_watchlist, is_available_for_free,
    _parse_providers_from_details, _match_providers,
    create_available_at_str,
    _mode_to_type,
)
from tmdbot.keyboards import (
    get_main_keyboard, build_media_keyboard,
    build_genre_picker_keyboard,
)
from tmdbot.messaging import (
    send_back_text, send_movie_list,
    _with_progress_bar,
)


class RecommendCommand(BaseCommand):
    async def execute(self, update, context, user):
        mode = state.user_data[user].get("mode", "movie")
        watchlist = "normal"
        if context.args:
            watchlist = context.args[0]
        if watchlist not in state.user_data[user]["watchlists"][mode]:
            await send_back_text(update, f'Watchlist "{watchlist}" not found.')
            return
        state._rec_genre_filter[user] = {"watchlist": watchlist, "genres": set()}
        keyboard = build_genre_picker_keyboard(set(), mode=mode)
        await update.message.reply_text(
            "Filter recommendations by genre (or skip for all):",
            reply_markup=keyboard)


class CheckCommand(BaseCommand):
    async def execute(self, update, context, user):
        my_providers = state.user_data[user]["providers"]
        if not my_providers:
            await send_back_text(update, "Set up your streaming services first with /services.")
            return
        mode = state.user_data[user].get("mode", "movie")
        mt = _mode_to_type(mode)
        api = get_api(mode)

        def fetch_details(media_id):
            details = api.details(
                media_id, append_to_response="watch/providers")
            providers = _parse_providers_from_details(
                details, state.user_data[user]["region"])
            avail, matched = _match_providers(my_providers, providers)
            if avail:
                return (media_id, matched, details)
            return None

        bot = update.get_bot()
        chat_id = update.message.chat_id
        wls = state.user_data[user]["watchlists"][mode]
        total = sum(len(w) for w in wls.values())
        watchlists_snapshot = list(wls.items())

        def do_check(tick):
            num_threads = min(multiprocessing.cpu_count(), 8)
            available_items = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                for wn, w in watchlists_snapshot:
                    results = []
                    for r in executor.map(fetch_details, w):
                        if r:
                            results.append(r)
                        tick()
                    available_items.append((wn, results))
            return available_items

        available_items = await _with_progress_bar(
            bot, chat_id, "Checking streaming availability...", total, do_check)
        for wn, items in available_items:
            if items:
                movies_info = []
                for media_id, prov, details in items:
                    _, _, desc, mid = extract_movie_info(
                        details, skip_trailer=True, mode=mode)
                    title = details.get("title") or details.get(
                        "name") or "Unknown"
                    provider_str = create_available_at_str(prov)
                    movies_info.append((mid, title, desc + "\n" + provider_str))
                await send_movie_list(
                    update.get_bot(), update.message.chat_id,
                    f'Items on your {wn} watchlist available on streaming services:',
                    movies_info, media_type=mt)
            else:
                await send_back_text(update, f'None of the items on your {wn} watchlist are available on streaming services.')


class PopularCommand(BaseCommand):
    async def execute(self, update, context, user):
        if not state.user_data[user]["providers"]:
            await send_back_text(update, "Set up your streaming services first with /services.")
            return
        mode = state.user_data[user].get("mode", "movie")
        mt = _mode_to_type(mode)
        api = get_api(mode)
        label = "Popular movies" if mode == "movie" else "Popular TV shows"
        status_msg = await update.message.reply_text(
            f"Finding {label.lower()}...", reply_markup=get_main_keyboard(user))
        target_count = 10
        page = 1
        results = api.popular(page=page)
        total_pages = results["total_pages"]
        candidates = []
        max_candidates = target_count * 3
        while page < total_pages and len(candidates) < max_candidates:
            if page != 1:
                results = api.popular(page=page)
            for m in results["results"]:
                if m["id"] not in state.user_data[user]["watched"][mode]:
                    candidates.append(m)
            page += 1
        my_providers = state.user_data[user]["providers"]
        user_region = state.user_data[user]["region"]

        def check_popular_item(m):
            available, prov = is_available_for_free(
                my_providers, m["id"], user_region, mode=mode)
            if available:
                _, poster, desc, mid = extract_movie_info(
                    m, skip_trailer=True, mode=mode)
                title = m.get("title") or m.get("name") or "Unknown"
                return (mid, title, desc, prov)
            return None

        num_threads = min(multiprocessing.cpu_count(), 8)
        pop_items = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(check_popular_item, m) for m in candidates]
            for future in futures:
                if future.cancelled():
                    continue
                result = future.result()
                if result:
                    pop_items.append(result)
                    if len(pop_items) >= target_count:
                        for f in futures:
                            f.cancel()
                        break
        try:
            await status_msg.delete()
        except Exception:
            pass
        if pop_items:
            movies_info = []
            for mid, title, caption, prov in pop_items[:target_count]:
                provider_str = create_available_at_str(prov)
                movies_info.append((mid, title, caption + "\n" + provider_str))
            await send_movie_list(
                update.get_bot(), update.message.chat_id,
                f'{label} available on your streaming services:',
                movies_info, media_type=mt)
        else:
            await send_back_text(update, f'No {label.lower()} found on your streaming services.')


class PickCommand(BaseCommand):
    async def execute(self, update, context, user):
        if not state.user_data[user]["providers"]:
            await send_back_text(update, "Set up your streaming services first with /services.")
            return
        mode = state.user_data[user].get("mode", "movie")
        watchlist = context.args[0] if context.args else None
        if watchlist and watchlist not in state.user_data[user]["watchlists"][mode]:
            await send_back_text(update, f'Watchlist "{watchlist}" not found.')
            return
        candidates, label = _collect_pick_candidates(user, watchlist)
        if not candidates:
            await send_back_text(update, f'{label} is empty.' if watchlist else 'All your watchlists are empty.')
            return
        await _do_pick(update.get_bot(), update.message.chat_id, user,
                       candidates, label, watchlist or '*')


def _collect_pick_candidates(user, watchlist=None):
    """Collect IDs from one or all watchlists for current mode."""
    mode = state.user_data[user].get("mode", "movie")
    wls = state.user_data[user]["watchlists"][mode]
    if watchlist:
        wl = wls.get(watchlist, [])
        return list(wl), f'"{watchlist}"'
    all_items = []
    for wl in wls.values():
        all_items.extend(wl)
    return list(set(all_items)), "your watchlists"


async def _do_pick(bot, chat_id, user, candidates, label, wl_cb_name):
    """Pick a random available item from candidates."""
    mode = state.user_data[user].get("mode", "movie")
    mt = _mode_to_type(mode)
    api = get_api(mode)
    my_providers = state.user_data[user]["providers"]
    user_region = state.user_data[user]["region"]
    random.shuffle(candidates)

    picked_mid = None
    picked_details = None
    matched_providers = None
    for mid in candidates:
        details = api.details(mid, append_to_response="watch/providers")
        providers = _parse_providers_from_details(details, user_region)
        avail, matched = _match_providers(my_providers, providers)
        if avail:
            picked_mid = mid
            picked_details = details
            matched_providers = matched
            break

    if picked_mid is None:
        await bot.send_message(
            chat_id, esc(f'Nothing in {label} is available on your services.'),
            parse_mode=ParseMode.MARKDOWN_V2, reply_markup=get_main_keyboard(user))
        return False

    _, poster_path, desc, _ = extract_movie_info(picked_details, mode=mode)
    desc += "\n" + create_available_at_str(matched_providers)
    keyboard = build_media_keyboard(picked_mid, user, mode=mode)
    pick_cb = f"rpick:{wl_cb_name}"
    if len(pick_cb.encode('utf-8')) <= 64:
        rows = list(keyboard.inline_keyboard) + [
            [InlineKeyboardButton("Pick another", callback_data=pick_cb)]]
        keyboard = InlineKeyboardMarkup(rows)
    escaped = esc(desc)
    if poster_path:
        await bot.send_photo(
            chat_id, poster_path,
            caption=escaped,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard)
    else:
        await bot.send_message(
            chat_id, escaped,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=keyboard)
    return True


async def _do_recommend(bot, chat_id, user, watchlist, genre_filter=None):
    """Core recommendation logic."""
    mode = state.user_data[user].get("mode", "movie")
    mt = _mode_to_type(mode)
    api = get_api(mode)
    sources = [(mid, 1.0)
               for mid in state.user_data[user]["watchlists"][mode][watchlist]]
    rated_watched = sorted(
        [(mid, r) for mid, r in state.user_data[user]["watched"][mode].items()
         if r is not None and r >= 7],
        key=lambda x: -x[1])[:20]
    sources.extend((mid, rating / 10.0) for mid, rating in rated_watched)

    if not sources:
        await bot.send_message(
            chat_id,
            esc(f'Your "{watchlist}" watchlist is empty and you have no highly-rated watched items.'),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=get_main_keyboard(user))
        return

    total = len(sources)

    def query_recommendations(source):
        source_id, weight = source
        available_recommendations = []
        results = api.recommendations(source_id)
        items = results["results"]
        for m in items:
            if genre_filter:
                item_genres = set(m.get("genre_ids", []))
                if not item_genres.intersection(genre_filter):
                    continue
            in_watchlist = is_in_any_watchlist(m["id"], user, mode=mode)
            if not in_watchlist and m["id"] not in state.user_data[user]["watched"][mode]:
                available, prov = is_available_for_free(
                    state.user_data[user]["providers"], m["id"], state.user_data[user]["region"], mode=mode)
                if available:
                    popularity, poster, desc, mid = extract_movie_info(
                        m, skip_trailer=True, mode=mode)
                    title = m.get("title") or m.get("name") or "Unknown"
                    available_recommendations.append(
                        (popularity, poster, desc, prov, mid, title, weight))
        return available_recommendations

    def do_recommend(tick):
        num_threads = min(multiprocessing.cpu_count(), 8)
        all_recs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for result in executor.map(query_recommendations, sources):
                all_recs.extend(result)
                tick()
        return all_recs

    available_recommendations = await _with_progress_bar(
        bot, chat_id, "Finding recommendations...", total, do_recommend)

    id_scores = {}
    for item in available_recommendations:
        mid = item[4]
        id_scores[mid] = id_scores.get(mid, 0) + item[6]

    def custom_sort_key(item):
        return (-id_scores[item[4]], -item[0])

    seen_ids = set()
    unique_tuples = []
    for tuple_item in available_recommendations:
        if tuple_item[4] not in seen_ids:
            seen_ids.add(tuple_item[4])
            unique_tuples.append(tuple_item)

    available_recommendations = sorted(unique_tuples, key=custom_sort_key)

    num_rec = min(50, len(available_recommendations))
    if available_recommendations:
        movies_info = []
        for _, poster_path, caption, prov, mid, title, _ in available_recommendations[:num_rec]:
            provider_str = create_available_at_str(prov)
            movies_info.append((mid, title, caption + "\n" + provider_str))
        label = "Recommended based on" if mode == "movie" else "Recommended shows based on"
        await send_movie_list(
            bot, chat_id,
            f'{label} your "{watchlist}" watchlist:',
            movies_info, media_type=mt)
    else:
        await bot.send_message(
            chat_id,
            esc(
                f'No recommendations found based on your "{watchlist}" watchlist.'),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=get_main_keyboard(user))


# Callback handlers

async def handle_gf(query, user, raw):
    genre_id = int(raw.split(":", 1)[1])
    if user not in state._rec_genre_filter:
        await query.answer("Session expired.", show_alert=True)
        return
    genres = state._rec_genre_filter[user]["genres"]
    if genre_id in genres:
        genres.discard(genre_id)
    else:
        genres.add(genre_id)
    mode = state.user_data[user].get("mode", "movie")
    keyboard = build_genre_picker_keyboard(genres, mode=mode)
    await query.edit_message_reply_markup(reply_markup=keyboard)
    await query.answer()


async def handle_recgo(query, user, raw):
    if user not in state._rec_genre_filter:
        await query.answer("Session expired.", show_alert=True)
        return
    s = state._rec_genre_filter.pop(user)
    recgo_mode = raw.split(":", 1)[1]
    genre_filter = s["genres"] if recgo_mode == "filter" and s["genres"] else None
    await query.answer()
    try:
        await query.message.delete()
    except Exception:
        pass
    await _do_recommend(
        query.get_bot(), query.message.chat_id,
        user, s["watchlist"], genre_filter=genre_filter)


async def handle_rpick(query, user, raw):
    wl_name = raw.split(":", 1)[1]
    watchlist = None if wl_name == "*" else wl_name
    candidates, label = _collect_pick_candidates(user, watchlist)
    if not candidates:
        await query.answer("Watchlists are empty.", show_alert=True)
        return
    await query.answer()
    await _do_pick(query.get_bot(), query.message.chat_id, user,
                   candidates, label, wl_name)


def register(app, router):
    app.add_handler(CommandHandler(['recommend', 'r'], RecommendCommand()))
    app.add_handler(CommandHandler(['check', 'c'], CheckCommand()))
    app.add_handler(CommandHandler(['popular', 'pop'], PopularCommand()))
    app.add_handler(CommandHandler(['pick', 'p'], PickCommand()))
    router.add('gf', handle_gf)
    router.add('recgo', handle_recgo)
    router.add('rpick', handle_rpick)
