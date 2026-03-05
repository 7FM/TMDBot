from collections import Counter

from telegram import ForceReply
from telegram.ext import CommandHandler

from tmdbot import state
from tmdbot.config import get_api, trending, person_api, search
from tmdbot.base import BaseCommand
from tmdbot.helpers import (
    extract_movie_info, extract_genre,
    _mode_to_type,
)
from tmdbot.messaging import (
    send_back_text, send_movie_list,
    _with_progress_bar,
)


class StatsCommand(BaseCommand):
    async def execute(self, update, context, user):
        mode = state.user_data[user].get("mode", "movie")
        watched = state.user_data[user]["watched"][mode]
        total = len(watched)
        if total == 0:
            label = "movies" if mode == "movie" else "TV shows"
            await send_back_text(update, f"You haven't watched any {label} yet.")
            return
        ratings = []
        for mid, rating in watched.items():
            if isinstance(rating, (int, float)) and rating > 0:
                ratings.append(rating)
        rated = len(ratings)
        unrated = total - rated
        avg = sum(ratings) / len(ratings) if ratings else 0
        buckets = {"9-10": 0, "7-8": 0, "5-6": 0, "3-4": 0, "1-2": 0}
        for r in ratings:
            if r >= 9:
                buckets["9-10"] += 1
            elif r >= 7:
                buckets["7-8"] += 1
            elif r >= 5:
                buckets["5-6"] += 1
            elif r >= 3:
                buckets["3-4"] += 1
            else:
                buckets["1-2"] += 1
        max_count = max(buckets.values()) if buckets.values() else 1
        bar_width = 15
        label = "movie" if mode == "movie" else "TV show"
        text = f"Your {label} stats:\n"
        text += f"Total watched: {total}\n"
        text += f"Rated: {rated} | Unrated: {unrated}\n"
        if ratings:
            text += f"Average rating: {avg:.1f}/10\n"
        if ratings:
            text += "\nRating distribution:\n"
            for bucket, count in buckets.items():
                bars = round(bar_width * count / max_count) if max_count > 0 else 0
                text += f"`{bucket:>4}: {'█' * bars}{'░' * (bar_width - bars)} {count}`\n"
        api = get_api(mode)
        mids = list(watched.keys())

        def fetch_genres(tick):
            genre_counter = Counter()
            for mid in mids:
                try:
                    details = api.details(mid)
                    genres = extract_genre(details, mode=mode)
                    for g in genres:
                        genre_counter[g] += 1
                except Exception:
                    pass
                tick()
            return genre_counter

        genre_counter = await _with_progress_bar(
            update.get_bot(), update.message.chat_id,
            "Fetching stats...", len(mids), fetch_genres)
        if genre_counter:
            text += "\nTop genres:\n"
            for i, (genre_name, count) in enumerate(genre_counter.most_common(10), 1):
                text += f"{i}. {genre_name} ({count})\n"
        await send_back_text(update, text)


class TrendingCommand(BaseCommand):
    async def execute(self, update, context, user):
        mode = state.user_data[user].get("mode", "movie")
        mt = _mode_to_type(mode)
        if mode == "movie":
            results = trending.movie_day()
            label = "Trending movies"
        else:
            results = trending.tv_day()
            label = "Trending TV shows"
        items = []
        for m in results:
            if m["id"] not in state.user_data[user]["watched"][mode]:
                _, poster, desc, mid = extract_movie_info(
                    m, skip_trailer=True, mode=mode)
                title = m.get("title") or m.get("name") or "Unknown"
                items.append((mid, title, desc))
            if len(items) >= 10:
                break
        if items:
            await send_movie_list(
                update.get_bot(), update.message.chat_id,
                f"{label} today:", items, media_type=mt)
        else:
            await send_back_text(update, f"No {label.lower()} found.")


class PersonCommand(BaseCommand):
    async def execute(self, update, context, user):
        if not context.args:
            state._pending_person[user] = True
            await update.message.reply_text(
                "Enter a person's name to search:",
                reply_markup=ForceReply(selective=True))
            return
        query = ' '.join(context.args)
        await do_person_search(update, query, user)


async def do_person_search(update, query, user):
    results = search.people(query)
    if not results or results["total_results"] == 0:
        await send_back_text(update, f'No person found for "{query}".')
        return
    person = results["results"][0]
    person_id = person["id"]
    person_name = person.get("name", "Unknown")
    mode = state.user_data[user].get("mode", "movie")
    mt = _mode_to_type(mode)
    try:
        credits = person_api.combined_credits(person_id)
    except Exception:
        await send_back_text(update, f'Could not fetch credits for "{person_name}".')
        return
    cast_list = []
    try:
        for c in credits.get("cast", []):
            media_type = c.get("media_type", "movie")
            if (mode == "movie" and media_type == "movie") or \
               (mode == "tv" and media_type == "tv"):
                vote = c.get("vote_average", 0) or 0
                vote_count = c.get("vote_count", 0) or 0
                if vote_count > 0:
                    _, poster, desc, mid = extract_movie_info(
                        c, skip_trailer=True, mode=mode)
                    title = c.get("title") or c.get("name") or "Unknown"
                    cast_list.append((vote, mid, title, desc))
    except (KeyError, TypeError, AttributeError):
        pass
    cast_list.sort(key=lambda x: x[0], reverse=True)
    cast_list = cast_list[:20]
    if cast_list:
        movies_info = [(mid, title, desc)
                       for _, mid, title, desc in cast_list]
        label = "movies" if mode == "movie" else "TV shows"
        await send_movie_list(
            update.get_bot(), update.message.chat_id,
            f"{person_name} - {label}:", movies_info, media_type=mt)
    else:
        label = "movies" if mode == "movie" else "TV shows"
        await send_back_text(
            update, f'No {label} found for "{person_name}".')


def register(app, router):
    app.add_handler(CommandHandler('stats', StatsCommand()))
    app.add_handler(CommandHandler(['trending', 'tr'], TrendingCommand()))
    app.add_handler(CommandHandler(['person', 'ps'], PersonCommand()))
