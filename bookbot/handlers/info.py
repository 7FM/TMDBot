import logging

from telegram import ForceReply
from telegram.ext import CommandHandler

from botlib import state
from botlib.base import BaseCommand
from botlib.helpers import get_watched_rating, _mode_to_type
from botlib.messaging import send_back_text, send_movie_list, _with_progress_bar
from bookbot.config import ol_search_authors, ol_author_works, ol_work
from bookbot.helpers import extract_book_info, sort_by_rating, work_key_to_id

logger = logging.getLogger(__name__)


class StatsCommand(BaseCommand):
    async def execute(self, update, context, user):
        mode = "book"
        watched = state.user_data[user].get("watched", {}).get(mode, {})
        if not watched:
            await send_back_text(update, "You haven't read any books yet.")
            return

        total = len(watched)
        rated = sum(1 for e in watched.values()
                    if get_watched_rating(e) is not None)
        unrated = total - rated
        ratings = [get_watched_rating(e) for e in watched.values()
                   if get_watched_rating(e) is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0

        # Rating distribution
        dist = [0] * 10
        for r in ratings:
            dist[r - 1] += 1
        max_count = max(dist) if dist else 1
        bar_width = 8

        lines = [f"Books read: {total}",
                 f"Rated: {rated}, Unrated: {unrated}",
                 f"Average rating: {round(avg_rating, 1)}/10",
                 "",
                 "Rating distribution:"]
        for i, count in enumerate(dist):
            bar_len = round(bar_width * count / max_count) if max_count else 0
            bar = "\u2588" * bar_len + "\u2591" * (bar_width - bar_len)
            lines.append(f"{i+1:2d} [{bar}] {count}")

        # Top subjects (requires API calls)
        bot = update.get_bot()
        chat_id = update.message.chat_id

        def fetch_subjects(tick):
            subject_counts = {}
            for mid in watched:
                try:
                    data = ol_work(mid)
                    for s in data.get("subjects", [])[:3]:
                        if len(s) < 30 and s.lower() not in ("fiction", "general"):
                            subject_counts[s] = subject_counts.get(s, 0) + 1
                except Exception:
                    pass
                tick()
            return subject_counts

        subject_counts = await _with_progress_bar(
            bot, chat_id, "Loading subjects...", len(watched), fetch_subjects)

        if subject_counts:
            top = sorted(subject_counts.items(),
                         key=lambda x: x[1], reverse=True)[:10]
            lines.append("")
            lines.append("Top subjects:")
            for name, count in top:
                lines.append(f"  {name}: {count}")

        await send_back_text(update, "\n".join(lines))


class AuthorCommand(BaseCommand):
    async def execute(self, update, context, user):
        query = " ".join(context.args) if context.args else None
        if not query:
            state._pending_person[user] = True
            await update.message.reply_text(
                "Which author are you looking for?",
                reply_markup=ForceReply(selective=True))
            return
        await do_author_search(update, query, user)


async def do_author_search(update, query, user):
    mode = "book"
    mt = _mode_to_type(mode)
    try:
        authors = ol_search_authors(query, limit=5)
    except Exception as e:
        logger.error("Author search failed: %s", e)
        await send_back_text(update, "Author search failed.")
        return

    if not authors:
        await send_back_text(update, f'No authors found for "{query}".')
        return

    # Take the top author match
    author = authors[0]
    author_name = author.get("name", "Unknown")
    author_key = author.get("key", "")

    try:
        works = ol_author_works(author_key, limit=50)
    except Exception as e:
        logger.error("Author works fetch failed: %s", e)
        await send_back_text(update, f"Failed to fetch works by {author_name}.")
        return

    if not works:
        await send_back_text(update, f"No works found by {author_name}.")
        return

    watched = state.user_data[user].get("watched", {}).get(mode, {})
    infos = []
    for w in works:
        key = w.get("key", "")
        wid = work_key_to_id(key)
        if wid is None:
            continue
        if wid in watched:
            continue
        title = w.get("title", "Unknown")
        covers = w.get("covers", [])
        cover_id = covers[0] if covers else None
        subjects = w.get("subjects", [])[:3] if "subjects" in w else []
        from bookbot.helpers import get_cover_url
        desc = f'`{title}` - {author_name}'
        if subjects:
            clean = [s for s in subjects if len(s) < 30][:3]
            if clean:
                desc += f' - {", ".join(clean)}'
        infos.append((wid, title, desc))

    if not infos:
        await send_back_text(update, f"No new works by {author_name} (all already read).")
        return

    infos = infos[:20]
    await send_movie_list(
        update.get_bot(), update.message.chat_id,
        f"Works by {author_name}:", infos,
        detail_action="det", media_type=mt)


def register(app, router):
    app.add_handler(CommandHandler('stats', StatsCommand()))
    app.add_handler(CommandHandler(['author', 'a'], AuthorCommand()))
