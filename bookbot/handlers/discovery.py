import logging
import random

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler

from botlib import state
from botlib.base import BaseCommand
from botlib.helpers import (
    esc, get_watched_rating, get_watched_category,
    is_in_any_watchlist, _mode_to_type,
)
from botlib.keyboards import build_media_keyboard, build_recommend_category_keyboard
from botlib.messaging import send_back_text, send_movie_list
from bookbot.keyboards import get_main_keyboard
from bookbot.helpers import extract_book_info, sort_by_rating
from bookbot.config import ol_work, ol_subject, ol_trending

logger = logging.getLogger(__name__)


class RecommendCommand(BaseCommand):
    async def execute(self, update, context, user):
        mode = "book"
        if not context.args:
            keyboard = build_recommend_category_keyboard(user, mode=mode)
            await update.message.reply_text(
                "Recommend based on which list?", reply_markup=keyboard)
            return
        watchlist = " ".join(context.args)
        await _do_recommend(update, user, watchlist)


async def _do_recommend(update, user, watchlist):
    mode = "book"
    mt = _mode_to_type(mode)
    watched = state.user_data[user].get("watched", {}).get(mode, {})
    bot = update.get_bot() if hasattr(update, 'get_bot') else update.message.get_bot()
    chat_id = update.message.chat_id if hasattr(
        update, 'message') and update.message else update.chat_id

    # Collect subjects from highly-rated watched books matching category
    seed_subjects = {}
    for mid, entry in watched.items():
        r = get_watched_rating(entry)
        if r is None or r < 7:
            continue
        cat = get_watched_category(entry)
        if watchlist != "all" and cat != watchlist:
            continue
        try:
            data = ol_work(mid)
            for s in data.get("subjects", [])[:5]:
                s_lower = s.lower().strip()
                if len(s_lower) < 3 or len(s_lower) > 40:
                    continue
                if s_lower in ("fiction", "general", "juvenile fiction"):
                    continue
                seed_subjects[s_lower] = seed_subjects.get(s_lower, 0) + 1
        except Exception:
            continue

    # Also add items from the selected watchlist
    if watchlist != "all":
        wl_items = state.user_data[user]["watchlists"][mode].get(watchlist, [])
    else:
        wl_items = []
        for wl in state.user_data[user]["watchlists"][mode].values():
            wl_items.extend(wl)
    for mid in wl_items:
        try:
            data = ol_work(mid)
            for s in data.get("subjects", [])[:3]:
                s_lower = s.lower().strip()
                if len(s_lower) < 3 or len(s_lower) > 40:
                    continue
                if s_lower in ("fiction", "general", "juvenile fiction"):
                    continue
                seed_subjects[s_lower] = seed_subjects.get(s_lower, 0) + 1
        except Exception:
            continue

    if not seed_subjects:
        await send_back_text(update, "Not enough data to generate recommendations. "
                             "Read and rate more books first!")
        return

    # Pick top subjects and fetch recommendations
    top_subjects = sorted(
        seed_subjects, key=seed_subjects.get, reverse=True)[:5]
    all_recs = {}
    already_known = set(int(k) for k in watched.keys())
    for wl in state.user_data[user]["watchlists"][mode].values():
        already_known.update(wl)

    for subject in top_subjects:
        try:
            subject_slug = subject.replace(" ", "_").lower()
            works = ol_subject(subject_slug, limit=20)
            for w in works:
                key = w.get("key", "")
                wid = None
                if key.startswith("/works/OL") and key.endswith("W"):
                    try:
                        wid = int(key[len("/works/OL"):-1])
                    except ValueError:
                        continue
                if wid and wid not in already_known and wid not in all_recs:
                    all_recs[wid] = w
        except Exception:
            continue

    if not all_recs:
        await send_back_text(update, "No recommendations found. Try reading more books!")
        return

    # Format and display
    infos = []
    for wid, w in all_recs.items():
        title = w.get("title", "Unknown")
        authors = [a.get("name", "") for a in w.get("authors", [])]
        author_str = ", ".join(authors[:2]) if authors else "Unknown author"
        cover_id = w.get("cover_id")
        from bookbot.helpers import get_cover_url
        cover_url = get_cover_url(cover_id)
        desc = f'`{title}` - {author_str}'
        infos.append((0, wid, title, desc))

    random.shuffle(infos)
    infos = infos[:20]
    movies_info = [(wid, title, desc) for _, wid, title, desc in infos]

    cat_label = f'"{watchlist}"' if watchlist != "all" else "all lists"
    await send_movie_list(bot, chat_id,
                          f"Recommendations based on {cat_label}:",
                          movies_info, detail_action="det", media_type=mt)


async def handle_rwl(query, user, raw):
    """Handle recommend category selection."""
    await query.answer()
    idx_str = raw.split(":", 1)[1] if ":" in raw else ""
    mode = "book"
    if idx_str == "all":
        watchlist = "all"
    else:
        try:
            idx = int(idx_str)
            wl_names = list(state.user_data[user]["watchlists"][mode].keys())
            watchlist = wl_names[idx] if idx < len(wl_names) else "all"
        except (ValueError, IndexError):
            watchlist = "all"

    # Need to create a fake update-like object for _do_recommend
    class FakeUpdate:
        def __init__(self, message):
            self.message = message

        def get_bot(self):
            return self.message.get_bot()
    await _do_recommend(FakeUpdate(query.message), user, watchlist)


class PickCommand(BaseCommand):
    """Pick a random book from reading lists."""

    async def execute(self, update, context, user):
        mode = "book"
        all_items = []
        for wn, wl in state.user_data[user]["watchlists"][mode].items():
            for mid in wl:
                all_items.append((mid, wn))
        if not all_items:
            await send_back_text(update, "Your reading lists are empty!")
            return

        mid, wl_name = random.choice(all_items)
        try:
            data = ol_work(mid)
            info = extract_book_info(data, from_search=False)
            _, cover_url, desc, _ = info
        except Exception:
            desc = f"Book ID: {mid}"
            cover_url = None

        from botlib.messaging import send_movie_message
        mt = _mode_to_type(mode)
        pick_text = f'Random pick from "{wl_name}":\n{desc}'

        keyboard = build_media_keyboard(mid, user, mode=mode)
        escaped = esc(pick_text)
        if cover_url:
            msg = await update.message.reply_photo(
                cover_url, escaped,
                parse_mode="MarkdownV2", reply_markup=keyboard)
        else:
            msg = await update.message.reply_text(
                escaped, parse_mode="MarkdownV2", reply_markup=keyboard)

        rpick_kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("Pick another", callback_data="rpick:*")
        ]])
        await update.message.reply_text("Want another?", reply_markup=rpick_kb)


async def handle_rpick(query, user, raw):
    """Pick another random book."""
    await query.answer()
    mode = "book"
    all_items = []
    for wn, wl in state.user_data[user]["watchlists"][mode].items():
        for mid in wl:
            all_items.append((mid, wn))
    if not all_items:
        await query.message.reply_text("Your reading lists are empty!",
                                       reply_markup=get_main_keyboard(user))
        return

    mid, wl_name = random.choice(all_items)
    try:
        data = ol_work(mid)
        info = extract_book_info(data, from_search=False)
        _, cover_url, desc, _ = info
    except Exception:
        desc = f"Book ID: {mid}"
        cover_url = None

    mt = _mode_to_type(mode)
    pick_text = f'Random pick from "{wl_name}":\n{desc}'
    keyboard = build_media_keyboard(mid, user, mode=mode)
    escaped = esc(pick_text)
    bot = query.message.get_bot()
    chat_id = query.message.chat_id
    if cover_url:
        await bot.send_photo(chat_id, cover_url, escaped,
                             parse_mode="MarkdownV2", reply_markup=keyboard)
    else:
        await bot.send_message(chat_id, escaped,
                               parse_mode="MarkdownV2", reply_markup=keyboard)

    rpick_kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("Pick another", callback_data="rpick:*")
    ]])
    try:
        await query.edit_message_reply_markup(reply_markup=rpick_kb)
    except Exception:
        pass


class TrendingCommand(BaseCommand):
    async def execute(self, update, context, user):
        mode = "book"
        mt = _mode_to_type(mode)
        try:
            results = ol_trending(limit=20)
        except Exception as e:
            logger.error("Trending fetch failed: %s", e)
            await send_back_text(update, "Failed to fetch trending books.")
            return

        watched = state.user_data[user].get("watched", {}).get(mode, {})
        infos = []
        for doc in results:
            info = extract_book_info(doc, from_search=True)
            if info[3] is not None and info[3] not in watched:
                infos.append((info[3], doc.get("title", "Unknown"), info[2]))
        if not infos:
            await send_back_text(update, "No new trending books (all already read).")
            return
        infos = infos[:10]
        await send_movie_list(
            update.get_bot(), update.message.chat_id,
            "Trending books today:", infos,
            detail_action="det", media_type=mt)


def register(app, router):
    app.add_handler(CommandHandler(['recommend', 'r'], RecommendCommand()))
    app.add_handler(CommandHandler(['pick', 'p'], PickCommand()))
    app.add_handler(CommandHandler(['trending', 'tr'], TrendingCommand()))
    router.add('rwl', handle_rwl)
    router.add('rpick', handle_rpick)
