import logging
from datetime import date, datetime, time as dt_time, timezone

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

from botlib import state
from botlib.helpers import esc, _mode_to_type
from botlib.messaging import send_back_text, send_movie_list
from bookbot.config import hc_book, hc_series, hc_series_books
from bookbot.keyboards import get_main_keyboard

logger = logging.getLogger(__name__)


def _format_book_line(b):
    title = b.get("title") or "Unknown"
    pos = b.get("position")
    pos_str = f"#{pos} — " if pos is not None else ""
    rd = b.get("release_date") or ""
    today = date.today().isoformat()
    suffix = ""
    if not rd:
        suffix = " (TBA)"
    elif rd > today:
        suffix = f" (upcoming, {rd})"
    elif b.get("release_year"):
        suffix = f" ({b['release_year']})"
    return f"{pos_str}{title}{suffix}"


def _followed(user, sid):
    return sid in (state.user_data[user].get("followed_series") or {})


def _follow_button(user, sid):
    label = "Unfollow series ✅" if _followed(user, sid) else "Follow series"
    return InlineKeyboardButton(label, callback_data=f"srsf:{sid}")


async def _show_series_view(bot, chat_id, user, series_id):
    s = hc_series(series_id)
    if not s:
        await bot.send_message(chat_id, "Series not found.")
        return
    books = hc_series_books(series_id)

    header_lines = [f'`{s["name"]}`']
    if s.get("author_name"):
        header_lines.append(f"by {s['author_name']}")
    if s.get("books_count"):
        header_lines.append(f"{s['books_count']} books")
    desc = s.get("description") or ""
    if len(desc) > 400:
        desc = desc[:397] + "..."
    if desc:
        header_lines.append("")
        header_lines.append(desc)

    kb = InlineKeyboardMarkup([[_follow_button(user, series_id)]])
    await bot.send_message(
        chat_id, esc("\n".join(header_lines)),
        parse_mode=ParseMode.MARKDOWN_V2, reply_markup=kb)

    if not books:
        await bot.send_message(chat_id, "No books found in this series.")
        return

    infos = [(b["book_id"], b["title"], _format_book_line(b)) for b in books]
    await send_movie_list(
        bot, chat_id, "Books:", infos,
        detail_action="det", media_type="b")


async def handle_srs(query, user, raw):
    """'srs:b:<book_id>' — show series for a book (picker if multi, jump if single)."""
    await query.answer()
    parts = raw.split(":")
    if len(parts) < 3:
        return
    try:
        book_id = int(parts[2])
    except ValueError:
        return

    book = hc_book(book_id)
    series_list = (book or {}).get("series") or []
    bot = query.message.get_bot()
    chat_id = query.message.chat_id

    if not series_list:
        await bot.send_message(chat_id, "This book isn't part of a series.")
        return

    if len(series_list) == 1:
        await _show_series_view(bot, chat_id, user, series_list[0]["id"])
        return

    rows = []
    for s in series_list:
        pos = s.get("position")
        pos_str = f" #{pos}" if pos is not None else ""
        label = f'{s["name"]}{pos_str}'
        rows.append([InlineKeyboardButton(label, callback_data=f'srsv:{s["id"]}')])
    await bot.send_message(
        chat_id, "Which series?",
        reply_markup=InlineKeyboardMarkup(rows))


async def handle_srsv(query, user, raw):
    """'srsv:<series_id>' — show series view."""
    await query.answer()
    parts = raw.split(":", 1)
    if len(parts) < 2:
        return
    try:
        sid = int(parts[1])
    except ValueError:
        return
    bot = query.message.get_bot()
    await _show_series_view(bot, query.message.chat_id, user, sid)


async def handle_srsf(query, user, raw):
    """'srsf:<series_id>' — toggle follow."""
    parts = raw.split(":", 1)
    if len(parts) < 2:
        await query.answer()
        return
    try:
        sid = int(parts[1])
    except ValueError:
        await query.answer()
        return

    followed = state.user_data[user].setdefault("followed_series", {})
    if sid in followed:
        followed.pop(sid)
        state.save_user_data()
        await query.answer("Unfollowed")
        await query.edit_message_reply_markup(
            reply_markup=InlineKeyboardMarkup([[_follow_button(user, sid)]]))
        return

    # New follow — seed with current book IDs so we don't notify about backlog
    s = hc_series(sid)
    books = hc_series_books(sid)
    seeded = [b["book_id"] for b in books]
    followed[sid] = {
        "name": (s or {}).get("name") or f"Series {sid}",
        "seeded": seeded,
        "added_at": datetime.now(timezone.utc).date().isoformat(),
    }
    state.save_user_data()
    await query.answer(f'Following "{followed[sid]["name"]}"')
    await query.edit_message_reply_markup(
        reply_markup=InlineKeyboardMarkup([[_follow_button(user, sid)]]))


async def _daily_series_check(context):
    """JobQueue callback — check followed series for newly-released books."""
    bot = context.bot
    today = date.today().isoformat()
    changed = False

    for uid, ud in state.user_data.items():
        if not isinstance(uid, int) or not isinstance(ud, dict):
            continue
        followed = ud.get("followed_series")
        if not followed:
            continue

        for sid, entry in list(followed.items()):
            try:
                books = hc_series_books(sid)
            except Exception:
                logger.exception("series fetch failed for sid=%s", sid)
                continue
            seeded = set(entry.get("seeded") or [])
            new_books = []
            for b in books:
                bid = b.get("book_id")
                rd = b.get("release_date") or ""
                if bid in seeded:
                    continue
                if not rd or rd > today:
                    continue
                new_books.append(b)
            if not new_books:
                continue

            lines = [f'New in *{esc(entry.get("name") or "")}*:']
            for b in new_books:
                lines.append(f'• {esc(_format_book_line(b))}')
                seeded.add(b["book_id"])
            entry["seeded"] = sorted(seeded)
            changed = True

            try:
                await bot.send_message(
                    uid, "\n".join(lines),
                    parse_mode=ParseMode.MARKDOWN_V2)
            except Exception:
                logger.exception("notify user %s failed", uid)

    if changed:
        state.save_user_data()


def register(app, router):
    router.add('srs', handle_srs)
    router.add('srsv', handle_srsv)
    router.add('srsf', handle_srsf)

    jq = app.job_queue
    if jq is not None:
        jq.run_daily(_daily_series_check, time=dt_time(hour=9, tzinfo=timezone.utc))
