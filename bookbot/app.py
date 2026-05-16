import sys
import logging
from datetime import time as dt_time, timezone

from telegram import InlineKeyboardButton, Update, BotCommand
from telegram.ext import (
    Application, CallbackQueryHandler,
    MessageHandler, filters,
)

from bookbot import config
from botlib.router import Router
from botlib.messaging import register_main_keyboard_fn
from botlib.keyboards import configure_labels, register_media_button
from botlib.hooks import register_metadata_fetcher
from bookbot.reply_handler import reply_handler
from bookbot.keyboards import get_main_keyboard
from bookbot.handlers import (
    onboarding, search, watchlist, shared_wl,
    read, discovery, info, misc, series,
)

logger = logging.getLogger(__name__)

_HANDLER_MODULES = [
    onboarding, search, watchlist, shared_wl,
    read, discovery, info, misc, series,
]


def _series_button(media_id, user, mode):
    if mode != "book":
        return None
    return InlineKeyboardButton("Series", callback_data=f"srs:b:{media_id}")


def _fetch_book_metadata(media_id, mode):
    """Fetch Hardcover metadata for the on_add hook."""
    from bookbot.config import hc_book, hc_book_isbn
    book = hc_book(media_id) or {}
    return {
        "TITLE": book.get("title") or "",
        "AUTHOR": ", ".join(book.get("authors") or []),
        "ISBN": hc_book_isbn(media_id),
        "MEDIA_TYPE": "book",
    }


async def post_init(application):
    await application.bot.set_my_commands(commands=[
        BotCommand("start", "Get started!"),
        BotCommand("search", "Search for books"),
        BotCommand("list", "Browse your reading lists"),
        BotCommand("add", "Add to reading list"),
        BotCommand("read", "Mark as read"),
        BotCommand("rate", "Rate or re-rate read books"),
        BotCommand("recommend", "Get book recommendations"),
        BotCommand("trending", "Trending books today"),
        BotCommand("pick", "Random book from your lists"),
        BotCommand("author", "Search by author"),
        BotCommand("stats", "View your reading statistics"),
        BotCommand("setname", "Set your display name"),
        BotCommand("fix", "Restore keyboard"),
    ])


_auth_alert_recent = {"sent_at": 0.0}
_AUTH_ALERT_COOLDOWN_SEC = 3600


def _setup_token_monitoring(application):
    """Wire up startup warning, daily check, and auth-failure alerting."""
    import asyncio
    import time
    from bookbot.config import (
        token_days_remaining, token_expiry_datetime,
        register_auth_failure_handler,
    )

    async def _alert_all(text):
        for uid in config.settings.get("allowed_users", []):
            try:
                await application.bot.send_message(uid, text)
            except Exception:
                logger.exception("token alert to %s failed", uid)

    def _on_auth_failure(msg):
        now = time.time()
        if now - _auth_alert_recent["sent_at"] < _AUTH_ALERT_COOLDOWN_SEC:
            return
        _auth_alert_recent["sent_at"] = now
        text = (
            f"⚠️ BookBot lost Hardcover access: {msg}. "
            "Please refresh `hardcover_token` in settings.yaml."
        )
        try:
            asyncio.get_event_loop().create_task(_alert_all(text))
        except Exception:
            logger.exception("scheduling auth-failure alert failed")

    register_auth_failure_handler(_on_auth_failure)

    async def _post(app):
        days = token_days_remaining()
        exp = token_expiry_datetime()
        if days is None:
            logger.warning("Could not decode hardcover_token expiry")
            await _alert_all(
                "⚠️ BookBot couldn't decode the Hardcover token. "
                "Check `hardcover_token` in settings.yaml.")
            return
        logger.info("Hardcover token expires in %.1f days (%s)", days, exp)
        if days < 30:
            await _alert_all(
                f"⚠️ Hardcover token expires in {int(days)} days "
                f"({exp.date().isoformat() if exp else '?'}). Please refresh it.")

    async def _daily(context):
        days = token_days_remaining()
        if days is None or days >= 14:
            return
        exp = token_expiry_datetime()
        await _alert_all(
            f"⚠️ Hardcover token expires in {int(days)} days "
            f"({exp.date().isoformat() if exp else '?'}). Please refresh it.")

    application.post_init = _chain_post_init(application.post_init, _post)
    jq = application.job_queue
    if jq is not None:
        jq.run_daily(_daily, time=dt_time(hour=9, minute=30, tzinfo=timezone.utc))


def _chain_post_init(existing, extra):
    async def chained(app):
        if existing is not None:
            await existing(app)
        await extra(app)
    return chained


async def error_handler(update, context):
    logger.error("Exception while handling an update:", exc_info=context.error)
    try:
        if isinstance(update, Update) and update.effective_chat:
            from botlib.helpers import get_user_id
            user = get_user_id(update) if update.effective_user else None
            kb = get_main_keyboard(user) if user else None
            await context.bot.send_message(
                update.effective_chat.id,
                "An unexpected error occurred. Please try again.",
                reply_markup=kb)
    except Exception:
        logger.error("Failed to send error message to user:", exc_info=True)


def main():
    settings_file = "settings.yaml" if len(sys.argv) < 2 else sys.argv[1]
    user_data_file = 'user_data.yaml' if len(sys.argv) < 3 else sys.argv[2]

    config.init(settings_file, user_data_file)

    # Register domain-specific overrides with botlib
    register_main_keyboard_fn(get_main_keyboard)
    configure_labels({"watched": "Read", "new_watchlist": "New list"})
    register_metadata_fetcher(_fetch_book_metadata)
    register_media_button(_series_button)

    application = Application.builder().token(
        config.settings["telegram_token"]).post_init(post_init).build()

    _setup_token_monitoring(application)

    router = Router()
    for module in _HANDLER_MODULES:
        module.register(application, router)

    application.add_handler(CallbackQueryHandler(router))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.REPLY,
        reply_handler
    ))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & ~filters.REPLY,
        search.default_search_handler
    ))

    application.add_error_handler(error_handler)

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
