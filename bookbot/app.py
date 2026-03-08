import sys
import logging

from telegram import Update, BotCommand
from telegram.ext import (
    Application, CallbackQueryHandler,
    MessageHandler, filters,
)

from bookbot import config
from botlib.router import Router
from botlib.messaging import register_main_keyboard_fn
from botlib.keyboards import configure_labels
from botlib.hooks import register_metadata_fetcher
from bookbot.reply_handler import reply_handler
from bookbot.keyboards import get_main_keyboard
from bookbot.handlers import (
    onboarding, search, watchlist, shared_wl,
    read, discovery, info, misc,
)

logger = logging.getLogger(__name__)

_HANDLER_MODULES = [
    onboarding, search, watchlist, shared_wl,
    read, discovery, info, misc,
]


def _fetch_book_metadata(media_id, mode):
    """Fetch Open Library metadata for the on_add hook."""
    from bookbot.config import ol_work, _rate_limited_get, OL_BASE
    data = ol_work(media_id)
    title = data.get("title", "")

    # Get author names from author references
    authors = []
    for a in data.get("authors", []):
        author_key = a.get("author", {}).get("key", "")
        if author_key:
            try:
                ad = _rate_limited_get(
                    f"{OL_BASE}{author_key}.json").json()
                authors.append(ad.get("name", ""))
            except Exception:
                pass
    author = ", ".join(authors)

    # Get ISBN from editions
    isbn = ""
    try:
        editions = _rate_limited_get(
            f"{OL_BASE}/works/OL{media_id}W/editions.json",
            params={"limit": 5}).json()
        for ed in editions.get("entries", []):
            isbn_13 = ed.get("isbn_13", [])
            if isbn_13:
                isbn = isbn_13[0]
                break
            isbn_10 = ed.get("isbn_10", [])
            if isbn_10:
                isbn = isbn_10[0]
                break
    except Exception:
        pass

    return {
        "TITLE": title,
        "AUTHOR": author,
        "ISBN": isbn,
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
    ])


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
    configure_labels({"watched": "Read"})
    register_metadata_fetcher(_fetch_book_metadata)

    application = Application.builder().token(
        config.settings["telegram_token"]).post_init(post_init).build()

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
