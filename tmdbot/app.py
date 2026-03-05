import sys
import re
import datetime
import logging

from telegram import Update, BotCommand
from telegram.ext import (
    Application, CallbackQueryHandler,
    MessageHandler, filters,
)

from tmdbot import config, state
from tmdbot.router import Router
from tmdbot.reply_handler import reply_handler
from tmdbot.keyboards import _MODE_SWITCH_TV, _MODE_SWITCH_MOVIE
from tmdbot.handlers import (
    onboarding, search, watchlist, shared_wl,
    watched, discovery, tv_seasons, info, misc,
)

logger = logging.getLogger(__name__)

_HANDLER_MODULES = [
    onboarding, search, watchlist, shared_wl,
    watched, discovery, tv_seasons, info, misc,
]


async def post_init(application):
    await application.bot.set_my_commands(commands=[
        BotCommand("start", "OKAAAAY LETS GO!!!"),
        BotCommand("search", "Search by keywords"),
        BotCommand("list", "Browse your watchlists"),
        BotCommand("add", "Add to your watchlist"),
        BotCommand("tadd", "Add to your trash watchlist"),
        BotCommand("watched", "Mark as watched"),
        BotCommand("remove", "Remove from all watchlists"),
        BotCommand("rate", "Rate or re-rate watched items"),
        BotCommand("services", "Manage my streaming services"),
        BotCommand("check", "Check streaming availability for your watchlist"),
        BotCommand("recommend", "Get recommendations based on your watchlist"),
        BotCommand("popular", "Show popular titles on your streaming services"),
        BotCommand("pick", "Pick a random title from your watchlists"),
        BotCommand("mode", "Switch between Movies and TV mode"),
        BotCommand("newseasons", "Check for new seasons of watched TV shows"),
        BotCommand("seasons", "View/edit watched seasons for TV shows"),
        BotCommand("stats", "View your watch statistics"),
        BotCommand("trending", "Show trending titles today"),
        BotCommand("person", "Search by actor/director"),
        BotCommand("setname", "Set your display name"),
    ])
    if application.job_queue:
        from tmdbot.handlers.tv_seasons import _daily_season_check
        application.job_queue.run_daily(
            _daily_season_check,
            time=datetime.time(hour=9, minute=0),
        )
    else:
        logger.warning("JobQueue not available. Daily season check disabled. "
                       "Install python-telegram-bot[job-queue] to enable.")


async def error_handler(update, context):
    logger.error("Exception while handling an update:", exc_info=context.error)
    try:
        if isinstance(update, Update) and update.effective_chat:
            from tmdbot.helpers import get_user_id
            from tmdbot.keyboards import get_main_keyboard
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
        filters.Regex(
            f"^({re.escape(_MODE_SWITCH_TV)}|{re.escape(_MODE_SWITCH_MOVIE)})$"),
        misc.ToggleModeCommand()
    ))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & ~filters.REPLY,
        search.default_search_handler
    ))

    application.add_error_handler(error_handler)

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
