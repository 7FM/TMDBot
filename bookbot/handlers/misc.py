from telegram.ext import CommandHandler

from botlib import state
from botlib.base import BaseCommand
from botlib.helpers import check_user_invalid
from botlib.messaging import (
    send_back_text, unauthorized_msg,
    _cleanup_search_results, _cleanup_rate_list,
)
from bookbot.keyboards import get_main_keyboard


class FixCommand(BaseCommand):
    require_onboarding = False

    async def execute(self, update, context, user):
        kb = get_main_keyboard(user)
        await update.message.reply_text("Keyboard restored.", reply_markup=kb)


class SetNameCommand(BaseCommand):
    async def execute(self, update, context, user):
        if not context.args:
            await send_back_text(update, "Usage: /setname <name>")
            return
        name = " ".join(context.args).strip()[:50]
        state.user_data[user]["name"] = name
        state.save_user_data()
        await send_back_text(update, f"Display name set to: {name}")


class ClearCommand(BaseCommand):
    async def execute(self, update, context, user):
        bot = update.get_bot()
        await _cleanup_search_results(bot, user)
        await _cleanup_rate_list(bot, user)
        await send_back_text(update, "Cleared.")


def register(app, router):
    app.add_handler(CommandHandler('fix', FixCommand()))
    app.add_handler(CommandHandler('setname', SetNameCommand()))
    app.add_handler(CommandHandler('clear', ClearCommand()))
