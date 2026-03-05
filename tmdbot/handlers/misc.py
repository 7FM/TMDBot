from telegram.ext import CommandHandler

from tmdbot import state
from tmdbot.config import settings
from tmdbot.base import BaseCommand
from tmdbot.helpers import check_user_invalid
from tmdbot.keyboards import get_main_keyboard
from tmdbot.messaging import (
    send_back_text, unauthorized_msg,
    _cleanup_search_results, _cleanup_rate_list,
)


class FixCommand(BaseCommand):
    require_onboarding = False

    async def execute(self, update, context, user):
        await update.message.reply_text("Keyboard restored.", reply_markup=get_main_keyboard(user))


class SetNameCommand(BaseCommand):
    require_onboarding = False

    async def execute(self, update, context, user):
        if not context.args:
            current = state.user_data[user].get("name", "")
            if current:
                await send_back_text(update, f'Your current name is "{current}". Use /setname <name> to change it.')
            else:
                await send_back_text(update, "Use /setname <name> to set your display name.")
            return
        name = ' '.join(context.args)[:50]
        state.user_data[user]["name"] = name
        state.save_user_data()
        await send_back_text(update, f'Display name set to "{name}".')


class ToggleModeCommand(BaseCommand):
    async def execute(self, update, context, user):
        current = state.user_data[user].get("mode", "movie")
        new_mode = "tv" if current == "movie" else "movie"
        state.user_data[user]["mode"] = new_mode
        state.save_user_data()
        label = "TV Shows" if new_mode == "tv" else "Movies"
        await update.message.reply_text(
            f"Switched to {label} mode.",
            reply_markup=get_main_keyboard(user))


class ClearCommand(BaseCommand):
    async def execute(self, update, context, user):
        chat_id = update.message.chat_id
        current_id = update.message.message_id
        bot = update.get_bot()
        await _cleanup_search_results(bot, user)
        await _cleanup_rate_list(bot, user)
        state._rec_genre_filter.pop(user, None)
        state._last_watched.pop(user, None)
        deleted = 0
        consecutive_fails = 0
        for msg_id in range(current_id, max(current_id - 500, 0), -1):
            try:
                await bot.delete_message(chat_id, msg_id)
                deleted += 1
                consecutive_fails = 0
            except Exception:
                consecutive_fails += 1
                if consecutive_fails >= 30:
                    break
        await bot.send_message(
            chat_id,
            f"Cleared {deleted} messages.",
            reply_markup=get_main_keyboard(user))


def register(app, router):
    app.add_handler(CommandHandler('fix', FixCommand()))
    app.add_handler(CommandHandler('setname', SetNameCommand()))
    app.add_handler(CommandHandler(['mode', 'm'], ToggleModeCommand()))
    app.add_handler(CommandHandler('clear', ClearCommand()))
