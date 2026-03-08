from telegram import ForceReply
from telegram.ext import CommandHandler

from botlib import state
from botlib.base import BaseCommand
from botlib.messaging import send_back_text
from bookbot.keyboards import get_main_keyboard


class StartCommand(BaseCommand):
    require_onboarding = False

    async def execute(self, update, context, user):
        if state.user_data[user].get("onboarded", False):
            await send_back_text(update, "You're already set up!")
            return
        state._pending_name[user] = True
        await update.message.reply_text(
            "Welcome to BookBot! What's your name?",
            reply_markup=ForceReply(selective=True))


async def handle_name_reply(update, user, text):
    if not text:
        await send_back_text(update, "Name cannot be empty.")
        return
    state.user_data[user]["name"] = text.strip()[:50]
    state.user_data[user]["onboarded"] = True
    state.save_user_data()
    kb = get_main_keyboard(user)
    await update.message.reply_text(
        f"Great, {text.strip()[:50]}! You're all set. Use /search to find books.",
        reply_markup=kb)


def register(app, router):
    app.add_handler(CommandHandler("start", StartCommand()))
