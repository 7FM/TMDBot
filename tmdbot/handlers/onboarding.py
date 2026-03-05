from telegram.ext import CommandHandler

from tmdbot import state
from tmdbot.config import _flag_emoji, _region_name
from tmdbot.base import BaseCommand
from tmdbot.helpers import get_all_movie_provider
from tmdbot.keyboards import (
    get_main_keyboard,
    build_region_keyboard,
    build_services_keyboard,
)
from tmdbot.messaging import send_back_text, unauthorized_msg
from tmdbot.helpers import check_user_invalid

from telegram import ForceReply


class StartCommand(BaseCommand):
    require_onboarding = False

    async def execute(self, update, context, user):
        if not state.user_data[user].get("onboarded", False):
            keyboard = build_region_keyboard()
            await update.message.reply_text(
                "Welcome to TMDBot! Let's get you set up.\n\nSelect your region:",
                reply_markup=keyboard)
        else:
            await update.message.reply_text(
                'Welcome back to TMDBot! Use /search to search for a movie.',
                reply_markup=get_main_keyboard(user)
            )


class ServicesCommand(BaseCommand):
    async def execute(self, update, context, user):
        keyboard = build_services_keyboard(user)
        await update.message.reply_text(
            "Your streaming services:",
            reply_markup=keyboard
        )


# Callback handlers

async def handle_reg(query, user, raw):
    code = raw.split(":", 1)[1]
    state.user_data[user]["region"] = code
    state._provider_cache.pop(code, None)
    state.save_user_data()
    if not state.user_data[user].get("onboarded", False) and not state.user_data[user].get("name"):
        state._pending_name[user] = True
        await query.edit_message_text(
            f"Region set to {_flag_emoji(code)} {_region_name(code)}.")
        await query.get_bot().send_message(
            query.message.chat_id,
            "What should other users call you? Enter your display name:",
            reply_markup=ForceReply(selective=True))
        await query.answer()
    else:
        keyboard = build_services_keyboard(user)
        await query.edit_message_text(
            f"Region set to {_flag_emoji(code)} {_region_name(code)}.\n\nSelect your streaming services:",
            reply_markup=keyboard)
        await query.answer()


async def handle_regp(query, user, raw):
    page = int(raw.split(":", 1)[1])
    keyboard = build_region_keyboard(page)
    await query.edit_message_reply_markup(reply_markup=keyboard)
    await query.answer()


async def handle_chreg(query, user, raw):
    keyboard = build_region_keyboard()
    await query.edit_message_text(
        "Select your region:",
        reply_markup=keyboard)
    await query.answer()


async def handle_sp(query, user, raw):
    provider_index = int(raw.split(":", 1)[1])
    all_providers = get_all_movie_provider(state.user_data[user]["region"])
    if provider_index < 0 or provider_index >= len(all_providers):
        await query.answer("Invalid provider.", show_alert=True)
        return
    name = all_providers[provider_index]
    if name in state.user_data[user]["providers"]:
        state.user_data[user]["providers"].remove(name)
        await query.answer(f"Removed {name}.")
    else:
        state.user_data[user]["providers"].append(name)
        await query.answer(f"Added {name}.")
    if not state.user_data[user].get("onboarded", False):
        state.user_data[user]["onboarded"] = True
    state.save_user_data()
    keyboard = build_services_keyboard(user)
    await query.edit_message_reply_markup(reply_markup=keyboard)


async def handle_name_reply(update, user, text):
    """Handle display name reply during onboarding."""
    if text:
        state.user_data[user]["name"] = text.strip()[:50]
        state.save_user_data()
        keyboard = build_services_keyboard(user)
        await update.message.reply_text(
            f"Nice to meet you, {text.strip()[:50]}!\n\nSelect your streaming services:",
            reply_markup=keyboard)
    else:
        await send_back_text(update, "Name cannot be empty. Use /start to try again.")


def register(app, router):
    app.add_handler(CommandHandler('start', StartCommand()))
    app.add_handler(CommandHandler('services', ServicesCommand()))
    router.add('reg', handle_reg)
    router.add('regp', handle_regp)
    router.add('chreg', handle_chreg)
    router.add('sp', handle_sp)
    router.add_onboarding_action('reg')
    router.add_onboarding_action('regp')
    router.add_onboarding_action('chreg')
    router.add_onboarding_action('sp')
