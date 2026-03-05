from tmdbot import state
from tmdbot.helpers import get_user_id, check_user_invalid, is_in_any_watchlist
from tmdbot.keyboards import build_member_select_keyboard
from tmdbot.messaging import send_back_text, unauthorized_msg


async def reply_handler(update, context):
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    text = update.message.text.strip()

    # Handle pending search
    if user in state._pending_search:
        state._pending_search.pop(user, None)
        if text:
            from tmdbot.handlers.search import do_search
            await do_search(update, text, user)
        return

    # Handle pending person search
    if user in state._pending_person:
        state._pending_person.pop(user, None)
        if text:
            from tmdbot.handlers.info import do_person_search
            await do_person_search(update, text, user)
        return

    # Handle pending display name (onboarding)
    if user in state._pending_name:
        state._pending_name.pop(user, None)
        from tmdbot.handlers.onboarding import handle_name_reply
        await handle_name_reply(update, user, text)
        return

    # Handle pending shared watchlist name
    if user in state._pending_shared_wl_name:
        s = state._pending_shared_wl_name.pop(user)
        if not text:
            await send_back_text(update, "Name cannot be empty.")
            return
        state._pending_shared_wl_members[user] = {
            "name": text.strip()[:50],
            "media_id": s.get("media_id"),
            "mode": s.get("mode", "movie"),
            "members": [],
        }
        keyboard = build_member_select_keyboard(user, [])
        await update.message.reply_text(
            f'Select members for "{text.strip()[:50]}":',
            reply_markup=keyboard)
        return

    # Handle pending new watchlist name
    if user in state._pending_new_watchlist:
        movie_id, nwl_mode = state._pending_new_watchlist.pop(user, (None, "movie"))
        if not text:
            await send_back_text(update, "Watchlist name cannot be empty.")
            return
        if text in state.user_data[user]["watchlists"][nwl_mode]:
            await send_back_text(update, f'Watchlist "{text}" already exists.')
            return
        state.user_data[user]["watchlists"][nwl_mode][text] = []
        if movie_id is None:
            state.save_user_data()
            await send_back_text(update, f'Created watchlist "{text}".')
        else:
            already_in = is_in_any_watchlist(movie_id, user, mode=nwl_mode)
            if already_in:
                await send_back_text(update, f'Already in your "{already_in}" watchlist.')
            else:
                state.user_data[user]["watchlists"][nwl_mode][text].append(movie_id)
                state.save_user_data()
                await send_back_text(update, f'Created watchlist "{text}" and added it.')
        return
