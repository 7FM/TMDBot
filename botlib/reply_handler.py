from botlib import state
from botlib.helpers import get_user_id, check_user_invalid, is_in_any_watchlist
from botlib.keyboards import build_member_select_keyboard
from botlib.messaging import send_back_text, unauthorized_msg

# Domain-specific pending handlers: list of (check_fn, handler_fn)
# check_fn(user) -> bool, handler_fn(update, user, text) -> coroutine
_pending_handlers = []

# Default text handler (e.g., search on plain text)
_default_text_handler = None


def register_pending_handler(check_fn, handler_fn):
    """Register a domain-specific pending state handler."""
    _pending_handlers.append((check_fn, handler_fn))


def register_default_text_handler(handler_fn):
    """Register handler for non-reply plain text (e.g., default search)."""
    global _default_text_handler
    _default_text_handler = handler_fn


async def reply_handler(update, context):
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    text = update.message.text.strip()

    # Check domain-specific pending handlers first
    for check_fn, handler_fn in _pending_handlers:
        if check_fn(user):
            await handler_fn(update, user, text)
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
        movie_id, nwl_mode = state._pending_new_watchlist.pop(
            user, (None, "movie"))
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
                state.user_data[user]["watchlists"][nwl_mode][text].append(
                    movie_id)
                state.save_user_data()
                await send_back_text(update, f'Created watchlist "{text}" and added it.')
        return
