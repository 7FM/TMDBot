from botlib import state
from botlib.reply_handler import (
    reply_handler,  # noqa: F401
    register_pending_handler,
)


async def _handle_pending_search(update, user, text):
    state._pending_search.pop(user, None)
    if text:
        from bookbot.handlers.search import do_search
        await do_search(update, text, user)


async def _handle_pending_name(update, user, text):
    state._pending_name.pop(user, None)
    from bookbot.handlers.onboarding import handle_name_reply
    await handle_name_reply(update, user, text)


async def _handle_pending_person(update, user, text):
    state._pending_person.pop(user, None)
    if text:
        from bookbot.handlers.info import do_author_search
        await do_author_search(update, text, user)


register_pending_handler(
    lambda user: user in state._pending_search,
    _handle_pending_search)

register_pending_handler(
    lambda user: user in state._pending_name,
    _handle_pending_name)

register_pending_handler(
    lambda user: user in state._pending_person,
    _handle_pending_person)
