import logging

from tmdbot import state
from tmdbot.config import settings

logger = logging.getLogger(__name__)


class Router:
    def __init__(self):
        self._handlers = {}
        self._onboarding_actions = set()
        self._fallback = None

    def add(self, action, handler):
        self._handlers[action] = handler

    def add_onboarding_action(self, action):
        """Mark an action as allowed before onboarding completes."""
        self._onboarding_actions.add(action)

    def set_fallback(self, handler):
        """Set fallback for parse_callback_data-based actions."""
        self._fallback = handler

    async def __call__(self, update, context):
        query = update.callback_query
        user = query.from_user.id

        if user not in settings['allowed_users']:
            await query.answer("Unauthorized user.")
            return

        raw = query.data
        action = raw.split(":", 1)[0]

        if action not in self._onboarding_actions:
            if not state.user_data.get(user, {}).get("onboarded", False):
                await query.answer("Please complete onboarding first. Use /start.", show_alert=True)
                return

        handler = self._handlers.get(action)
        if handler:
            await handler(query, user, raw)
            return

        if self._fallback:
            await self._fallback(query, user, raw)
            return

        await query.answer("Unknown action.", show_alert=True)
