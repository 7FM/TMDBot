from botlib import state
from botlib.config import settings
from botlib.helpers import get_user_id
from botlib.messaging import unauthorized_msg


class BaseCommand:
    require_onboarding = True

    async def __call__(self, update, context):
        user = get_user_id(update)
        if user not in settings['allowed_users']:
            await unauthorized_msg(update)
            return
        if self.require_onboarding:
            if not state.user_data.get(user, {}).get("onboarded", False):
                await update.message.reply_text(
                    "Please complete onboarding first. Use /start to begin.")
                return
        await self.execute(update, context, user)

    async def execute(self, update, context, user):
        raise NotImplementedError
