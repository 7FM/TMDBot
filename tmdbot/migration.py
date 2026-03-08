import logging

from tmdbot import state

logger = logging.getLogger(__name__)

CURRENT_VERSION = 1

_USER_KEYS = set()


def _iter_users(ud):
    """Iterate over all user entries in user data (integer keys)."""
    for key, val in ud.items():
        if isinstance(key, int) and isinstance(val, dict):
            yield key, val


def migrate():
    """Run all pending migrations on user data."""
    ud = state.user_data
    version = ud.get("_version", 0)
    if version >= CURRENT_VERSION:
        return

    migrations = [
        _migrate_to_v1,
    ]

    for target_version, migrate_fn in enumerate(migrations, start=1):
        if version < target_version:
            logger.info("Migrating user data v%d -> v%d", version, target_version)
            migrate_fn(ud)
            version = target_version

    ud["_version"] = CURRENT_VERSION
    state.save_user_data()


def _migrate_to_v1(ud):
    """Watched entries become dicts with rating and category."""
    for user, u in _iter_users(ud):
        for m in ("movie", "tv"):
            if m in u.get("watched", {}):
                for mid in list(u["watched"][m]):
                    val = u["watched"][m][mid]
                    if not isinstance(val, dict):
                        u["watched"][m][mid] = {"rating": val, "category": "normal"}
