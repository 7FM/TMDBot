import logging

from botlib import state

logger = logging.getLogger(__name__)


def _iter_users(ud):
    """Iterate over all user entries in user data (integer keys)."""
    for key, val in ud.items():
        if isinstance(key, int) and isinstance(val, dict):
            yield key, val


def run_migrations(current_version, migrations):
    """Run all pending migrations on user data.

    current_version: the target version number
    migrations: list of callables, each taking ud dict
    """
    ud = state.user_data
    version = ud.get("_version", 0)
    if version >= current_version:
        return

    for target_version, migrate_fn in enumerate(migrations, start=1):
        if version < target_version:
            logger.info("Migrating user data v%d -> v%d",
                        version, target_version)
            migrate_fn(ud)
            version = target_version

    ud["_version"] = current_version
    state.save_user_data()
