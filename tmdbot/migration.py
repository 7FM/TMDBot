from botlib.migration import run_migrations, _iter_users  # noqa: F401

CURRENT_VERSION = 1


def _migrate_to_v1(ud):
    """Watched entries become dicts with rating and category."""
    for user, u in _iter_users(ud):
        for m in ("movie", "tv"):
            if m in u.get("watched", {}):
                for mid in list(u["watched"][m]):
                    val = u["watched"][m][mid]
                    if not isinstance(val, dict):
                        u["watched"][m][mid] = {
                            "rating": val, "category": "normal"}


def migrate():
    run_migrations(CURRENT_VERSION, [_migrate_to_v1])
