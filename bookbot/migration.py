from botlib.migration import run_migrations  # noqa: F401

CURRENT_VERSION = 0


def migrate():
    run_migrations(CURRENT_VERSION, [])
