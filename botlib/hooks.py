"""Run external scripts on bot events (e.g., adding to a watchlist)."""

import logging
import os
import subprocess

from botlib.config import settings

logger = logging.getLogger(__name__)

# Domain packages register a callback: (media_id, mode) -> dict of env vars
_fetch_metadata = None


def register_metadata_fetcher(fn):
    """Register a function that fetches metadata for a media item.

    fn(media_id, mode) -> dict with keys like TITLE, AUTHOR/DIRECTOR, YEAR, etc.
    """
    global _fetch_metadata
    _fetch_metadata = fn


def run_on_add(media_id, mode, user, watchlist_name):
    """Run the on_add_script if configured."""
    script = settings.get("on_add_script")
    if not script:
        return

    env = os.environ.copy()
    env["MEDIA_ID"] = str(media_id)
    env["MODE"] = mode
    env["USER_ID"] = str(user)
    env["WATCHLIST"] = watchlist_name

    # Fetch domain-specific metadata
    if _fetch_metadata:
        try:
            meta = _fetch_metadata(media_id, mode)
            for key, value in meta.items():
                if value is not None:
                    env[key] = str(value)
        except Exception:
            logger.warning("Failed to fetch metadata for hook", exc_info=True)

    try:
        subprocess.Popen(
            script, shell=True, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        logger.error("Failed to run on_add_script", exc_info=True)
