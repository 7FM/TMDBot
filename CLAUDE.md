# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TMDBot is a Telegram bot for discovering and managing movies using The Movie Database (TMDb) API. It provides movie search, watchlist management, streaming provider integration, rating, recommendation features, and user onboarding. The entire bot is implemented in a single file (`tmdbot.py`, ~1285 lines).

## Build & Run

This project uses **Nix flakes** for environment management with **direnv** integration.

- **Enter dev shell:** `nix develop` (or automatic via direnv)
- **Run directly:** `python tmdbot.py [settings.yaml] [user_data.yaml]`
- **Build Nix package:** `nix build`
- **Run built package:** `./result/bin/tmdbot`
- **Format code:** `autopep8 --in-place tmdbot.py`

There are no tests in this project.

## Configuration

Two YAML files (git-ignored) are required at runtime:
- `settings.yaml` — Telegram bot token, TMDb API key, allowed user IDs
- `user_data.yaml` — Per-user state (watchlists, providers, watched history with ratings, region, onboarded flag); auto-created if missing

## Architecture

**Single-file design:** All logic lives in `tmdbot.py`. Key layers:

1. **Global initialization** (lines 1-125): Loads settings, initializes TMDb API clients, genre cache, and `REGIONS` list at module level
2. **Global state** (lines 127-143): In-memory caches and UI constants — `_provider_cache`, `_pending_new_watchlist`, `_pending_search`, `_chunk_movies`/`_chunk_id_counter` (expand/collapse state), `_search_results` (search message tracking), `_rate_list_messages` (rate list tracking), `MAIN_KEYBOARD` (persistent reply keyboard)
3. **Helper functions** (lines 145-530): Movie info extraction, MarkdownV2 escaping, message chunking, provider lookups (including `append_to_response` optimization), keyboard builders, `send_movie_message()`, `send_movie_list()`
4. **Command handlers** (lines 535-930): Async handlers registered via `python-telegram-bot`'s `CommandHandler`
5. **Callback handler** (`button_callback_handler()`, line 933): Central dispatcher for all inline keyboard button presses
6. **Reply handler** (`reply_handler()`, line 1206): Handles `ForceReply` responses for search and new watchlist creation
7. **Entry point** (`main()`, line 1259): Builds the `Application` with `post_init` for async setup, registers all handlers, starts polling

**Key patterns:**

- All command handlers are `async def` functions taking `(Update, ContextTypes.DEFAULT_TYPE)`
- Every handler starts with user authorization check via `check_user_invalid()`
- User state is a global `user_data` dict persisted to YAML via `save_user_data()`
- `user_data_initialize()` handles data migrations (e.g., `watched` list → dict for ratings, `onboarded` flag for existing users)
- Text output uses `send_back_text()` which escapes for MarkdownV2, splits long messages, and always restores the persistent keyboard
- Movie search results use `send_movie_message()` which attaches inline keyboard buttons (Add/Remove/Watched); search result message IDs are tracked in `_search_results` for cleanup after user action
- List-based views (watchlist browsing, recommend, check, popular, rate) use `send_movie_list(bot, chat_id, ...)` which produces chunked itemized lists with expand/collapse buttons, link preview disabled, and returns sent `Message` objects for tracking
- `extract_movie_info(m, skip_trailer=False)` — list views pass `skip_trailer=True` to avoid per-movie API calls; trailers are fetched lazily when the user taps a detail button
- Recommendations use `concurrent.futures.ThreadPoolExecutor` for parallel TMDb API calls, with sources capped at watchlist + top 20 highest-rated watched movies
- `/check` uses `movie.details(id, append_to_response="watch/providers")` to combine details and provider lookup into a single API call per movie
- Detail messages (`det`/`rdet` callbacks) are sent as standalone messages (not replies) via `bot.send_photo`/`bot.send_message` so Telegram clients auto-scroll to them

**Inline keyboard & callback system:**

All button presses route through `button_callback_handler()` using colon-delimited callback data (must fit in 64 bytes). Action prefixes:
- `pick:<id>` — show watchlist picker; `a:<id>:<watchlist>` — add to watchlist; `rm:<id>` — remove; `w:<id>` — mark watched (shows rating keyboard)
- `back:<id>` — return from picker to Add/Watched buttons; `new:<id>` — create new watchlist and add movie
- `rate:<id>:<1-10>` — submit rating (from search/watchlist context); `rrate:<id>:<1-10>` — submit rating (from `/rate` flow, triggers list refresh)
- `wl:<name>` — browse watchlist contents; `det:<id>` — show full movie detail card; `rdet:<id>` — show movie with rating keyboard (from `/rate` list)
- `exp:<chunk_id>` / `col:<chunk_id>` — expand/collapse movie button lists
- `nwl` — new watchlist from list view; `wledit` / `wlback` — enter/exit edit mode; `dwl:<name>` / `dwly:<name>` / `dwln` — delete watchlist flow
- `sp:<index>` — toggle streaming provider
- `reg:<code>` — select region; `regp:<page>` — region picker pagination; `chreg` — change region; `obdone` — complete onboarding

**Onboarding:** New users (flagged with `onboarded: false`) get a region picker → streaming service selector flow on `/start`. Region picker is paginated with flag emojis. The services keyboard includes a "Change region" button for all users and a "Done" button during onboarding.

**Message cleanup:** Search results are tracked per-user in `_search_results` and deleted after user action (add/remove/watched), with a confirmation message sent to restore `MAIN_KEYBOARD`. Rate list messages are tracked in `_rate_list_messages` and refreshed after rating via `rrate` (but not `rate` from other contexts). Previous search results are also cleaned up when a new search starts.

**Multi-step interactions** use `ForceReply` + pending state dicts (`_pending_new_watchlist`, `_pending_search`), handled by `reply_handler()`.

**MarkdownV2 escaping:** `esc()` uses regex to preserve `[text](url)` links (escaping text inside brackets, escaping `\` and `)` in URLs) and `` `code` `` spans, while `_esc_plain()` escapes all reserved characters in plain text. tmdbv3api's `AsObj` wrapper raises `AttributeError` instead of `KeyError` for missing keys — catch both in try/except blocks.

**Dependencies:** `python-telegram-bot`, `tmdbv3api` (v1.9.0, custom Nix build), `pyyaml`

## Bot Commands

Commands are registered with short aliases (e.g., `/search`/`/s`, `/list`/`/l`, `/recommend`/`/r`). The full mapping is in `main()` at the handler registration block (line 1259+). A persistent reply keyboard provides quick access to `/search`, `/list`, `/check`, `/recommend`, `/popular`.
