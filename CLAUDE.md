# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TMDBot is a Telegram bot for discovering and managing movies using The Movie Database (TMDb) API. It provides movie search, watchlist management, streaming provider integration, rating, recommendation features, random movie picker, and user onboarding. The entire bot is implemented in a single file (`tmdbot.py`, ~1760 lines).

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

1. **Global initialization** (lines 1-130): Loads settings, initializes TMDb API clients, genre cache, logger, and `REGIONS` list at module level
2. **Global state** (lines 124-145): In-memory caches and UI constants — `_provider_cache`, `_pending_new_watchlist`, `_pending_search`, `_chunk_movies`/`_chunk_id_counter` (expand/collapse state), `_search_results` (search message tracking), `_search_more` (search pagination state), `_rate_list_messages` (rate list tracking), `_rec_genre_filter` (recommendation genre filter state), `_last_watched` (undo state), `MAIN_KEYBOARD` (persistent reply keyboard)
3. **Helper functions** (lines 147-540): Movie info extraction, MarkdownV2 escaping, message chunking, provider lookups (all using `append_to_response` via `_parse_providers_from_details`), keyboard builders (including `build_genre_picker_keyboard`), `send_movie_message()`, `send_movie_list()`
4. **Command handlers** (lines 543-1040): Async handlers registered via `python-telegram-bot`'s `CommandHandler`
5. **Callback handler** (`button_callback_handler()`, line 1194): Central dispatcher for all inline keyboard button presses
6. **Reply handler** (`reply_handler()`, line 1641): Handles `ForceReply` responses for search and new watchlist name input
7. **Default text handler** (`default_search_handler()`, line 1680): Handles plain text (non-reply) as search queries
8. **Error handler** (`error_handler()`, line 1691): Logs exceptions and sends user-friendly error messages
9. **Entry point** (`main()`, line 1721): Builds the `Application` with `post_init` for async setup, registers all handlers and error handler, starts polling

**Key patterns:**

- All command handlers are `async def` functions taking `(Update, ContextTypes.DEFAULT_TYPE)`
- Every handler starts with user authorization check via `check_user_invalid()`
- User state is a global `user_data` dict persisted to YAML via `save_user_data()`
- `user_data_initialize()` handles data migrations (e.g., `watched` list → dict for ratings, `onboarded` flag for existing users)
- Text output uses `send_back_text()` which escapes for MarkdownV2, splits long messages, and always restores the persistent keyboard
- Movie search results use `send_movie_message()` which attaches inline keyboard buttons (Add/Remove/Watched); search result message IDs are tracked in `_search_results` for cleanup after user action
- List-based views (watchlist browsing, recommend, check, popular, rate) use `send_movie_list(bot, chat_id, ...)` which produces chunked itemized lists with expand/collapse buttons, link preview disabled, and returns sent `Message` objects for tracking
- `extract_movie_info(m, skip_trailer=False)` — list views pass `skip_trailer=True` to avoid per-movie API calls; trailers are fetched lazily when the user taps a detail button
- `/check` and `/recommend` use `_with_progress_bar(bot, chat_id, label, total, work_fn)` which runs blocking ThreadPoolExecutor work in a background thread via `asyncio.to_thread()`, while an async task updates a text progress bar (`[████░░░░░░] 4/10`) every 0.5 seconds
- `/popular` uses `ThreadPoolExecutor` for parallel provider lookups with a simple status message (no progress bar)
- All provider lookups go through `movie.details(id, append_to_response="watch/providers")` + `_parse_providers_from_details`
- Search results show 5 at a time with a "Show more" button; remaining results are stored in `_search_more` per user
- `/pick` picks a random movie from all watchlists (or a named one) that is available on the user's streaming services, with a "Pick another" inline button (`rpick` callback)
- Detail messages (`det`/`rdet` callbacks) are sent as standalone messages (not replies) via `bot.send_photo`/`bot.send_message` so Telegram clients auto-scroll to them
- `/search` without args uses `ForceReply` to prompt for input; new watchlist name entry (`nwl`/`new` callbacks) also uses `ForceReply`. Reply handler dispatches based on `_pending_search` and `_pending_new_watchlist` state. Plain text without reply defaults to search via `default_search_handler`
- Marking a movie as watched sends a confirmation with `MAIN_KEYBOARD`, then a separate "Undo?" message with an inline button. The `undo` callback restores the movie to its previous watchlist and rating state via `_last_watched`

**Inline keyboard & callback system:**

All button presses route through `button_callback_handler()` using colon-delimited callback data (must fit in 64 bytes). Action prefixes:
- `pick:<id>` — show watchlist picker; `a:<id>:<watchlist>` — add to watchlist; `rm:<id>` — remove; `w:<id>` — mark watched (shows rating keyboard with skip option)
- `back:<id>` — return from picker to Add/Watched buttons; `new:<id>` — create new watchlist (ForceReply) and add movie
- `rate:<id>:<0-10>` — submit rating (from search/watchlist context, 0=skip); `rrate:<id>:<0-10>` — submit rating (from `/rate` flow, triggers list refresh)
- `wl:<name>` — browse watchlist contents; `det:<id>` — show full movie detail card; `rdet:<id>` — show movie with rating keyboard (from `/rate` list)
- `exp:<chunk_id>` / `col:<chunk_id>` — expand/collapse movie button lists
- `nwl` — new watchlist from list view (ForceReply); `wledit` / `wlback` — enter/exit edit mode; `dwl:<name>` / `dwly:<name>` / `dwln` — delete watchlist flow
- `sp:<index>` — toggle streaming provider (first selection completes onboarding)
- `reg:<code>` — select region; `regp:<page>` — region picker pagination; `chreg` — change region
- `rpick:<watchlist|*>` — pick another random available movie (`*` = all watchlists)
- `smore` — show next 5 search results
- `gf:<genre_id>` — toggle genre in recommendation filter; `recgo:skip` / `recgo:filter` — launch recommendations (all genres or filtered)
- `undo` — undo last "mark as watched" action (restores watchlist placement and previous rating)

**Onboarding:** New users (flagged with `onboarded: false`) get a region picker → streaming service selector flow on `/start`. Region picker is paginated with flag emojis. Onboarding completes automatically when the first service is selected.

**Message cleanup:** Search results are tracked per-user in `_search_results` and deleted after user action (add/remove/watched), with a confirmation message sent to restore `MAIN_KEYBOARD`. `_search_more` and `_pending_search` state are also cleaned up alongside search results. Rate list messages are tracked in `_rate_list_messages` and refreshed after rating via `rrate` (but not `rate` from other contexts). Previous search results are also cleaned up when a new search starts.

**Multi-step interactions** use `ForceReply` prompts with pending state dicts (`_pending_new_watchlist`, `_pending_search`), handled by `reply_handler()`. Plain text input (non-reply) defaults to search via `default_search_handler()`. `/fix` restores the persistent keyboard if ForceReply causes it to disappear.

**MarkdownV2 escaping:** `esc()` uses regex to preserve `[text](url)` links (escaping text inside brackets, escaping `\` and `)` in URLs) and `` `code` `` spans, while `_esc_plain()` escapes all reserved characters in plain text. tmdbv3api's `AsObj` wrapper raises `AttributeError` instead of `KeyError` for missing keys — catch both in try/except blocks.

**Dependencies:** `python-telegram-bot`, `tmdbv3api` (v1.9.0, custom Nix build), `pyyaml`

## Bot Commands

Commands are registered with short aliases (e.g., `/search`/`/s`, `/list`/`/l`, `/recommend`/`/r`, `/pick`/`/p`). The full mapping is in `main()` at the handler registration block (line 1721+). A persistent reply keyboard provides quick access to `/search`, `/list`, `/check`, `/recommend`, `/popular`, `/pick`, and `/clear`. `/fix` restores the keyboard if lost. Plain text without a command triggers a search. `/search` without args uses ForceReply for immediate input.
