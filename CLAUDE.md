# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TMDBot is a Telegram bot for discovering and managing movies and TV shows using The Movie Database (TMDb) API. It provides search, watchlist management, streaming provider integration, rating, recommendation features, random picker, and user onboarding — with a per-user mode switch between Movies and TV. The entire bot is implemented in a single file (`tmdbot.py`, ~1960 lines).

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
- `user_data.yaml` — Per-user state (mode, watchlists nested by mode, providers, watched history with ratings nested by mode, region, onboarded flag); auto-created if missing

## Architecture

**Single-file design:** All logic lives in `tmdbot.py`. Key layers:

1. **Global initialization** (lines 1-155): Loads settings, initializes TMDb API clients (`Movie`, `TV`, `Search`, `Genre`, `Provider`), dual genre caches (`movie_genre_dict`, `tv_genre_dict`), logger, `REGIONS` list, and mode helpers (`get_api()`, `get_genre_dict()`, `_mode_to_type()`, `_type_to_mode()`) at module level
2. **Global state** (lines 157-175): In-memory caches and UI constants — `_provider_cache` (with TTL), `_pending_new_watchlist` (stores `(movie_id, mode)` tuple), `_pending_search`, `_chunk_movies`/`_chunk_id_counter` (expand/collapse state, 3-tuple with media_type), `_search_results` (search message tracking), `_search_more` (search pagination state), `_rate_list_messages` (rate list tracking), `_rec_genre_filter` (recommendation genre filter state), `_last_watched` (undo state with mode), `get_main_keyboard(user)` (dynamic persistent reply keyboard with mode toggle button)
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
- `user_data_initialize()` handles data migrations (e.g., `watched` list → dict for ratings, `onboarded` flag, flat watchlists/watched → nested by mode `{"movie": ..., "tv": ...}`)
- Text output uses `send_back_text()` which escapes for MarkdownV2, splits long messages, and always restores the persistent keyboard
- Movie search results use `send_movie_message()` which attaches inline keyboard buttons (Add/Remove/Watched); search result message IDs are tracked in `_search_results` for cleanup after user action
- List-based views (watchlist browsing, recommend, check, popular, rate) use `send_movie_list(bot, chat_id, ...)` which produces chunked itemized lists with expand/collapse buttons, link preview disabled, and returns sent `Message` objects for tracking
- `extract_movie_info(m, skip_trailer=False, mode="movie")` — handles both movie and TV fields (`title`/`name`, `release_date`/`first_air_date`, `number_of_seasons` for TV); list views pass `skip_trailer=True` to avoid per-movie API calls; trailers are fetched lazily when the user taps a detail button
- `/check` and `/recommend` use `_with_progress_bar(bot, chat_id, label, total, work_fn)` which runs blocking ThreadPoolExecutor work in a background thread via `asyncio.to_thread()`, while an async task updates a text progress bar (`[████░░░░░░] 4/10`) every 0.5 seconds
- `/popular` uses `ThreadPoolExecutor` for parallel provider lookups with a simple status message (no progress bar)
- All provider lookups go through `get_api(mode).details(id, append_to_response="watch/providers")` + `_parse_providers_from_details`
- Search results show 5 at a time with a "Show more" button; remaining results are stored in `_search_more` per user
- `/pick` picks a random item from all watchlists (or a named one) that is available on the user's streaming services, with a "Pick another" inline button (`rpick` callback)
- Detail messages (`det`/`rdet` callbacks) are sent as standalone messages (not replies) via `bot.send_photo`/`bot.send_message` so Telegram clients auto-scroll to them
- `/search` without args uses `ForceReply` to prompt for input; new watchlist name entry (`nwl`/`new` callbacks) also uses `ForceReply`. Reply handler dispatches based on `_pending_search` and `_pending_new_watchlist` state. Plain text without reply defaults to search via `default_search_handler`
- Marking an item as watched sends a confirmation with the main keyboard, then a separate "Undo?" message with an inline button. The `undo` callback restores the item to its previous watchlist and rating state via `_last_watched` (which stores the mode used)

**Inline keyboard & callback system:**

All button presses route through `button_callback_handler()` using colon-delimited callback data (must fit in 64 bytes). Most actions use `action:type:id` format where type is `"m"` (movie) or `"tv"` (TV show). `parse_callback_data()` returns `(action, media_type, media_id, watchlist)`. Action prefixes:
- `pick:type:id` — show watchlist picker; `a:type:id:watchlist` — add to watchlist; `rm:type:id` — remove; `w:type:id` — mark watched (shows rating keyboard with skip option)
- `back:type:id` — return from picker to Add/Watched buttons; `new:type:id` — create new watchlist (ForceReply) and add item
- `rate:type:id:0-10` — submit rating (from search/watchlist context, 0=skip); `rrate:type:id:0-10` — submit rating (from `/rate` flow, triggers list refresh)
- `wl:<name>` — browse watchlist contents (uses current mode); `det:type:id` — show full detail card; `rdet:type:id` — show item with rating keyboard (from `/rate` list)
- `exp:<chunk_id>` / `col:<chunk_id>` — expand/collapse item button lists
- `nwl` — new watchlist from list view (ForceReply, stores mode); `wledit` / `wlback` — enter/exit edit mode; `dwl:<name>` / `dwly:<name>` / `dwln` — delete watchlist flow (uses current mode)
- `sp:<index>` — toggle streaming provider (first selection completes onboarding)
- `reg:<code>` — select region; `regp:<page>` — region picker pagination; `chreg` — change region
- `rpick:<watchlist|*>` — pick another random available item (`*` = all watchlists, uses current mode)
- `smore` — show next 5 search results (uses current mode)
- `gf:<genre_id>` — toggle genre in recommendation filter; `recgo:skip` / `recgo:filter` — launch recommendations (all genres or filtered)
- `undo` — undo last "mark as watched" action (restores watchlist placement and previous rating, uses stored mode)

**Onboarding:** New users (flagged with `onboarded: false`) get a region picker → streaming service selector flow on `/start`. Region picker is paginated with flag emojis. Onboarding completes automatically when the first service is selected.

**Message cleanup:** Search results are tracked per-user in `_search_results` and deleted after user action (add/remove/watched), with a confirmation message sent to restore `MAIN_KEYBOARD`. `_search_more` and `_pending_search` state are also cleaned up alongside search results. Rate list messages are tracked in `_rate_list_messages` and refreshed after rating via `rrate` (but not `rate` from other contexts). Previous search results are also cleaned up when a new search starts.

**Mode system:** Each user has a `"mode"` field (`"movie"` or `"tv"`). The dynamic keyboard (`get_main_keyboard(user)`) shows a toggle button. `/mode` command and the keyboard button both toggle the mode. All commands operate on the current mode's watchlists, watched history, genre dict, and API client. Callback data carries a type prefix (`"m"` or `"tv"`) so buttons remain valid regardless of the user's current mode. Data migration in `user_data_initialize()` wraps old flat structures into nested `{"movie": old_data, "tv": empty}`.

**Multi-step interactions** use `ForceReply` prompts with pending state dicts (`_pending_new_watchlist` stores `(movie_id, mode)`, `_pending_search`), handled by `reply_handler()`. Plain text input (non-reply) defaults to search via `default_search_handler()`. `/fix` restores the persistent keyboard if ForceReply causes it to disappear.

**MarkdownV2 escaping:** `esc()` uses regex to preserve `[text](url)` links (escaping text inside brackets, escaping `\` and `)` in URLs) and `` `code` `` spans, while `_esc_plain()` escapes all reserved characters in plain text. tmdbv3api's `AsObj` wrapper raises `AttributeError` instead of `KeyError` for missing keys — catch both in try/except blocks.

**Dependencies:** `python-telegram-bot`, `tmdbv3api` (v1.9.0, custom Nix build), `pyyaml`

## Bot Commands

Commands are registered with short aliases (e.g., `/search`/`/s`, `/list`/`/l`, `/recommend`/`/r`, `/pick`/`/p`, `/mode`/`/m`). The full mapping is in `main()` at the handler registration block. The dynamic persistent reply keyboard provides quick access to `/search`, `/list`, `/check`, `/recommend`, `/popular`, `/pick`, `/clear`, and the mode toggle button. `/fix` restores the keyboard if lost. Plain text without a command triggers a search. `/search` without args uses ForceReply for immediate input.
