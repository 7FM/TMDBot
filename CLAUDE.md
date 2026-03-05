# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TMDBot is a Telegram bot for discovering and managing movies and TV shows using The Movie Database (TMDb) API. It provides search, watchlist management (personal and shared/collaborative), streaming provider integration, rating, recommendation features, random picker, new season detection, person search, trending titles, watch statistics, and user onboarding — with a per-user mode switch between Movies and TV.

## Build & Run

This project uses **Nix flakes** for environment management with **direnv** integration.

- **Enter dev shell:** `nix develop` (or automatic via direnv)
- **Run directly:** `python -m tmdbot [settings.yaml] [user_data.yaml]`
- **Build Nix package:** `nix build`
- **Run built package:** `./result/bin/tmdbot`
- **Format code:** `autopep8 --in-place -r tmdbot/`

There are no tests in this project.

## Configuration

Two YAML files (git-ignored) are required at runtime:
- `settings.yaml` — Telegram bot token, TMDb API key, allowed user IDs
- `user_data.yaml` — Per-user state (mode, name, watchlists nested by mode, providers, watched history with ratings nested by mode, region, onboarded flag, `tv_season_counts` for season tracking) plus top-level `shared_watchlists` dict and `_shared_wl_next_id` counter; auto-created if missing

## Architecture

**Package structure:** The bot is organized as a Python package (`tmdbot/`):

```
tmdbot/
├── __init__.py          # from tmdbot.app import main
├── __main__.py          # from tmdbot import main; main()
├── config.py            # Settings, TMDb API client init (deferred via init()), regions, genre caches, user_data_initialize()
├── state.py             # All global state dicts (mutable module-level), save/load user data
├── helpers.py           # Text escaping, media info extraction, provider logic, watchlist lookups, parse_callback_data, mode converters
├── keyboards.py         # All build_*_keyboard() functions, get_main_keyboard()
├── messaging.py         # send_back_text, send_movie_message, send_movie_list, progress bar, cleanup helpers, _notify_shared_wl_members
├── base.py              # BaseCommand class (auth + onboarding check, delegates to execute())
├── router.py            # Router class — callback dispatcher with action→handler table and onboarding exemptions
├── reply_handler.py     # reply_handler() for ForceReply dispatch (delegates to handler modules)
├── handlers/
│   ├── __init__.py
│   ├── search.py        # SearchCommand + do_search() + smore, det, rdet, exp/col callbacks + default_search_handler
│   ├── watchlist.py     # ListCommand, AddCommand, TrashAddCommand, RemoveCommand + wl, nwl, wledit, wlback, dwl/dwly/dwln callbacks + fallback handler (pick, back, a, rm, new, sa, srm, w, ws)
│   ├── shared_wl.py     # nswl, smu, smd, sdwl/sdwly/sdwln, swb, swdet callbacks (no command — all callback-driven from /list)
│   ├── watched.py       # WatchedCommand, RateCommand + rate/rrate, undo callbacks + w/ws action helpers
│   ├── discovery.py     # RecommendCommand, CheckCommand, PopularCommand, PickCommand + gf, recgo, rpick callbacks
│   ├── tv_seasons.py    # NewSeasonsCommand, ViewSeasonsCommand + sdet, supd callbacks + _daily_season_check
│   ├── info.py          # StatsCommand, TrendingCommand, PersonCommand + do_person_search()
│   ├── onboarding.py    # StartCommand, ServicesCommand + reg, regp, chreg, sp callbacks + name reply handler
│   └── misc.py          # FixCommand, SetNameCommand, ToggleModeCommand, ClearCommand
└── app.py               # main(), post_init(), error_handler(), handler registration loop
```

**Key layers:**

1. **`config.py`**: TMDb API clients and settings are initialized lazily via `config.init(settings_file, user_data_file)` called from `main()`. No import-time side effects.
2. **`state.py`**: All mutable global state (pending dicts, caches, user data) as module-level variables. Mutations visible across all importers. `next_chunk_id()` helper for the integer counter.
3. **`helpers.py`**: Pure utility functions — text escaping (`esc()`), media info extraction, provider logic, watchlist lookup helpers, callback data parsing, mode converters.
4. **`keyboards.py`**: All `build_*_keyboard()` functions and `get_main_keyboard()`.
5. **`messaging.py`**: `send_back_text()`, `send_movie_message()`, `send_movie_list()`, progress bar, cleanup helpers, notification helper.
6. **`base.py`**: `BaseCommand` class with `__call__(update, context)` → auth check → onboarding check → `execute(update, context, user)`. Set `require_onboarding = False` to skip the onboarding gate.
7. **`router.py`**: `Router` class holds action→handler dispatch table. Handler modules populate it via `register()`. Acts as the `CallbackQueryHandler` callable. Supports `add_onboarding_action()` for pre-onboarding callbacks.
8. **Handler modules**: Each exports a `register(app, router)` function that registers its own `CommandHandler`s and `router.add()` calls. Self-contained per feature area.
9. **`app.py`**: Thin orchestrator — calls `config.init()`, creates `Application`, iterates `_HANDLER_MODULES` calling `register()`, adds global handlers, starts polling.

**Import dependency graph (no cycles):**
`state` ← `config` ← `helpers` ← `keyboards` ← `messaging` ← `base` ← `handlers/*` → `router` ← `app`

**BaseCommand pattern:**

```python
class SearchCommand(BaseCommand):
    async def execute(self, update, context, user):
        # ... no boilerplate auth/onboarding check needed

class StartCommand(BaseCommand):
    require_onboarding = False  # allows pre-onboarding access
    async def execute(self, update, context, user):
        # ...
```

Registration: `CommandHandler(['search', 's'], SearchCommand())` — works because `__call__` matches the `(update, context)` signature.

**Self-registration pattern:**

```python
# handlers/search.py
def register(app, router):
    app.add_handler(CommandHandler(['search', 's'], SearchCommand()))
    router.add('smore', handle_smore)
    router.add('det', handle_detail)
```

Adding a new feature = adding one handler file + one entry in `_HANDLER_MODULES` in `app.py`.

**Callback routing:** The `Router` in `router.py` dispatches by action prefix. The fallback handler in `watchlist.py` handles `parse_callback_data`-based actions (`pick`, `back`, `a`, `rm`, `new`, `sa`, `srm`, `w`, `ws`).

**Key patterns (unchanged from before refactoring):**

- User state is `state.user_data` dict persisted to YAML via `state.save_user_data()`
- `user_data_initialize()` in `config.py` handles data migrations
- Text output uses `send_back_text()` which escapes for MarkdownV2, splits long messages, and always restores the persistent keyboard
- Movie search results use `send_movie_message()` which attaches inline keyboard buttons (Add/Remove/Watched); search result message IDs are tracked in `state._search_results` for cleanup after user action
- List-based views use `send_movie_list(bot, chat_id, ...)` which produces chunked itemized lists with expand/collapse buttons
- `extract_movie_info(m, skip_trailer=False, mode="movie")` — handles both movie and TV fields
- `/check` and `/recommend` use `_with_progress_bar(bot, chat_id, label, total, work_fn)` for background ThreadPoolExecutor work with live progress bar
- `/popular` uses `ThreadPoolExecutor` for parallel provider lookups with a simple status message
- All provider lookups go through `get_api(mode).details(id, append_to_response="watch/providers")` + `_parse_providers_from_details`
- Search results show 5 at a time with a "Show more" button; remaining results are stored in `state._search_more` per user
- Marking an item as watched sends a confirmation with the main keyboard, then a separate "Undo?" message with an inline button
- For TV shows, the watched flow adds a season picker step: `[Watched]` → season picker (`ws:` callback) → rating keyboard → save
- `/newseasons` (`/ns`) checks all watched TV shows for new seasons. A daily `JobQueue` job (`_daily_season_check`) runs at 9:00 AM

**Inline keyboard & callback system:**

All button presses route through `Router.__call__()` using colon-delimited callback data (must fit in 64 bytes). Most actions use `action:type:id` format where type is `"m"` (movie) or `"tv"` (TV show). `parse_callback_data()` returns `(action, media_type, media_id, watchlist)`. Action prefixes:
- `pick:type:id` — show watchlist picker; `a:type:id:watchlist` — add to watchlist; `rm:type:id` — remove; `w:type:id` — mark watched (for TV: shows season picker then rating; for movies: shows rating keyboard with skip option)
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
- `ws:type:id:season` — season picked in TV watched flow (then shows rating keyboard)
- `sdet:tv:id` — show season detail from `/seasons` list (shows season picker to update)
- `supd:tv:id:season` — update watched season from `/seasons` view
- `undo` — undo last "mark as watched" action (restores watchlist placement, previous rating, and season data, uses stored mode)
- Shared watchlist callbacks: `nswl` — new shared watchlist (ForceReply for name, then member selection); `smu:<index>` — toggle member; `smd` — done selecting members; `swb:<sw_id>` — browse shared watchlist; `swdet:type:id` — detail card from shared context; `sa:type:id:sw_id` — add to shared watchlist; `srm:type:id:sw_id` — remove from shared watchlist; `sdwl:<sw_id>` / `sdwly:<sw_id>` / `sdwln` — delete shared watchlist flow (owner only)

**Onboarding:** New users (flagged with `onboarded: false`) get a region picker → **display name prompt (ForceReply)** → streaming service selector flow on `/start`. Region picker is paginated with flag emojis. Onboarding completes automatically when the first service is selected.

**Message cleanup:** Search results are tracked per-user in `state._search_results` and deleted after user action (add/remove/watched), with a confirmation message sent to restore `MAIN_KEYBOARD`. `state._search_more` and `state._pending_search` state are also cleaned up alongside search results. Rate list messages are tracked in `state._rate_list_messages` and refreshed after rating via `rrate` (but not `rate` from other contexts). Previous search results are also cleaned up when a new search starts.

**Mode system:** Each user has a `"mode"` field (`"movie"` or `"tv"`). The dynamic keyboard (`get_main_keyboard(user)`) shows a toggle button. `/mode` command and the keyboard button both toggle the mode. All commands operate on the current mode's watchlists, watched history, genre dict, and API client. Callback data carries a type prefix (`"m"` or `"tv"`) so buttons remain valid regardless of the user's current mode. Data migration in `user_data_initialize()` wraps old flat structures into nested `{"movie": old_data, "tv": empty}`.

**Shared watchlists:** Stored in a top-level `shared_watchlists` dict in `user_data.yaml` (keyed by numeric ID, not nested under users). Each entry has `name`, `owner` (user ID), `members` (list of user IDs), and `items` (nested by mode like personal watchlists). `_shared_wl_next_id` tracks the next available ID. Helper functions: `_get_shared_wl()`, `_next_shared_wl_id()`, `_user_shared_watchlists()`, `_get_user_display_name()`, `is_in_any_shared_watchlist()`, `find_all_shared_watchlists()`. Creation flow: name (ForceReply) → member selection (toggle keyboard like streaming services) → create. Any member can add/remove items; only owner can delete. Marking watched is personal (item stays in shared list). Notifications via `_notify_shared_wl_members()` are sent when items are added/removed/watched. When an item is removed, members who haven't watched it get a "Watched" button. `/list` shows both personal and shared (prefixed with 👥) watchlists. The watchlist picker when adding items also includes shared watchlists. `/setname` lets users set/change their display name.

**Multi-step interactions** use `ForceReply` prompts with pending state dicts (`state._pending_new_watchlist` stores `(movie_id, mode)`, `state._pending_search`, `state._pending_person`, `state._pending_name`, `state._pending_shared_wl_name`), handled by `reply_handler()` in `reply_handler.py`. Plain text input (non-reply) defaults to search via `default_search_handler()`. `/fix` restores the persistent keyboard if ForceReply causes it to disappear.

**MarkdownV2 escaping:** `esc()` uses regex to preserve `[text](url)` links (escaping text inside brackets, escaping `\` and `)` in URLs) and `` `code` `` spans, while `_esc_plain()` escapes all reserved characters in plain text. tmdbv3api's `AsObj` wrapper raises `AttributeError` instead of `KeyError` for missing keys — catch both in try/except blocks.

**Dependencies:** `python-telegram-bot`, `tmdbv3api` (v1.9.0, custom Nix build), `pyyaml`

## Bot Commands

Commands are registered with short aliases (e.g., `/search`/`/s`, `/list`/`/l`, `/recommend`/`/r`, `/pick`/`/p`, `/mode`/`/m`, `/newseasons`/`/ns`, `/seasons`/`/ss`, `/trending`/`/tr`, `/person`/`/ps`). Each handler module's `register()` function adds its own commands. The dynamic persistent reply keyboard provides quick access to `/search`, `/list`, `/check`, `/recommend`, `/popular`, `/pick`, `/clear`, and the mode toggle button. `/fix` restores the keyboard if lost. Plain text without a command triggers a search. `/search` and `/person` without args use ForceReply for immediate input. `/setname <name>` sets or changes the user's display name (used in shared watchlist notifications and member selection).

**Season tracking:** `tv_season_counts` is a per-user dict mapping TV show media IDs to `{"total": int, "watched": int}`. `total` is the last-known number of seasons from TMDb; `watched` is the season the user watched up to. Populated automatically when marking a TV show as watched (season picker step) and checked by `/newseasons` and the daily job.

**Stats:** `/stats` shows watch statistics for the current mode — total watched count, rated/unrated counts, average rating, rating distribution (bar chart), and top genres. Genre fetching uses `_with_progress_bar` since it requires per-item API calls.

**Trending:** `/trending` (alias `/tr`) shows trending titles for the current mode using TMDb's `Trending` API (`trending.movie_day()` / `trending.tv_day()`). Displays 10 results via `send_movie_list()`, filters out already-watched items.

**Person search:** `/person` (alias `/ps`) searches for an actor/director via `search.people()`, fetches their `combined_credits` via `person_api`, filters by current mode, sorts by rating, and displays top 20 results via `send_movie_list()`. Uses `state._pending_person` state for ForceReply when called without args.

**Duplicate warning:** When adding an already-watched item to a watchlist, the warning includes the user's previous rating (e.g., "rated 7/10") if one exists.
