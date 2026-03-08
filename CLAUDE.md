# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains two Telegram bots sharing a common library (`botlib/`):

- **TMDBot** — movies and TV shows using [The Movie Database (TMDb)](https://www.themoviedb.org/) API
- **BookBot** — books using the [Open Library](https://openlibrary.org/) API

Both bots provide search, watchlist management (personal and shared/collaborative), rating, recommendations, trending, person/author search, statistics, and user onboarding. TMDBot additionally has streaming provider integration, TV season tracking, and a movie/TV mode switch.

## Build & Run

This project uses **Nix flakes** for environment management with **direnv** integration.

- **Enter dev shell:** `nix develop` (or automatic via direnv)
- **Run TMDBot:** `python -m tmdbot [settings.yaml] [user_data.yaml]`
- **Run BookBot:** `python -m bookbot [settings.yaml] [user_data.yaml]`
- **Build Nix packages:** `nix build .#tmdbot` / `nix build .#bookbot`
- **Run built package:** `./result/bin/tmdbot` / `./result/bin/bookbot`
- **Format code:** `autopep8 --in-place -r tmdbot/ bookbot/ botlib/`

There are no tests in this project.

## Configuration

Each bot needs its own settings YAML (git-ignored):

**TMDBot** (`settings.yaml`):
- `telegram_token` — Telegram bot token
- `tmdb_api_key` — TMDb API key
- `allowed_users` — list of allowed Telegram user IDs

**BookBot** (`bookbot_settings.yaml`):
- `telegram_token` — Telegram bot token
- `email` — email for Open Library API identification (required for 3 req/s rate limit)
- `allowed_users` — list of allowed Telegram user IDs

User data is stored in `user_data.yaml` (auto-created). Each bot should use a separate user data file. Both files can be overridden via command line arguments.

## Architecture

### Package structure

```
botlib/                          # Shared library
├── __init__.py
├── state.py                     # All global state dicts, save/load user data
├── config.py                    # load_settings(), settings dict
├── helpers.py                   # esc(), watchlist lookups, parse_callback_data, mode converters, watched accessors
├── keyboards.py                 # Rating, watchlist, chunk, member, category keyboards (configurable labels)
├── messaging.py                 # send_back_text, send_movie_message/list, progress bar, cleanup, notifications
├── base.py                      # BaseCommand class (auth + onboarding check)
├── router.py                    # Router class — callback dispatcher
├── migration.py                 # Migration framework (run_migrations, _iter_users)
└── reply_handler.py             # ForceReply dispatch with registry for domain handlers

tmdbot/                          # Movie/TV bot
├── __init__.py                  # from tmdbot.app import main
├── __main__.py                  # from tmdbot import main; main()
├── state.py                     # sys.modules redirect → botlib.state
├── base.py                      # sys.modules redirect → botlib.base
├── router.py                    # sys.modules redirect → botlib.router
├── messaging.py                 # sys.modules redirect → botlib.messaging
├── config.py                    # TMDb API clients, genre caches, REGIONS, user_data_initialize()
├── migration.py                 # TMDb-specific migrations (uses botlib.migration.run_migrations)
├── helpers.py                   # Re-exports botlib helpers + extract_movie_info, providers, trailers, genres
├── keyboards.py                 # Re-exports botlib keyboards + get_main_keyboard, mode switch, region, services, genre, season keyboards
├── reply_handler.py             # Registers TMDb-specific pending handlers (search, person, name) with botlib
├── app.py                       # main(), post_init(), error_handler(), handler registration
└── handlers/
    ├── search.py                # SearchCommand + do_search() + smore, det, rdet, exp/col callbacks + default_search_handler
    ├── watchlist.py             # ListCommand, AddCommand, TrashAddCommand, RemoveCommand + wl, nwl, wledit, wlback, dwl/dwly/dwln + fallback handler (pick, back, a, rm, new, sa, srm, w, ws)
    ├── shared_wl.py             # nswl, smu, smd, sdwl/sdwly/sdwln, swb, swdet callbacks
    ├── watched.py               # WatchedCommand, RateCommand + rate/rrate, undo, wcat, ccat + w/ws action helpers
    ├── discovery.py             # RecommendCommand, CheckCommand, PopularCommand, PickCommand + gf, recgo, rpick, rwl callbacks
    ├── tv_seasons.py            # NewSeasonsCommand, ViewSeasonsCommand + sdet, supd + _daily_season_check
    ├── info.py                  # StatsCommand, TrendingCommand, PersonCommand + do_person_search()
    ├── onboarding.py            # StartCommand, ServicesCommand + reg, regp, chreg, sp callbacks + name reply handler
    └── misc.py                  # FixCommand, SetNameCommand, ToggleModeCommand, ClearCommand

bookbot/                         # Book bot
├── __init__.py                  # from bookbot.app import main
├── __main__.py                  # from bookbot import main; main()
├── config.py                    # Open Library API client (rate-limited requests.Session), user_data_initialize()
├── migration.py                 # Book-specific migrations
├── helpers.py                   # extract_book_info, extract_book_detail, cover URLs, work_key_to_id/id_to_work_key
├── keyboards.py                 # get_main_keyboard (no mode toggle, no /check)
├── reply_handler.py             # Registers book-specific pending handlers (search, author, name) with botlib
├── app.py                       # main(), post_init(), error_handler(), handler registration
└── handlers/
    ├── search.py                # SearchCommand + do_search() + smore, det, rdet, exp/col + default_search_handler
    ├── watchlist.py             # ListCommand, AddCommand, RemoveCommand + fallback handler
    ├── shared_wl.py             # nswl, smu, smd, sdwl/sdwly/sdwln, swb, swdet callbacks
    ├── read.py                  # ReadCommand, RateCommand + rate/rrate, undo, wcat, ccat + w action helpers
    ├── discovery.py             # RecommendCommand, PickCommand, TrendingCommand
    ├── info.py                  # StatsCommand, AuthorCommand + do_author_search()
    ├── onboarding.py            # StartCommand + name reply handler (no region/providers)
    └── misc.py                  # FixCommand, SetNameCommand, ClearCommand
```

### Key architectural patterns

**Shared library (`botlib/`):** Contains all domain-agnostic infrastructure. Both bots import from it directly. The `tmdbot/` package maintains backward-compatible imports via `sys.modules` redirects (e.g., `tmdbot/state.py` redirects to `botlib.state` so both packages share the same mutable state object).

**Registry pattern:** `botlib` uses registries to break circular dependencies with domain packages:
- `botlib.messaging.register_main_keyboard_fn(fn)` — domain provides its `get_main_keyboard`
- `botlib.keyboards.configure_labels(overrides)` — domain overrides button text (e.g., BookBot sets `"watched": "Read"`)
- `botlib.reply_handler.register_pending_handler(check_fn, handler_fn)` — domain registers ForceReply handlers

**Re-export pattern:** `tmdbot/helpers.py` and `tmdbot/keyboards.py` re-export generic functions from `botlib` (via `from botlib.helpers import ... # noqa: F401`) alongside domain-specific functions. This allows existing handler code to use `from tmdbot.helpers import esc` without change. `bookbot/` handlers import from `botlib` directly.

**Import dependency graph (no cycles):**
`botlib.state` ← `botlib.config` ← `botlib.helpers` ← `botlib.keyboards` ← `botlib.messaging` ← `botlib.base` ← `handlers/*` → `botlib.router` ← `app`

Domain packages (`tmdbot/`, `bookbot/`) import from `botlib` and add domain-specific layers.

**Mode system:** Each mode has a type prefix for callback data:
- `"movie"` ↔ `"m"`, `"tv"` ↔ `"tv"`, `"book"` ↔ `"b"`
- Converters: `_mode_to_type(mode)`, `_type_to_mode(media_type)` in `botlib.helpers`
- TMDBot users have `mode` field (`"movie"` or `"tv"`); BookBot users always use `mode="book"`
- All user data (watchlists, watched) is nested by mode

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

**Migration framework:** `botlib.migration.run_migrations(current_version, migrations)` iterates all integer-keyed user entries and runs pending migration functions. Domain packages define their own `CURRENT_VERSION` and migration list. Version stored as `_version` in `user_data.yaml`.

### State management

All mutable global state lives in `botlib.state` as module-level variables:
- `user_data` — persisted to YAML via `save_user_data()`
- Pending dicts: `_pending_new_watchlist`, `_pending_search`, `_pending_person`, `_pending_name`, `_pending_shared_wl_name`, `_pending_shared_wl_members`, `_pending_season`, `_pending_watched_category`
- Caches/tracking: `_provider_cache`, `_chunk_movies`, `_chunk_id_counter`, `_search_results`, `_search_more`, `_rate_list_messages`, `_rec_genre_filter`, `_last_watched`

### Callback system

All button presses route through `Router.__call__()` using colon-delimited callback data (must fit in 64 bytes). Most actions use `action:type:id` format where type is `"m"` (movie), `"tv"` (TV show), or `"b"` (book). `parse_callback_data()` returns `(action, media_type, media_id, watchlist)`. The fallback handler in `watchlist.py` handles `parse_callback_data`-based actions (`pick`, `back`, `a`, `rm`, `new`, `sa`, `srm`, `w`, `ws`).

Action prefixes (shared):
- `pick:type:id` — show watchlist picker; `a:type:id:watchlist` — add to watchlist; `rm:type:id` — remove; `w:type:id` — mark watched/read
- `back:type:id` — return from picker; `new:type:id` — create new watchlist (ForceReply) and add item
- `rate:type:id:0-10` — submit rating (0=skip); `rrate:type:id:0-10` — submit rating from `/rate` flow (triggers list refresh)
- `wl:<name>` — browse watchlist; `det:type:id` — detail card; `rdet:type:id` — detail with rating keyboard
- `exp:<chunk_id>` / `col:<chunk_id>` — expand/collapse item lists
- `nwl` — new watchlist; `wledit` / `wlback` — enter/exit edit mode; `dwl/dwly/dwln` — delete watchlist flow
- `wcat:<index|s>` — category picker; `ccat:type:id` — change category for watched/read item
- `rwl:<index|all>` — recommendation category picker
- `smore` — show next search results; `undo` — undo last watch/read
- Shared: `nswl`, `smu:<index>`, `smd`, `swb:<sw_id>`, `swdet:type:id`, `sa:type:id:sw_id`, `srm:type:id:sw_id`, `sdwl/sdwly/sdwln`

TMDBot-specific:
- `sp:<index>` — toggle streaming provider; `reg:<code>` / `regp:<page>` / `chreg` — region
- `gf:<genre_id>` — genre filter; `recgo:skip` / `recgo:filter` — launch recommendations
- `rpick:<watchlist|*>` — random pick; `ws:type:id:season` — season picker; `sdet:tv:id` / `supd:tv:id:season` — season detail/update

### Key patterns

- Watched/read entries are dicts: `{"rating": int|None, "category": str|None}`. Use `get_watched_rating(entry)` and `get_watched_category(entry)` from `botlib.helpers` to access fields
- Text output uses `send_back_text()` which escapes for MarkdownV2, splits long messages, and restores the persistent keyboard
- Search results use `send_movie_message()` with inline keyboard buttons; message IDs tracked in `state._search_results` for cleanup
- List views use `send_movie_list()` with chunked expand/collapse buttons
- Long operations use `_with_progress_bar(bot, chat_id, label, total, work_fn)` for background ThreadPoolExecutor work
- Marking watched/read: confirmation with main keyboard + separate "Undo?" inline button
- Category from watchlist is set automatically; category picker shown when watching/reading from search
- `ForceReply` prompts with pending state dicts, dispatched by `reply_handler()`. Plain text input defaults to search
- MarkdownV2 escaping: `esc()` preserves `[text](url)` links and `` `code` `` spans

### Shared watchlists

Stored in top-level `shared_watchlists` dict in `user_data.yaml` (keyed by numeric ID). Each entry has `name`, `owner`, `members`, `items` (nested by mode). `_shared_wl_next_id` tracks IDs. Creation: name (ForceReply) → member selection → create. Any member can add/remove; only owner can delete. Notifications via `_notify_shared_wl_members()`.

## TMDBot-specific

**Mode system:** Per-user `"mode"` field (`"movie"` or `"tv"`). Dynamic keyboard shows toggle button. Callback data carries type prefix so buttons stay valid across mode switches.

**Streaming providers:** Region-based provider setup during onboarding. `/check` checks availability, `/popular` shows popular titles on user's services.

**TV seasons:** `tv_season_counts` per-user dict. Watched flow: `[Watched]` → (category if from search) → season picker → rating. Daily `JobQueue` job checks for new seasons at 9:00 AM.

**Recommendations:** Category-aware seeding. `/recommend` without args shows category picker. Genre filtering via toggle keyboard.

**Person search:** `/person` searches actors/directors, shows top 20 results filtered by current mode.

**Dependencies:** `python-telegram-bot`, `tmdbv3api` (v1.9.0, custom Nix build), `pyyaml`

## BookBot-specific

**Open Library API:** Direct `requests` usage against `openlibrary.org` REST API. Rate-limited at 3 requests/second with `threading.Lock`. User-Agent set to `BookBot/0.1 ({email})` using email from settings.

**API functions** in `bookbot/config.py`: `ol_search()`, `ol_work()`, `ol_search_authors()`, `ol_author_works()`, `ol_trending()`, `ol_subject()`.

**Work IDs:** Open Library OLIDs like `/works/OL27482W` are converted to numeric IDs (`27482`) for callback data compatibility. Converters: `work_key_to_id()`, `id_to_work_key()` in `bookbot/helpers.py`.

**No mode toggle:** BookBot always uses `mode="book"`. No region/provider setup. Onboarding is name-entry only.

**Recommendations:** Subject-based, seeded from read books' subjects.

**Dependencies:** `python-telegram-bot`, `pyyaml`, `requests`

## Bot Commands

### TMDBot

Commands registered with aliases: `/search`/`/s`, `/list`/`/l`, `/recommend`/`/r`, `/pick`/`/p`, `/mode`/`/m`, `/newseasons`/`/ns`, `/seasons`/`/ss`, `/trending`/`/tr`, `/person`/`/ps`. Dynamic persistent keyboard provides quick access. Plain text without a command triggers search. `/fix` restores keyboard.

### BookBot

Commands registered with aliases: `/search`/`/s`, `/list`/`/l`, `/recommend`/`/r`, `/pick`/`/p`, `/trending`/`/tr`, `/author`/`/ps`. Persistent keyboard for quick access. Plain text triggers search.
