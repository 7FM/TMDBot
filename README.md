# TMDBot & BookBot

Two Telegram bots sharing a common codebase (`botlib/`) -- one for movies/TV, one for books.

## TMDBot

A Telegram bot for discovering and managing movies and TV shows using [The Movie Database (TMDb)](https://www.themoviedb.org/) API.

### Features

- **Movies & TV** -- per-user mode switch between Movies and TV Shows, with separate watchlists for each
- **Search** by title with paginated results
- **Watchlists** -- create multiple named watchlists to organize titles
- **Shared watchlists** -- collaborative lists shared with other users
- **Streaming availability** -- check which watchlist items are available on your streaming services
- **Recommendations** -- category-aware suggestions based on your watchlist and rated items, with optional genre filtering
- **Popular titles** -- browse currently popular movies/shows available on your services
- **Random picker** -- pick a random available title from any watchlist
- **Rating** -- rate watched titles (1-10) and browse/re-rate your history
- **Trending** -- browse today's trending movies/shows
- **Person search** -- search by actor/director and browse their filmography
- **TV seasons** -- track new seasons for watched shows, with daily automatic checks
- **Stats** -- view your watch statistics with rating distribution and top genres
- **Onboarding** -- region and streaming service setup for new users

### Commands

| Command | Alias | Description |
|---------|-------|-------------|
| `/start` | | Initial setup / welcome message |
| `/search` | `/s` | Search for movies/shows by title |
| `/list` | `/l` | Browse your watchlists |
| `/add` | `/a` | Add a title by ID to a watchlist |
| `/tadd` | `/t` | Add a title to the trash watchlist |
| `/watched` | `/w` | Mark a title as watched |
| `/remove` | `/rm` | Remove a title from all watchlists |
| `/rate` | | Rate or re-rate watched titles |
| `/services` | | Manage your streaming services |
| `/check` | `/c` | Check streaming availability |
| `/recommend` | `/r` | Get recommendations based on a watchlist |
| `/popular` | `/pop` | Show popular titles on your services |
| `/pick` | `/p` | Pick a random title from a watchlist |
| `/mode` | `/m` | Switch between Movies and TV mode |
| `/trending` | `/tr` | Browse trending titles today |
| `/person` | `/ps` | Search by actor/director |
| `/newseasons` | `/ns` | Check for new TV seasons |
| `/seasons` | `/ss` | View your tracked seasons |
| `/stats` | | View your watch statistics |
| `/setname` | | Set your display name |
| `/fix` | | Restore the keyboard if lost |
| `/clear` | | Clear the chat |

## BookBot

A Telegram bot for discovering and managing books using the [Open Library](https://openlibrary.org/) API.

### Features

- **Search** books by title with paginated results
- **Reading lists** -- create multiple named lists to organize books
- **Shared reading lists** -- collaborative lists shared with other users
- **Mark as read** -- track read books with ratings and categories
- **Recommendations** -- subject-based suggestions seeded from your read books
- **Trending** -- browse currently trending books
- **Author search** -- search by author and browse their works
- **Random picker** -- pick a random book from your lists
- **Stats** -- view your reading statistics with rating distribution and top subjects
- **Rating** -- rate read books (1-10) and browse/re-rate your history

### Commands

| Command | Alias | Description |
|---------|-------|-------------|
| `/start` | | Get started / set your display name |
| `/search` | `/s` | Search for books by title |
| `/list` | `/l` | Browse your reading lists |
| `/add` | `/a` | Add a book by ID to a reading list |
| `/read` | `/w` | Mark a book as read |
| `/rate` | | Rate or re-rate read books |
| `/recommend` | `/r` | Get book recommendations |
| `/trending` | `/tr` | Browse trending books today |
| `/pick` | `/p` | Pick a random book from a list |
| `/author` | `/ps` | Search by author |
| `/stats` | | View your reading statistics |
| `/setname` | | Set your display name |
| `/fix` | | Restore the keyboard if lost |
| `/clear` | | Clear the chat |

## Setup

### Prerequisites

- Python 3.11+
- A [Telegram Bot Token](https://core.telegram.org/bots#botfather) (one per bot)
- A [TMDb API Key](https://developer.themoviedb.org/docs/getting-started) (TMDBot only)

### Using Nix (recommended)

This project uses [Nix flakes](https://nixos.wiki/wiki/Flakes) with [direnv](https://direnv.net/) integration.

```sh
# Enter dev shell (or automatic via direnv)
nix develop

# Run TMDBot
python -m tmdbot [settings.yaml] [user_data.yaml]

# Run BookBot
python -m bookbot [settings.yaml] [user_data.yaml]
```

To build and run as Nix packages:

```sh
nix build .#tmdbot
./result/bin/tmdbot

nix build .#bookbot
./result/bin/bookbot
```

### Manual setup

```sh
pip install python-telegram-bot tmdbv3api pyyaml requests
python -m tmdbot   # or python -m bookbot
```

### Configuration

Each bot needs its own `settings.yaml` (separate Telegram bot tokens). Create one in the project root:

**TMDBot** (`settings.yaml`):

```yaml
telegram_token: "YOUR_TELEGRAM_BOT_TOKEN"
tmdb_api_key: "YOUR_TMDB_API_KEY"
allowed_users:
  - 123456789  # Telegram user IDs allowed to use the bot
```

**BookBot** (`bookbot_settings.yaml`):

```yaml
telegram_token: "YOUR_BOOKBOT_TELEGRAM_TOKEN"
email: "your@email.com"  # Required for Open Library API identification
allowed_users:
  - 123456789
```

User data is stored in `user_data.yaml` (auto-created on first run). Each bot should use a separate user data file:

```sh
python -m tmdbot settings.yaml user_data.yaml
python -m bookbot bookbot_settings.yaml bookbot_user_data.yaml
```

### Hooks

Both bots support an optional `on_add_script` that runs whenever an item is added to a watchlist (personal or shared). Add it to your settings:

```yaml
on_add_script: "/path/to/your/script.sh"
```

The script receives metadata via environment variables:

| Variable | Bots | Description |
|----------|------|-------------|
| `MEDIA_ID` | both | TMDb ID or Open Library work ID |
| `MODE` | both | `movie`, `tv`, or `book` |
| `USER_ID` | both | Telegram user ID |
| `WATCHLIST` | both | Watchlist name |
| `MEDIA_TYPE` | both | `movie`, `tv`, or `book` |
| `TITLE` | both | Title |
| `YEAR` | TMDBot | Release year |
| `AUTHOR` | BookBot | Author name(s) |
| `ISBN` | BookBot | ISBN-13 (or ISBN-10 fallback) |

## Architecture

```
TMDBot/
├── botlib/              # Shared library (state, keyboards, messaging, routing, etc.)
├── tmdbot/              # Movie/TV bot (TMDb API)
│   ├── app.py           # Entry point, handler registration
│   ├── config.py        # TMDb API clients, genre caches
│   ├── helpers.py       # Movie/TV info extraction, providers, trailers
│   ├── keyboards.py     # Mode switch, region, services, genre, season keyboards
│   └── handlers/        # search, watchlist, watched, discovery, tv_seasons, info, ...
├── bookbot/             # Book bot (Open Library API)
│   ├── app.py           # Entry point, handler registration
│   ├── config.py        # Open Library API client with rate limiting
│   ├── helpers.py       # Book info extraction, cover URLs
│   ├── keyboards.py     # Main keyboard (no mode toggle)
│   └── handlers/        # search, watchlist, read, discovery, info, ...
├── setup.py
└── flake.nix
```

The shared `botlib/` package provides: state management, YAML persistence, callback routing, base command class, keyboard builders, messaging helpers, watchlist mechanics, migration framework, and reply handler dispatch. Both bots import from it, keeping domain-specific logic in their own packages.
