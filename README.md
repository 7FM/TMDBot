# TMDBot

A Telegram bot for discovering and managing movies using [The Movie Database (TMDb)](https://www.themoviedb.org/) API.

## Features

- **Search** movies by title with paginated results
- **Watchlists** -- create multiple named watchlists to organize movies
- **Streaming availability** -- check which watchlist movies are available on your streaming services
- **Recommendations** -- get personalized suggestions based on your watchlist and rated movies, with optional genre filtering
- **Popular movies** -- browse currently popular movies available on your services
- **Random picker** -- pick a random movie from any watchlist
- **Rating** -- rate watched movies (1-10) and browse/re-rate your history
- **Onboarding** -- region and streaming service setup for new users

## Setup

### Prerequisites

- Python 3.11+
- A [Telegram Bot Token](https://core.telegram.org/bots#botfather)
- A [TMDb API Key](https://developer.themoviedb.org/docs/getting-started)

### Using Nix (recommended)

This project uses [Nix flakes](https://nixos.wiki/wiki/Flakes) with [direnv](https://direnv.net/) integration. All dependencies including a custom build of `tmdbv3api` v1.9.0 are managed by Nix.

```sh
# Enter dev shell (or automatic via direnv)
nix develop

# Run the bot
python tmdbot.py
```

To build and run as a Nix package:

```sh
nix build
./result/bin/tmdbot
```

### Manual setup

```sh
pip install python-telegram-bot tmdbv3api pyyaml
python tmdbot.py
```

### Configuration

Create a `settings.yaml` in the project root:

```yaml
telegram_token: "YOUR_TELEGRAM_BOT_TOKEN"
tmdb_api_key: "YOUR_TMDB_API_KEY"
allowed_users:
  - 123456789  # Telegram user IDs allowed to use the bot
```

User data (watchlists, providers, watched history, ratings) is stored in `user_data.yaml`, which is created automatically on first run.

Both files can be overridden via command line arguments:

```sh
python tmdbot.py [settings.yaml] [user_data.yaml]
```

## Bot Commands

| Command | Alias | Description |
|---------|-------|-------------|
| `/start` | | Initial setup / welcome message |
| `/search` | `/s` | Search for movies by title |
| `/list` | `/l` | Browse your watchlists |
| `/add` | `/a` | Add a movie by ID to a watchlist |
| `/tadd` | `/t` | Add a movie to the trash watchlist |
| `/watched` | `/w` | Mark a movie as watched |
| `/remove` | `/rm` | Remove a movie from all watchlists |
| `/rate` | | Rate or re-rate watched movies |
| `/services` | | Manage your streaming services |
| `/check` | `/c` | Check streaming availability for watchlist movies |
| `/recommend` | `/r` | Get recommendations based on a watchlist |
| `/popular` | `/pop` | Show popular movies on your streaming services |
| `/pick` | `/p` | Pick a random movie from a watchlist |
