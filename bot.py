import os
import yaml
from telegram import Update, MenuButtonCommands
from telegram.ext import Application, CommandHandler, CallbackContext, ContextTypes
from telegram.constants import ParseMode
from tmdbv3api import TMDb, Movie, Search, Genre, Provider


def load_settings(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


SETTINGS_FILE = "settings.yaml"
settings = load_settings(SETTINGS_FILE)

# Set up TMDb API
tmdb = TMDb()
tmdb.api_key = settings["tmdb_api_key"]
movie = Movie()
search = Search()
genre = Genre()
provider = Provider()

# File path for storing the user data
USER_DATA_FILE = 'user_data.yaml'
user_data = dict()
region = "DE"


def save_user_data():
    with open(USER_DATA_FILE, 'w') as file:
        yaml.safe_dump(user_data, file)


def user_data_initialize():
    for user in settings['allowed_users']:
        if user not in user_data:
            user_data[user] = dict()
            user_data[user]["region"] = region
            user_data[user]["watched"] = []
            user_data[user]["watchlists"] = dict()
            user_data[user]["watchlists"]["normal"] = []
            user_data[user]["watchlists"]["trash"] = []
            user_data[user]["providers"] = [
                "Netflix", "Amazon Prime Video", "Disney"]
    save_user_data()


# Load user data from file if it exists
if os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, 'r') as file:
        user_data = yaml.safe_load(file)
user_data_initialize()


def get_user_id(update: Update):
    return update.message.from_user.id


def check_user_invalid(user):
    return user not in settings['allowed_users']


def get_movie_genres():
    genre_dict = dict()
    movie_genres = genre.movie_list()["genres"]
    for mg in movie_genres:
        genre_dict[mg["id"]] = mg["name"]
    return genre_dict


genre_dict = get_movie_genres()

# Helper functions


def get_image_url(path):
    if path:
        return f"https://image.tmdb.org/t/p/original{path}"
    return None


def get_all_movie_provider(region):
    movie_provider = provider.movie_providers(region=region)
    if movie_provider:
        movie_provider = movie_provider["results"]
        return [mp["provider_name"] for mp in movie_provider]
    return []


# TODO JustWatch Attribution Required
def get_free_provider(id, country_code):
    watch_providers = movie.watch_providers(id)
    free_provider = None
    if watch_providers:
        watch_providers = watch_providers["results"]
        for w in watch_providers:
            if country_code == w["results"]:
                provider = w[country_code]
                for p in provider:
                    if not isinstance(p, str) and "flatrate" == p[country_code]:
                        free_provider = []
                        for p in p["flatrate"]:
                            free_provider.append(
                                (p["provider_name"], get_image_url(p['logo_path'])))
                break
    return free_provider


def is_available_for_free(my_providers, id, country_code):
    provider = get_free_provider(id, country_code)
    available = []
    if provider:
        for p, logo in provider:
            for mp in my_providers:
                if p.startswith(mp):
                    available.append((p, logo))
    return len(available) > 0, available


def create_available_at_str(provider):
    return "Available at: " + (", ".join([p[0] for p in provider]))


def extract_trailer_url(m):
    if "trailers" not in m:
        m = movie.details(m["id"], append_to_response="trailers")
    if "trailers" in m and "youtube" in m["trailers"]:
        for t in m["trailers"]["youtube"]:
            if t["type"] == "Trailer":
                return f'https://www.youtube.com/watch?v={t["source"]}'

    return None


def extract_genre(m):
    global genre_dict
    genre = []
    if "genre_ids" in m:
        for g in m["genre_ids"]:
            if g not in genre_dict:
                genre_dict = get_movie_genres()
            genre.append(genre_dict[g])
    elif "genres" in m:
        for g in m["genres"]:
            genre.append(g["name"])
    return genre


def esc(s):
    # Escape string for weird telegram markdown v2 parser
    return s.replace("-", "\\-").replace(".", "\\.").replace("=", "\\=").replace("!", "\\!")


def extract_movie_info(m):
    title = m["title"]
    poster_path = get_image_url(m['poster_path'])
    popularity = m["popularity"] if "popularity" in m else -1
    rating = m["vote_average"] if "vote_average" in m and m["vote_count"] > 0 else None
    release_date = m["release_date"] if "release_date" in m and m["release_date"] != "" else None
    id = m["id"]
    genre = extract_genre(m)
    trailer = extract_trailer_url(m)
    if trailer:
        title = f'[{title}]({trailer})'
    else:
        title = f'`{title}`'
    return (popularity, poster_path, title + (" - " + release_date if release_date else "") + (" - " + (", ".join(genre)) if genre else "") + ' - ' + (str(round(rating, 1)) if rating else "?") + '/10 - id=`' + str(id) + '`')


def sort_by_popularity(movie_list):
    return sorted(movie_list, key=lambda x: x[0], reverse=True)


def is_in_any_watchlist(movie_id, user):
    for wn, w in user_data[user]["watchlists"].items():
        if movie_id in w:
            return wn
    return None


# Define command handlers
async def unauthorized_msg(update: Update) -> None:
    user_id = get_user_id(update)
    await update.message.reply_text(esc(f'*Unauthorized user detected!*\nPlease contact the bot admin to whitelist your user id = `{user_id}`.\nOtherwise, consider hosting your own bot instance. The source code is publicly available at [GitHub](https://github.com/7FM/TMDBot).'), parse_mode=ParseMode.MARKDOWN_V2)


async def show_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    def lookup_movies_in_watchlist(watchlist):
        movies = []
        for movie_id in watchlist:
            movie_details = movie.details(movie_id)
            _, poster, desc = extract_movie_info(movie_details)
            movies.append(desc)

    reply_text = ""
    nl = "\n"
    for wn, w in user_data[user]["watchlists"].items():
        reply_text.append(
            f"{'' if reply_text == '' else nl + nl}{wn} watchlist:{nl}")
        reply_text.append("\n".join(lookup_movies_in_watchlist(w)))

    await update.message.reply_text(esc(reply_text), parse_mode=ParseMode.MARKDOWN_V2)


async def show_my_providers(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    provider = [f'`{mp}`' for mp in user_data[user]["providers"]]
    reply_text = "Your selected streaming services:\n" + "\n".join(provider)

    await update.message.reply_text(esc(reply_text), parse_mode=ParseMode.MARKDOWN_V2)


async def add_provider(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not context.args:
        movie_provider = [f'`{mp}`' for mp in get_all_movie_provider(
            user_data[user]["region"])]
        await update.message.reply_text(esc(f'Please provide a stream service.\nHere is a list of all available services in the region {user_data[user]["region"]}:\n' + "\n".join(movie_provider)), parse_mode=ParseMode.MARKDOWN_V2)
        return

    provider = ' '.join(context.args)
    if provider in user_data[user]["providers"]:
        await update.message.reply_text("Streaming service is in your provider list")
    else:
        user_data[user]["providers"].append(provider)
        save_user_data()
        await update.message.reply_text("Streaming service was added")


async def rm_provider(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not context.args:
        movie_provider = [f'`{mp}`' for mp in user_data[user]["providers"]]
        await update.message.reply_text(esc(f'Please provide a stream service.\nHere is a list of your selected services:\n' + "\n".join(movie_provider)), parse_mode=ParseMode.MARKDOWN_V2)
        return

    provider = ' '.join(context.args)
    if provider in user_data[user]["providers"]:
        user_data[user]["providers"].remove(provider)
        save_user_data()
        await update.message.reply_text("Streaming service was removed from the provider list")
    else:
        movie_provider = [f'`{mp}`' for mp in user_data[user]["providers"]]
        await update.message.reply_text(esc(f'Streaming service is not in your provider list!\nHere is a list of your selected services:\n' + "\n".join(movie_provider)), parse_mode=ParseMode.MARKDOWN_V2)


async def add_to_watchlist_helper(watchlist, movie_id, user, update: Update):
    if not movie_id.isdigit():
        await update.message.reply_text(f'The provided movie id is not a number!')
        # TODO check that the id is actually valid! aka tmdb knows it
        return
    movie_id = int(movie_id)
    # Check if the movie is already in the watchlist
    already_in = is_in_any_watchlist(movie_id, user)
    if already_in:
        await update.message.reply_text(f'This movie is already in your "{already_in}" watchlist.')
    else:
        user_data[user]["watchlists"][watchlist].append(movie_id)
        save_user_data()
        await update.message.reply_text('Movie added to watchlist.')


async def add_to_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not context.args:
        await update.message.reply_text('Please provide the movie ID.')
        return

    watchlist = "normal"
    if len(context.args) == 2:
        watchlist = context.args[1]
    if watchlist not in user_data[user]["watchlists"]:
        await update.message.reply_text(f'Info: creating new watchlist "{watchlist}"')
        user_data[user]["watchlists"][watchlist] = []

    movie_id = context.args[0]
    await add_to_watchlist_helper(watchlist, movie_id, user, update)


async def add_to_trash_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not context.args:
        await update.message.reply_text('Please provide the movie ID.')
        return

    movie_id = context.args[0]
    await add_to_watchlist_helper("trash", movie_id, user, update)


async def add_to_watched(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not context.args:
        await update.message.reply_text('Please provide the movie ID.')
        return

    movie_id = context.args[0]
    if not movie_id.isdigit():
        await update.message.reply_text(f'The provided movie id is not a digit!')
        # TODO check that the id is actually valid! aka tmdb knows it
        return
    movie_id = int(movie_id)
    for _, w in user_data[user]["watchlists"].items():
        if movie_id in w:
            w.remove(movie_id)

    if movie_id not in user_data[user]["watched"]:
        user_data[user]["watched"].append(movie_id)
        save_user_data()
        await update.message.reply_text('Movie marked as watched.')
    else:
        await update.message.reply_text('Movie was already marked as watched.')


async def remove_from_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    if not context.args:
        await update.message.reply_text('Please provide the movie ID.')
        return

    movie_id = context.args[0]
    if not movie_id.isdigit():
        await update.message.reply_text(f'The provided movie id is not a digit!')
        # TODO check that the id is actually valid! aka tmdb knows it
        return
    movie_id = int(movie_id)
    removed_smth = False
    for _, w in user_data[user]["watchlists"].items():
        if movie_id in w:
            w.remove(movie_id)
            removed_smth = True
    if removed_smth:
        save_user_data()
        await update.message.reply_text('Movie removed from watchlist.')
    else:
        await update.message.reply_text('This movie is not in your watchlist.')


async def recommend(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    watchlist = "normal"
    if context.args:
        watchlist = context.args[0]

    if not user_data[user]["watchlists"][watchlist]:
        await update.message.reply_text(f'Your "{watchlist}" watchlist is empty.')
        return

    message = await update.message.reply_text('THIS WILL TAKE A WHILE! Lay back and wait c:')
    last_message_id = message.message_id

    recommended_movies = []
    available_recommendations = []
    counter = 0
    watchlist_size = len(user_data[user]["watchlists"][watchlist])
    for movie_id in user_data[user]["watchlists"][watchlist]:
        results = movie.recommendations(movie_id)
        movies = results["results"]
        # TODO parallelize this loop?
        for m in movies:
            in_watchlist = is_in_any_watchlist(m["id"], user)
            if not in_watchlist and m["id"] not in user_data[user]["watched"] and m["id"] not in recommended_movies:
                recommended_movies.append(m["id"])
                available, provider = is_available_for_free(
                    user_data[user]["providers"], m["id"], user_data[user]["region"])
                if available:
                    popularity, poster, desc = extract_movie_info(m)
                    available_recommendations.append(
                        (popularity, poster, desc, provider))
        counter += 1
        # Edit the message by sending a new message with the same chat and message ID
        await context.bot.edit_message_text(chat_id=update.effective_chat.id,
                                            message_id=last_message_id,
                                            text=f'Progress: processed {counter}/{watchlist_size} watchlist movies.\nFound recommendations: {len(available_recommendations)}')

    available_recommendations = sort_by_popularity(available_recommendations)
    num_rec = min(50, len(available_recommendations))
    if available_recommendations:
        await update.message.reply_text(f'Recommended movies based on your "{watchlist}" watchlist:')
        # Giving all the recommendations lol
        for _, poster_path, caption, provider in available_recommendations[:num_rec]:
            provider_str = create_available_at_str(provider)
            if poster_path:
                await update.message.reply_photo(poster_path, esc(caption + "\n" + provider_str), parse_mode=ParseMode.MARKDOWN_V2)
            else:
                await update.message.reply_text(esc(caption + "\n" + provider_str), parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_text(f'No recommendations found based on your "{watchlist}" watchlist.')


async def check_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    def collect_available(watchlist, my_providers):
        movies = []
        for movie_id in watchlist:
            avail, provider = is_available_for_free(
                my_providers, movie_id, user_data[user]["region"])
            if avail:
                movies.append((movie_id, provider))
        return movies

    my_providers = user_data[user]["providers"]
    available_movies = []
    for wn, w in user_data[user]["watchlists"].items():
        available_movies.append((wn, collect_available(w, my_providers)))

    async def print_available(movies, prefix):
        if movies:
            await update.message.reply_text(f'Movies on your {prefix} watchlist available on streaming services:')
            for movie_id, provider in movies:
                movie_details = movie.details(movie_id)
                _, poster, desc = extract_movie_info(movie_details)
                provider_str = create_available_at_str(provider)
                if poster:
                    await update.message.reply_photo(poster, esc(desc + "\n" + provider_str), parse_mode=ParseMode.MARKDOWN_V2)
                else:
                    await update.message.reply_text(esc(desc + "\n" + provider_str), parse_mode=ParseMode.MARKDOWN_V2)

        else:
            await update.message.reply_text(f'None of the movies on your {prefix} watchlist are available on streaming services.')

    for wn, movies in available_movies:
        await print_available(movies, wn)


async def popular_movies(update: Update, context: CallbackContext) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    # Get currently popular movies
    target_count = 10
    page = 1
    results = movie.popular(page=page)
    total_pages = results["total_pages"]
    pop_movies = []
    while page < total_pages and len(pop_movies) < target_count:
        if page != 1:
            results = movie.popular(page=page)
        movies = results["results"]
        for m in movies:
            if m["id"] in user_data[user]["watched"]:
                continue
            available, provider = is_available_for_free(
                user_data[user]["providers"], m["id"], user_data[user]["region"])
            if available:
                _, poster, desc = extract_movie_info(m)
                pop_movies.append((poster, desc, provider))
        page += 1

    for poster_path, caption, provider in pop_movies[:target_count]:
        provider_str = create_available_at_str(provider)
        if poster_path:
            await update.message.reply_photo(poster_path, esc(caption + "\n" + provider_str), parse_mode=ParseMode.MARKDOWN_V2)
        else:
            await update.message.reply_text(esc(caption + "\n" + provider_str), parse_mode=ParseMode.MARKDOWN_V2)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return
    await update.message.reply_text('Welcome to TMDBot! Use /search to search for a movie.')


async def search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = get_user_id(update)
    if check_user_invalid(user):
        await unauthorized_msg(update)
        return

    query = ' '.join(context.args)
    results = search.movies(query)
    if results and results["total_results"] > 0:
        await update.message.reply_text(f'Search results for "{query}":')
        movies = results["results"]
        res = []
        for m in movies:
            res.append(extract_movie_info(m))
        sorted_res = sort_by_popularity(res)
        show_results = min(5, len(sorted_res))
        # show_results = min(10, results["total_results"])
        for _, poster_path, caption in sorted_res[:show_results]:
            if poster_path:
                await update.message.reply_photo(poster_path, esc(caption), parse_mode=ParseMode.MARKDOWN_V2)
            else:
                await update.message.reply_text(esc(caption), parse_mode=ParseMode.MARKDOWN_V2)

    else:
        await update.message.reply_text('No results found.')


def main():
    application = Application.builder().token(
        settings["telegram_token"]).build()

    # TODO add command menu (and ideally exclude the shortcuts)
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('search', search_handler))
    application.add_handler(CommandHandler('s', search_handler))
    application.add_handler(CommandHandler('show', show_watchlist))
    application.add_handler(CommandHandler('sh', show_watchlist))
    application.add_handler(CommandHandler('add', add_to_watchlist))
    application.add_handler(CommandHandler('a', add_to_watchlist))
    application.add_handler(CommandHandler('tadd', add_to_trash_watchlist))
    application.add_handler(CommandHandler('t', add_to_trash_watchlist))
    application.add_handler(CommandHandler('watched', add_to_watched))
    application.add_handler(CommandHandler('w', add_to_watched))
    application.add_handler(CommandHandler('remove', remove_from_watchlist))
    application.add_handler(CommandHandler('rm', remove_from_watchlist))
    application.add_handler(CommandHandler('services', show_my_providers))
    application.add_handler(CommandHandler('add_service', add_provider))
    application.add_handler(CommandHandler('rm_service', rm_provider))
    application.add_handler(CommandHandler('check', check_watchlist))
    application.add_handler(CommandHandler('c', check_watchlist))
    application.add_handler(CommandHandler('recommend', recommend))
    application.add_handler(CommandHandler('r', recommend))
    application.add_handler(CommandHandler('popular', popular_movies))
    application.add_handler(CommandHandler('pop', popular_movies))

    application.run_polling()


if __name__ == '__main__':
    main()
