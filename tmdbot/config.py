import yaml
import logging
from tmdbv3api import TMDb, Movie, TV, Search, Genre, Provider, Trending, Person

from tmdbot import state

logger = logging.getLogger(__name__)

# These are set by init()
settings = {}
tmdb = TMDb()
movie = Movie()
tv = TV()
search = Search()
genre = Genre()
provider = Provider()
trending = Trending()
person_api = Person()

movie_genre_dict = {}
tv_genre_dict = {}

REGIONS = [
    ("AR", "Argentina"), ("AT", "Austria"), ("AU", "Australia"),
    ("BE", "Belgium"), ("BR", "Brazil"), ("CA", "Canada"),
    ("CH", "Switzerland"), ("CL", "Chile"), ("CO", "Colombia"),
    ("CZ", "Czech Republic"), ("DE", "Germany"), ("DK", "Denmark"),
    ("EC", "Ecuador"), ("EE", "Estonia"), ("ES", "Spain"),
    ("FI", "Finland"), ("FR", "France"), ("GB", "United Kingdom"),
    ("GR", "Greece"), ("HU", "Hungary"), ("ID", "Indonesia"),
    ("IE", "Ireland"), ("IN", "India"), ("IT", "Italy"),
    ("JP", "Japan"), ("KR", "South Korea"), ("LT", "Lithuania"),
    ("LV", "Latvia"), ("MX", "Mexico"), ("MY", "Malaysia"),
    ("NL", "Netherlands"), ("NO", "Norway"), ("NZ", "New Zealand"),
    ("PE", "Peru"), ("PH", "Philippines"), ("PL", "Poland"),
    ("PT", "Portugal"), ("RO", "Romania"), ("RU", "Russia"),
    ("SE", "Sweden"), ("SG", "Singapore"), ("TH", "Thailand"),
    ("TR", "Turkey"), ("TW", "Taiwan"), ("US", "United States"),
    ("VE", "Venezuela"), ("ZA", "South Africa"),
]
REGIONS_PER_PAGE = 8


def load_settings(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def get_api(mode):
    return movie if mode == "movie" else tv


def get_genre_dict(mode):
    return movie_genre_dict if mode == "movie" else tv_genre_dict


def get_movie_genres():
    genre_dict = dict()
    movie_genres = genre.movie_list()["genres"]
    for mg in movie_genres:
        genre_dict[mg["id"]] = mg["name"]
    return genre_dict


def get_tv_genres():
    gd = dict()
    tv_genres = genre.tv_list()["genres"]
    for tg in tv_genres:
        gd[tg["id"]] = tg["name"]
    return gd


def _flag_emoji(code):
    return "".join(chr(0x1F1E6 + ord(c) - ord('A')) for c in code)


def _region_name(code):
    return next((name for c, name in REGIONS if c == code), code)


def user_data_initialize():
    ud = state.user_data
    for user in settings['allowed_users']:
        if user not in ud:
            ud[user] = dict()
            ud[user]["region"] = "DE"
            ud[user]["watched"] = {"movie": {}, "tv": {}}
            ud[user]["watchlists"] = {
                "movie": {"normal": [], "trash": []},
                "tv": {"normal": [], "trash": []},
            }
            ud[user]["providers"] = []
            ud[user]["onboarded"] = False
            ud[user]["mode"] = "movie"
            ud[user]["tv_season_counts"] = {}
            ud[user]["name"] = ""
    # Initialize shared watchlist storage (top-level)
    if "shared_watchlists" not in ud:
        ud["shared_watchlists"] = {}
    if "_shared_wl_next_id" not in ud:
        ud["_shared_wl_next_id"] = 1
    state.save_user_data()


def init(settings_file, user_data_file):
    global movie_genre_dict, tv_genre_dict

    settings.update(load_settings(settings_file))

    tmdb.api_key = settings["tmdb_api_key"]

    state.init(user_data_file)
    from tmdbot.migration import migrate
    migrate()
    user_data_initialize()

    movie_genre_dict = get_movie_genres()
    tv_genre_dict = get_tv_genres()
