import os
import yaml
import logging

logger = logging.getLogger(__name__)

# User data
user_data = dict()

# File path — set by init()
_user_data_file = 'user_data.yaml'


def init(user_data_file):
    global _user_data_file, user_data
    _user_data_file = user_data_file
    if os.path.exists(_user_data_file):
        with open(_user_data_file, 'r') as file:
            user_data = yaml.safe_load(file) or {}


def save_user_data():
    with open(_user_data_file, 'w') as file:
        yaml.safe_dump(user_data, file)


# Provider cache: region -> (timestamp, list[str])
_provider_cache = {}
_PROVIDER_CACHE_TTL = 86400  # 24 hours

# Pending state dicts
_pending_new_watchlist = {}
_pending_search = {}
_pending_person = {}  # user_id -> True (ForceReply state for person search)
_pending_name = {}  # user_id -> True (ForceReply state for display name)
_pending_shared_wl_name = {}  # user_id -> {"media_id": int|None, "mode": str}
# user_id -> {"name": str, "media_id": int|None, "mode": str, "members": [user_ids]}
_pending_shared_wl_members = {}
_pending_season = {}  # user_id -> {"mid": int, "total": int, "media_type": str}
# user_id -> {"mid": int, "media_type": str, "mode": str}
_pending_watched_category = {}

# Chunk expand/collapse state
_chunk_movies = {}
_chunk_id_counter = 0
_CHUNK_MOVIES_MAX = 200


def next_chunk_id():
    global _chunk_id_counter
    cid = _chunk_id_counter
    _chunk_id_counter += 1
    return cid


# Search result tracking: user_id -> (chat_id, [message_ids])
_search_results = {}
# Search pagination: user_id -> (remaining_sorted_results, query)
_search_more = {}
# Rate list tracking: user_id -> (chat_id, [message_ids])
_rate_list_messages = {}
# Recommendation genre filter: user_id -> {"watchlist": str, "genres": set}
_rec_genre_filter = {}
# Undo state: user_id -> {"mid": int, "watchlist": str|None, ...}
_last_watched = {}
