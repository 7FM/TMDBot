from telegram import (
    ReplyKeyboardMarkup, KeyboardButton,
)

from botlib.keyboards import (  # noqa: F401
    build_media_keyboard, build_watchlist_picker_keyboard,
    build_chunk_keyboard, build_watchlist_select_keyboard,
    build_member_select_keyboard, build_rating_keyboard,
    build_category_picker_keyboard, build_recommend_category_keyboard,
)


def get_main_keyboard(user):
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("/search"), KeyboardButton("/list")],
            [KeyboardButton("/recommend"), KeyboardButton("/trending")],
            [KeyboardButton("/pick"), KeyboardButton("/clear")],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )
