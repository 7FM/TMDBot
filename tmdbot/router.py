# Router now lives in botlib.router
import botlib.router
import sys
sys.modules[__name__] = botlib.router
