# BaseCommand now lives in botlib.base
import botlib.base
import sys
sys.modules[__name__] = botlib.base
