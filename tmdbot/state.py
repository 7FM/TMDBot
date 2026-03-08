# All state now lives in botlib.state
# This module redirects to it so existing imports keep working
import botlib.state
import sys
sys.modules[__name__] = botlib.state
