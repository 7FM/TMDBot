# Messaging now lives in botlib.messaging
import botlib.messaging
import sys
sys.modules[__name__] = botlib.messaging
