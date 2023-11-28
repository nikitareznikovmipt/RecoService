import os

CURRENT_DIR = os.path.dirname(os.getcwd())

INTERACTIONS_DATA = os.path.join(CURRENT_DIR, "data/interactions.csv")
USER_DATA = os.path.join(CURRENT_DIR, "data/users.csv")
ITEM_DATA = os.path.join(CURRENT_DIR, "data/items.csv")

TOP_POPULAR_RECS = [10440, 15297, 9728, 13865, 4151, 3734, 2657, 4880, 142, 6809]
