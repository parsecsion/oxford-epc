"""COM6003 Oxford EPC project — shared package."""
import os
import random

import numpy as np

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Canonical ordinal ordering of EPC bands (best -> worst).
RATING_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
RATING_TO_INT = {r: i for i, r in enumerate(RATING_ORDER)}
INT_TO_RATING = {i: r for i, r in enumerate(RATING_ORDER)}
