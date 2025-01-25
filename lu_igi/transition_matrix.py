import random
import pandas as pd
from .land_use import LandUse

TRANSITION_MATRIX = pd.DataFrame(1.0, index=list(LandUse), columns=list(LandUse))
for lu_a in TRANSITION_MATRIX.index:
    for lu_b in TRANSITION_MATRIX.columns:
        if lu_a == lu_b : continue
        TRANSITION_MATRIX.loc[lu_a, lu_b] = random.random()

def _normalize(series):
    sum_prob = sum(series)
    return series.apply(lambda v : v/sum_prob)

TRANSITION_MATRIX = TRANSITION_MATRIX.apply(_normalize, axis=1)
