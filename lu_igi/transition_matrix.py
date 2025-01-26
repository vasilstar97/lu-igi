import random
import pandas as pd
from .land_use import LandUse
# Список значений LandUse

# Определение вероятностей переходов вручную (пример с предположениями)
transition_probabilities = {
    LandUse.RESIDENTIAL: [0.05, 0.25, 0.2, 0.05, 0.1, 0.15, 0.2],
    LandUse.BUSINESS: [0.15, 0.1, 0.1, 0.1, 0.2, 0.1, 0.25],
    LandUse.RECREATION: [0.25, 0.1, 0.1, 0.05, 0.05, 0.3, 0.15],
    LandUse.SPECIAL: [0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2],
    LandUse.INDUSTRIAL: [0.1, 0.2, 0.05, 0.1, 0.25, 0.15, 0.15],
    LandUse.AGRICULTURE: [0.15, 0.1, 0.3, 0.1, 0.1, 0.05, 0.2],
    LandUse.TRANSPORT: [0.2, 0.25, 0.1, 0.1, 0.1, 0.15, 0.1],
}

# Создание DataFrame
TRANSITION_MATRIX = pd.DataFrame(transition_probabilities, index=list(LandUse), columns=list(LandUse))

# def _normalize(series):
#     sum_prob = sum(series)
#     return series.apply(lambda v : v/sum_prob)

# Нормализация строк (вдруг есть ошибки)
TRANSITION_MATRIX = TRANSITION_MATRIX.div(TRANSITION_MATRIX.sum(axis=1), axis=0)

# TRANSITION_MATRIX = TRANSITION_MATRIX.apply(_normalize, axis=1)
