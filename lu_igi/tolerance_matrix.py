import random
import pandas as pd
from .land_use import LandUse
# Список значений LandUse

# Определение вероятностей переходов вручную (пример с предположениями)
adjacency_tolerances = {
    LandUse.RESIDENTIAL: [1.0, 0.52, 0.18, 0.24, 0.35, 0.29, 0.06],
    LandUse.BUSINESS: [0.52, 1.0, 0.32, 0.26, 0.44, 0.28, 0.69],
    LandUse.INDUSTRIAL: [0.18, 0.32, 1.0, 0.54, 0.18, 0.46, 0.14],
    LandUse.TRANSPORT: [0.24, 0.25, 0.54, 1.0, 0.16, 0.26, 0.2],
    LandUse.RECREATION: [0.35, 0.44, 0.18, 0.16, 1.0, 0.28, 0.06],
    LandUse.AGRICULTURE: [0.29, 0.28, 0.46, 0.26, 0.28, 1.0, 0.11],
    LandUse.SPECIAL: [0.06, 0.07, 0.14, 0.2, 0.06, 0.11, 1.0],
}

# Создание DataFrame
TOLERANCE_MATRIX = pd.DataFrame(adjacency_tolerances, index=adjacency_tolerances.keys(), columns=adjacency_tolerances.keys())
