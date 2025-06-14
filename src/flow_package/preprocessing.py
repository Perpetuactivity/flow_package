import pandas as pd
import numpy as np
from .const import Const


CONST = Const()


def calcurate(p: pd.Series, slide: bool = False, sliding_window: int = 1000):
    if not slide or (slide and len(p) < sliding_window):
        normalized = (p - p.min()) / (p.max() - p.min())
        normalized = normalized.replace([np.inf, -np.inf], np.nan)
        return normalized
    
    # スライディングウィンドウで正規化
    normalized = pd.Series(0, index=p.index)
    length = len(p)
    for i in range(length):
        buf = p[:sliding_window] if i <= sliding_window else p[i-sliding_window:i]
        normalized[i] = (p[i] - buf.min()) / (buf.max() - buf.min())

    normalized = normalized.replace([np.inf, -np.inf], np.nan)
    return normalized


def normalization(
        df: pd.DataFrame,
        categorical_columns: list[str] = [],
        slide: bool = False,
        sliding_window: int = 1000,
        debug: bool = False,
    ):
    exclude_columns = ["Label"] + categorical_columns

    normalize_columns = [
        col for col in CONST.features_labels if col not in exclude_columns
    ]

    for col in normalize_columns:
        df[col] = calcurate(df[col], slide, sliding_window)
    
    return df