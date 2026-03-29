from typing import Dict, Literal

import pandas as pd


# normaliseMetric and addCompositeScore adapted from:
# github.com/WillEdgington/football-torch-project
# (experiments/experiment_results.py as of 24/03/26)
def normaliseMetric(
    df: pd.DataFrame,
    col: str,
    method: Literal["standard", "minmax"] = "standard",
    eps: float = 1e-8,
) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    series = df[col].astype(float)
    if method == "standard":
        std = series.std()
        return (series - series.mean()) / max(eps, std)
    elif method == "minmax":
        low, high = series.min(), series.max()
        return (series - low) / (high - low)


def addCompositeScore(
    df: pd.DataFrame,
    weights: Dict[str, float],
    ascending: Dict[str, bool] | bool,
    colName: str = "composite_score",
    normMethod: (
        Literal["standard", "minmax"] | Dict[str, Literal["standard", "minmax"]]
    ) = "standard",
) -> pd.DataFrame:
    if isinstance(ascending, bool):
        ascending = {col: ascending for col in weights.keys()}
    if set(weights.keys()) != set(ascending.keys()):
        raise ValueError("weights and ascending must have the same keys")
    if isinstance(normMethod, dict):
        if set(weights.keys()) != set(normMethod.keys()):
            raise ValueError(
                "if normMethod is dict then it must have "
                "the same keys as weights and ascending"
            )

    df = df.copy()
    totalWeight = sum(weights.values())
    score = pd.Series(0.0, index=df.index)

    for col, weight in weights.items():
        score += (
            normaliseMetric(
                df,
                col,
                method=(
                    normMethod if not isinstance(normMethod, dict) else normMethod[col]
                ),
            )
            * (weight / totalWeight)
            * (-1 if ascending[col] else 1)
        )

    df[colName] = score
    return df
