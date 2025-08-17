# -*- coding: utf-8 -*-
"""
Tính tần suất 00–99, Dirichlet smoothing, χ², xếp hạng.
"""
from collections import Counter
from typing import Dict, List, Tuple
import pandas as pd
from scipy.stats import chi2

def counts_from_days(day2pairs: Dict[str, List[str]]) -> Tuple[pd.DataFrame, int, int]:
    pairs = []
    for _, arr in day2pairs.items():
        pairs.extend(arr)
    cnt = Counter(pairs)
    index = [f"{i:02d}" for i in range(100)]
    df = pd.DataFrame({"cap": index, "count": [cnt.get(k, 0) for k in index]})
    N = int(df["count"].sum())
    days = len(day2pairs)
    return df, N, days

def dirichlet_smoothing(df: pd.DataFrame, N: int, alpha0: float = 200.0) -> pd.DataFrame:
    df = df.copy()
    df["freq"] = df["count"] / max(N, 1)
    df["post_p"] = (df["count"] + alpha0/100.0) / (N + alpha0)
    return df

def chi_square_test(df: pd.DataFrame, N: int) -> Tuple[float, float]:
    E = N / 100.0 if N > 0 else 0.0
    if E == 0:
        return 0.0, 1.0
    chi2_stat = ((df["count"] - E) ** 2 / E).sum()
    p_value = chi2.sf(chi2_stat, df=99)
    return float(chi2_stat), float(p_value)

def rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("post_p", ascending=False).reset_index(drop=True)
