# -*- coding: utf-8 -*-
"""
Tần suất 00–99, Dirichlet smoothing, χ², xếp hạng
+ BỔ SUNG: so sánh đa khung 1–7–30–90 ngày và tạo 'đánh_giá'
"""

from collections import Counter
from typing import Dict, List, Tuple
import pandas as pd
from scipy.stats import chi2


# ================== CƠ BẢN ==================
def counts_from_days(day2pairs: Dict[str, List[str]]) -> Tuple[pd.DataFrame, int, int]:
    """
    day2pairs: {day_key: ['01','19','19',...], ...}
    -> df(cap,count), N (tổng lượt), days (số ngày)
    """
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
    """
    Thêm cột freq và post_p (posterior sau làm mượt Dirichlet).
    """
    df = df.copy()
    df["freq"] = df["count"] / max(N, 1)
    df["post_p"] = (df["count"] + alpha0 / 100.0) / (N + alpha0)
    return df


def chi_square_test(df: pd.DataFrame, N: int) -> Tuple[float, float]:
    """
    Kiểm định χ² so với phân phối đều (mỗi cap ~ 1%).
    """
    E = N / 100.0 if N > 0 else 0.0
    if E == 0:
        return 0.0, 1.0
    chi2_stat = ((df["count"] - E) ** 2 / E).sum()
    p_value = chi2.sf(chi2_stat, df=99)
    return float(chi2_stat), float(p_value)


def rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xếp hạng theo posterior giảm dần.
    """
    return df.sort_values("post_p", ascending=False).reset_index(drop=True)


# ================== ĐA KHUNG THỜI GIAN ==================
def _days_to_list(day2pairs: Dict[str, List[str]]) -> List[List[str]]:
    """
    Chuyển dict ngày -> list[ list[str] ] theo thứ tự đã crawl (mới -> cũ).
    (Python 3.7+ giữ insertion order của dict)
    """
    return list(day2pairs.values())


def window_counts(
    day2pairs: Dict[str, List[str]],
    windows: List[int] = [1, 7, 30, 90]
) -> pd.DataFrame:
    """
    Đếm số lần xuất hiện theo từng khung:
    -> DataFrame: cap, hits_1d, hits_7d, hits_30d, hits_90d, N_1d, N_7d, N_30d, N_90d
    """
    days_list = _days_to_list(day2pairs)  # list of days, each is list[str]
    index = [f"{i:02d}" for i in range(100)]
    out = pd.DataFrame({"cap": index})

    def _cnt_in_first_k_days(k: int) -> Tuple[List[int], int]:
        sub = days_list[:k] if k <= len(days_list) else days_list
        flat = [p for day in sub for p in day]
        c = Counter(flat)
        N = len(flat)
        return [c.get(x, 0) for x in index], N

    for k in windows:
        counts, N = _cnt_in_first_k_days(k)
        out[f"hits_{k}d"] = counts
        out[f"N_{k}d"] = N

    return out


def multiwindow_posterior(
    day2pairs: Dict[str, List[str]],
    alpha0: float = 200.0,
    windows: List[int] = [7, 30, 90]
) -> pd.DataFrame:
    """
    Posterior theo từng khung (7/30/90):
    -> DataFrame: cap, post_7d, post_30d, post_90d
    """
    wc = window_counts(day2pairs, windows=sorted(set(windows + [1])))
    df = wc[["cap"]].copy()
    for k in windows:
        counts = wc[f"hits_{k}d"]
        N = wc[f"N_{k}d"].iloc[0]  # cùng N cho mọi cap
        df[f"post_{k}d"] = (counts + alpha0 / 100.0) / (max(N, 0) + alpha0)
    return df


def evaluate_numbers(
    day2pairs: Dict[str, List[str]],
    df_ranked: pd.DataFrame,
    alpha0_long: float = 200.0
) -> pd.DataFrame:
    """
    Gắn thêm các cột:
      hits_1d, hits_7d, hits_30d, hits_90d, post_7d, post_30d, post_90d, danh_gia

    Quy tắc 'danh_gia' (heuristic):
      - 'Rơi mạnh (ngắn + dài)':      hits_1d>0 AND hits_7d>=2 AND post_90d>=0.013
      - 'Nóng ngắn hạn, nền yếu':     hits_7d>=2 AND post_90d<0.012
      - 'Nền tốt, tuần nguội':        hits_7d==0 AND post_30d>=0.013 AND post_90d>=0.013
      - 'Chờ kích hoạt (dài hạn tốt)':hits_30d==0 AND post_90d>=0.013
      - else:                          'Trung tính'
    """
    wc = window_counts(day2pairs, windows=[1, 7, 30, 90])
    posts = multiwindow_posterior(day2pairs, alpha0=alpha0_long, windows=[7, 30, 90])

    merged = df_ranked.merge(wc, on="cap", how="left").merge(posts, on="cap", how="left")

    def _judge(row):
        h1, h7, h30, h90 = row["hits_1d"], row["hits_7d"], row["hits_30d"], row["hits_90d"]
        p7, p30, p90 = row["post_7d"], row["post_30d"], row["post_90d"]

        if h1 > 0 and h7 >= 2 and p90 >= 0.013:
            return "Rơi mạnh (ngắn + dài)"
        if h7 >= 2 and p90 < 0.012:
            return "Nóng ngắn hạn, nền yếu"
        if h7 == 0 and p30 >= 0.013 and p90 >= 0.013:
            return "Nền tốt, tuần nguội"
        if h30 == 0 and p90 >= 0.013:
            return "Chờ kích hoạt (dài hạn tốt)"
        return "Trung tính"

    merged["danh_gia"] = merged.apply(_judge, axis=1)
    return merged
