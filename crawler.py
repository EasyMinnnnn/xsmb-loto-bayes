# -*- coding: utf-8 -*-
"""
Crawl XSMB 60–90 ngày từ xosodaiphat.com.

- Nếu ngày có bảng "Loto miền Bắc" (Đầu/Loto) -> dùng trực tiếp.
- Nếu không có, đọc "bảng kết quả" của ngày đó -> trích mọi số, lấy 2 số cuối.
- Chuẩn hóa mỗi ngày tối đa 27 đuôi (một kỳ mở thưởng có 27 giải).

Trả về: dict { 'YYYY-MM-DD' (hoặc 'day_XX'): ['01','19','19', ...], ... }
"""

import re
import time
from typing import Dict, List
import requests
from bs4 import BeautifulSoup, Tag

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

URL_60 = "https://xosodaiphat.com/xsmb-60-ngay.html"
URL_90 = "https://xosodaiphat.com/xsmb-90-ngay.html"

# tìm chuỗi ngày trong các heading/breadcrumb
DATE_PAT = re.compile(r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})")


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=45)
    resp.raise_for_status()
    return resp.text


def _nearest_date_key(node: Tag) -> str:
    """Lấy text ngày gần nhất phía trên node (nếu có)."""
    prev = node.find_previous(["h1", "h2", "h3", "h4", "h5", "p", "div"])
    while prev:
        txt = prev.get_text(" ", strip=True)
        m = DATE_PAT.search(txt)
        if m:
            return m.group(0)
        prev = prev.find_previous(["h1", "h2", "h3", "h4", "h5", "p", "div"])
    return ""


def _is_loto_table(tbl: Tag) -> bool:
    head = tbl.find("tr")
    if not head:
        return False
    ths = [c.get_text(strip=True).lower() for c in head.find_all(["th", "td"])]
    return any("đầu" in t for t in ths) and any("loto" in t for t in ths)


def _extract_pairs_from_loto_table(tbl: Tag) -> List[str]:
    """Lấy các 2 số từ cột 'Loto' (giữ trùng)."""
    pairs: List[str] = []
    for tr in tbl.find_all("tr")[1:]:
        tds = tr.find_all(["td", "th"])
        if len(tds) < 2:
            continue
        right = tds[-1].get_text(" ", strip=True)
        pairs.extend(re.findall(r"\b\d{2}\b", right))
    return pairs


def _looks_like_result_table(tbl: Tag) -> bool:
    """Heuristic: bảng kết quả có nhiều số (>= 20 nhóm số) và có các nhãn G.1, G.2..."""
    txt = tbl.get_text(" ", strip=True).lower()
    if "g.đb" in txt or "g.db" in txt or "giải đặc biệt" in txt or "g.1" in txt:
        # có nhãn giải -> khả năng là bảng kết quả
        return True
    # fallback: đếm các nhóm số 2-5 chữ số
    nums = re.findall(r"\b\d{2,5}\b", txt)
    return len(nums) >= 20


def _extract_pairs_from_result_table(tbl: Tag) -> List[str]:
    """
    Từ một bảng 'kết quả trong ngày', trích mọi số ở các ô <td>,
    lấy 2 số cuối của mỗi số (đối với G7 đã là 2 số).
    """
    tails: List[str] = []
    # chỉ lấy <td> để tránh header 12LG, 1LG...
    for td in tbl.find_all("td"):
        text = td.get_text(" ", strip=True)
        # lấy các cụm số dài 2-5 chữ số
        for m in re.findall(r"\b\d{2,5}\b", text):
            last2 = m[-2:]
            tails.append(last2)
    return tails


def _normalize_day_pairs(pairs: List[str]) -> List[str]:
    """Chuẩn hóa số lượng trong ngày: lấy tối đa 27 đuôi (đúng 27 giải)."""
    # chỉ giữ đúng định dạng 2 chữ số
    pairs = [p.zfill(2)[-2:] for p in pairs if p.isdigit()]
    if len(pairs) > 27:
        return pairs[:27]
    return pairs


# ----------------------------------------------------------------------
# Main parser
# ----------------------------------------------------------------------
def crawl(window: str = "60", max_days: int = 60) -> Dict[str, List[str]]:
    """
    Ưu tiên:
      1) nếu có bảng Loto (Đầu/Loto) -> dùng
      2) nếu không, dùng bảng Kết quả trong ngày
    """
    url = URL_60 if window == "60" else URL_90
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    results: Dict[str, List[str]] = {}
    day_idx = 0

    # chiến lược: duyệt toàn bộ table theo thứ tự xuất hiện trong trang
    for tbl in soup.find_all("table"):
        pairs: List[str] = []

        if _is_loto_table(tbl):
            pairs = _extract_pairs_from_loto_table(tbl)
        elif _looks_like_result_table(tbl):
            pairs = _extract_pairs_from_result_table(tbl)

        if not pairs:
            continue

        pairs = _normalize_day_pairs(pairs)
        # bỏ ngày nếu quá ít số (ví dụ parse nhầm)
        if len(pairs) < 20:
            continue

        key = _nearest_date_key(tbl)
        if not key:
            day_idx += 1
            key = f"day_{day_idx:02d}"
        # nếu key trùng, đánh số tiếp
        base = key
        suffix = 1
        while key in results:
            suffix += 1
            key = f"{base}#{suffix}"

        results[key] = pairs

        if len(results) >= max_days:
            break

    time.sleep(0.3)
    return results


# ----------------------------------------------------------------------
# CLI test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dat = crawl("60", max_days=60)
    days = len(dat)
    total = sum(len(v) for v in dat.values())
    print(f"Ngày thu thập: {days}; tổng lượt 2 số: {total}")
    for i, (k, v) in enumerate(dat.items()):
        print(k, v[:10], "…")
        if i >= 1:
            break
