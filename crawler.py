# -*- coding: utf-8 -*-
"""
Thu thập 'Loto miền Bắc' 60–90 ngày gần nhất từ xosodaiphat.com
Kết quả: list các cặp 2 chữ số (có trùng) + thống kê theo ngày.
"""
import re
import time
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
}

URL_60 = "https://xosodaiphat.com/xsmb-60-ngay.html"
URL_90 = "https://xosodaiphat.com/xsmb-90-ngay.html"  # có thể không phải lúc nào cũng có

def _is_loto_table(tbl) -> bool:
    ths = [th.get_text(strip=True) for th in tbl.find_all("th")]
    return any("Đầu" in t for t in ths) and any("Loto" in t for t in ths)

def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=45)
    resp.raise_for_status()
    return resp.text

def parse_loto_tables(html: str) -> Dict[str, List[str]]:
    """
    Trả dict: { 'YYYY-MM-DD': ['01','19','19',...], ... }
    Nếu trang không gắn ngày rõ ràng cho mỗi bảng, dùng chỉ mục 'day_1', 'day_2', ...
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = [t for t in soup.find_all("table") if _is_loto_table(t)]
    result: Dict[str, List[str]] = {}

    # Nhiều trang gắn ngày ở thẻ <h3> / <h2> ngay trước table
    headings = []
    for tag in soup.find_all(["h2", "h3", "h4"]):
        headings.append(tag)

    # Map gần-đúng: heading ngay trước table
    def _nearest_heading_text(tbl):
        prev = tbl.find_previous(["h2", "h3", "h4"])
        if prev:
            return prev.get_text(" ", strip=True)
        return ""

    day_counter = 0
    for tbl in tables:
        rows = tbl.find_all("tr")
        if not rows:
            continue
        day_pairs: List[str] = []
        for tr in rows[1:]:
            tds = tr.find_all(["td", "th"])
            if len(tds) < 2:
                continue
            right_text = tds[-1].get_text(" ", strip=True)
            two_digits = re.findall(r"\b\d{2}\b", right_text)
            day_pairs.extend(two_digits)

        if day_pairs:
            day_counter += 1
            key = _nearest_heading_text(tbl)
            # Chuẩn hóa key ngày, fallback nếu không trích được ngày
            if not key or not re.search(r"\d{4}", key):
                key = f"day_{day_counter:02d}"
            result[key] = day_pairs
    return result

def crawl(window: str = "60") -> Dict[str, List[str]]:
    url = URL_60 if window == "60" else URL_90
    html = fetch_html(url)
    data = parse_loto_tables(html)
    # Thỉnh thoảng trang dài -> tải chậm; nghỉ 1 tí nếu cần mở rộng thêm nguồn khác
    time.sleep(0.5)
    return data

if __name__ == "__main__":
    dat = crawl("60")
    total_pairs = sum(len(v) for v in dat.values())
    print(f"Ngày thu thập: {len(dat)}; tổng lượt 2 số: {total_pairs}")
    # In thử 1 ngày
    if dat:
        k, v = list(dat.items())[0]
        print(k, v[:10], "…")
