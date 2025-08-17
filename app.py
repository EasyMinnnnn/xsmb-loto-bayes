# -*- coding: utf-8 -*-
import os
import pandas as pd
import streamlit as st

from crawler import crawl
from bayes import (
    counts_from_days,
    dirichlet_smoothing,
    chi_square_test,
    rank,
    evaluate_numbers,   # <- thêm: đánh giá đa khung 1–7–30–90 ngày
)

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="XSMB Loto – Bayes/Dirichlet",
    page_icon="🎯",
    layout="wide",
)

# ===== CSS THEME =====
st.markdown("""
<style>
/* Nền tổng thể */
.main {
    background: linear-gradient(135deg, #f9fafb 0%, #eef2f7 100%);
}

/* Container */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Header */
h1 {
    color: #1e3a8a;
    font-weight: 800;
    text-align: center;
    padding: 0.5rem;
    border-bottom: 3px solid #2563eb;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #f1f5f9;
    border-right: 1px solid #e2e8f0;
}

/* Card metric */
div[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    text-align: center;
}

/* DataFrame */
thead tr th {
    background: #2563eb !important;
    color: white !important;
    font-weight: 600 !important;
    text-align: center;
}
tbody tr:nth-child(even) {
    background-color: #f9fafb !important;
}
tbody tr:nth-child(odd) {
    background-color: #ffffff !important;
}

/* Buttons */
button[kind="primary"] {
    background: linear-gradient(90deg, #2563eb, #1e40af);
    color: white;
    border-radius: 10px;
    font-weight: 600;
}

/* Expander */
.streamlit-expanderHeader {
    background: #f1f5f9;
    color: #1e3a8a;
    font-weight: 600;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.title("🎯 XSMB Loto – Phân tích Bayes (Dirichlet)")
st.caption("Thống kê 2 số đuôi 00–99 trong 60/90 ngày • Làm mượt Bayes • Kiểm định χ² • Gợi ý & Đánh giá đa khung 1–7–30–90 ngày")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Cấu hình")
    window = st.selectbox("Phân tích dữ liệu (ngày)", ["60", "90"], index=0)
    alpha0 = st.number_input("alpha0 (Dirichlet)", min_value=1.0, max_value=1000.0, value=200.0, step=10.0)
    threshold = st.number_input("Ngưỡng posterior", min_value=0.0, max_value=0.1, value=0.013, step=0.001, format="%.3f")
    topk = st.slider("Số cặp gợi ý tối đa", 5, 30, 10, 1)
    crawl_btn = st.button("🚀 Crawl & Phân tích", use_container_width=True)

# ===== MAIN =====
if crawl_btn:
    with st.spinner("Đang crawl & phân tích…"):
        # 1) Thu thập & đếm
        day2pairs = crawl(window)
        df, N, days = counts_from_days(day2pairs)

        # 2) Bayes smoothing (posterior dài hạn theo cửa sổ đã chọn)
        df = dirichlet_smoothing(df, N, alpha0)

        # 3) Kiểm định χ²
        chi2_stat, pval = chi_square_test(df, N)

        # 4) Xếp hạng theo posterior
        ranked = rank(df)

        # 5) Gắn ĐÁNH GIÁ đa khung 1–7–30–90 ngày (không thêm control mới)
        evaluated = evaluate_numbers(day2pairs, ranked, alpha0_long=alpha0)

    # ===== METRICS =====
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số ngày", f"{days}")
    c2.metric("Tổng lượt", f"{N:,}")
    c3.metric("Chi-square", f"{chi2_stat:,.2f}")
    c4.metric("p-value", f"{pval:.6f}")

    # ===== TOP GỢI Ý (kèm đánh giá) =====
    st.subheader("⭐ Top gợi ý (kèm đánh giá 1–7–30–90 ngày)")
    suggest = evaluated[evaluated["post_p"] >= threshold].head(topk)
    st.dataframe(
        suggest[[
            "cap","count","freq","post_p",
            "hits_1d","hits_7d","hits_30d",
            "post_7d","post_30d","post_90d",
            "danh_gia"
        ]].rename(columns={
            "cap":"Cặp","count":"Lượt","freq":"Tần suất","post_p":"Posterior",
            "hits_1d":"Hôm qua","hits_7d":"7 ngày","hits_30d":"30 ngày",
            "post_7d":"Post_7d","post_30d":"Post_30d","post_90d":"Post_90d",
            "danh_gia":"Đánh giá"
        }),
        use_container_width=True, hide_index=True
    )

    # ===== BẢNG ĐẦY ĐỦ =====
    with st.expander("Xem toàn bộ 00–99 (kèm đánh giá)"):
        st.dataframe(
            evaluated[[
                "cap","count","freq","post_p",
                "hits_1d","hits_7d","hits_30d","hits_90d",
                "post_7d","post_30d","post_90d",
                "danh_gia"
            ]].rename(columns={
                "cap":"Cặp","count":"Lượt","freq":"Tần suất","post_p":"Posterior",
                "hits_1d":"Hôm qua","hits_7d":"7 ngày","hits_30d":"30 ngày","hits_90d":"90 ngày",
                "post_7d":"Post_7d","post_30d":"Post_30d","post_90d":"Post_90d",
                "danh_gia":"Đánh giá"
            }),
            use_container_width=True, hide_index=True
        )

    # ===== LƯU & TẢI CSV =====
    os.makedirs("data", exist_ok=True)
    ranked.to_csv("data/xsmb_ranked.csv", index=False, encoding="utf-8-sig")
    df.to_csv("data/xsmb_full.csv", index=False, encoding="utf-8-sig")
    evaluated.to_csv("data/xsmb_eval.csv", index=False, encoding="utf-8-sig")

    cL, cM, cR = st.columns(3)
    with cL:
        st.download_button("⬇️ Tải bảng xếp hạng (CSV)",
                           data=open("data/xsmb_ranked.csv","rb").read(),
                           file_name="xsmb_ranked.csv", mime="text/csv",
                           use_container_width=True)
    with cM:
        st.download_button("⬇️ Tải bảng đầy đủ (CSV)",
                           data=open("data/xsmb_full.csv","rb").read(),
                           file_name="xsmb_full.csv", mime="text/csv",
                           use_container_width=True)
    with cR:
        st.download_button("⬇️ Tải bảng đánh giá (CSV)",
                           data=open("data/xsmb_eval.csv","rb").read(),
                           file_name="xsmb_eval.csv", mime="text/csv",
                           use_container_width=True)

else:
    st.info("👈 Chọn tham số trong Sidebar và nhấn **🚀 Crawl & Phân tích** để bắt đầu.")
