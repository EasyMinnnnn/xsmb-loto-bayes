# -*- coding: utf-8 -*-
import os
import io
import pandas as pd
import streamlit as st

from crawler import crawl
from bayes import counts_from_days, dirichlet_smoothing, chi_square_test, rank

st.set_page_config(page_title="XSMB Loto Bayes 60–90 ngày", layout="wide")

st.title("XSMB Loto – Bayes/Dirichlet")
tab1, tab2 = st.tabs(["📥 Crawl & Phân tích", "📤 Tải lên & Phân tích"])

with tab1:
    colL, colR = st.columns([1,1])
    with colL:
        window = st.selectbox("Cửa sổ dữ liệu", ["60", "90"], index=0,
                              help="Thu thập 60 hoặc 90 ngày gần nhất")
        alpha0 = st.number_input("alpha0 (Dirichlet total prior strength)",
                                 min_value=1.0, max_value=1000.0, value=200.0, step=10.0)
        threshold = st.number_input("Ngưỡng posterior (ví dụ 0.013 = 1.3%)",
                                    min_value=0.0, max_value=0.1, value=0.013, step=0.001, format="%.3f")
        topk = st.slider("Số cặp gợi ý tối đa", 5, 30, 10, 1)
        run_btn = st.button("🚀 Crawl & Phân tích", use_container_width=True)

    if run_btn:
        with st.spinner("Đang thu thập dữ liệu …"):
            day2pairs = crawl(window)
        df, N, days = counts_from_days(day2pairs)
        df = dirichlet_smoothing(df, N, alpha0)
        chi2_stat, pval = chi_square_test(df, N)
        ranked = rank(df)

        st.success(f"Đã thu thập {days} ngày – tổng lượt 2 số: {N}")
        st.write(f"**Chi-square:** {chi2_stat:,.2f} | **p-value:** {pval:.6f}")
        st.caption("Nếu p-value < 0.05 ⇒ có lệch đáng kể so với phân phối đều 1%.")

        st.subheader("Top gợi ý theo posterior")
        suggest = ranked[ranked["post_p"] >= threshold].head(topk)
        st.dataframe(suggest[["cap", "count", "freq", "post_p"]], use_container_width=True)

        # Lưu CSV vào /data và cho phép tải
        os.makedirs("data", exist_ok=True)
        ranked.to_csv("data/xsmb_ranked.csv", index=False, encoding="utf-8-sig")
        df.to_csv("data/xsmb_full.csv", index=False, encoding="utf-8-sig")

        st.download_button("Tải bảng xếp hạng (CSV)", data=open("data/xsmb_ranked.csv","rb").read(),
                           file_name="xsmb_ranked.csv", mime="text/csv")
        st.download_button("Tải bảng đầy đủ (CSV)", data=open("data/xsmb_full.csv","rb").read(),
                           file_name="xsmb_full.csv", mime="text/csv")

with tab2:
    st.write("Bạn có thể **tải lên** 1 CSV tự thống kê (cột `cap`, `count`) để chạy mô hình.")
    upl = st.file_uploader("Tải CSV của bạn", type=["csv"])
    alpha0_u = st.number_input("alpha0 cho file tải lên", min_value=1.0,
                               max_value=1000.0, value=200.0, step=10.0, key="alpha_u")
    if upl:
        dfu = pd.read_csv(upl)
        if not {"cap","count"}.issubset(set(dfu.columns)):
            st.error("CSV cần có cột 'cap' và 'count'.")
        else:
            N = int(dfu["count"].sum())
            dfu = dirichlet_smoothing(dfu, N, alpha0_u)
            chi2_stat, pval = chi_square_test(dfu, N)
            st.write(f"**Chi-square:** {chi2_stat:,.2f} | **p-value:** {pval:.6f}")
            st.dataframe(rank(dfu)[["cap","count","freq","post_p"]], use_container_width=True)
