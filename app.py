# -*- coding: utf-8 -*-
import os
import io
import pandas as pd
import streamlit as st

from crawler import crawl
from bayes import counts_from_days, dirichlet_smoothing, chi_square_test, rank

# ============ PAGE CONFIG & THEME TWEAK ============
st.set_page_config(
    page_title="XSMB Loto – Bayes/Dirichlet",
    page_icon="🎯",
    layout="wide",
)

# Minimal css polish
st.markdown("""
<style>
/* Tăng độ sắc nét cho bảng và card */
.block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
div[data-testid="stMetricValue"] {font-weight: 700;}
thead tr th {font-weight: 700 !important;}
/* Nút bấm full width gọn gàng */
button[kind="primary"] {border-radius: 10px; font-weight: 600;}
/* Hộp thông tin */
.info-card {padding: 0.9rem 1rem; border: 1px solid #E5E7EB; border-radius: 10px; background: #FAFAFA;}
.small {font-size: 0.86rem; color: #6B7280;}
</style>
""", unsafe_allow_html=True)

# ============ HEADER ============
st.title("🎯 XSMB Loto – Phân tích Bayes (Dirichlet)")
st.caption("Thống kê 2 số đuôi 00–99 trong 60/90 ngày • Làm mượt Bayes • Kiểm định χ² • Gợi ý số nổi bật")

# ============ SIDEBAR CONTROLS ============
with st.sidebar:
    st.header("⚙️ Cấu hình")
    window = st.selectbox("Phân tích dữ liệu (ngày)", ["60", "90"], index=0,
                          help="Chọn khoảng ngày gần nhất để thu thập & phân tích")
    alpha0 = st.number_input("alpha0 (Dirichlet total prior strength)",
                             min_value=1.0, max_value=1000.0, value=200.0, step=10.0,
                             help="Độ mạnh của prior đều 1%. Dữ liệu ít → alpha0 lớn; dữ liệu dài → alpha0 nhỏ.")
    threshold = st.number_input("Ngưỡng posterior (vd 0.013 = 1.3%)",
                                min_value=0.0, max_value=0.1, value=0.013, step=0.001, format="%.3f")
    topk = st.slider("Số cặp gợi ý tối đa", 5, 30, 10, 1)
    st.markdown("---")
    crawl_btn = st.button("🚀 Crawl & Phân tích", use_container_width=True)

tabs = st.tabs(["📊 Kết quả phân tích", "📁 Quick view dữ liệu `data/`", "📤 Tải lên & Phân tích"])

# ============ MAIN TAB 1: RUN + SHOW ============
with tabs[0]:
    if crawl_btn:
        with st.spinner("Đang thu thập và phân tích…"):
            day2pairs = crawl(window)                 # lấy 60/90 ngày
            df, N, days = counts_from_days(day2pairs) # đếm 00–99
            df = dirichlet_smoothing(df, N, alpha0)   # Bayes smoothing
            chi2_stat, pval = chi_square_test(df, N)   # kiểm định
            ranked = rank(df)

        # Summary cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số ngày thu thập", f"{days}")
        c2.metric("Tổng lượt 2 số", f"{N:,}")
        c3.metric("Chi-square", f"{chi2_stat:,.2f}")
        c4.metric("p-value", f"{pval:.6f}")

        # Hint box
        st.markdown(
            '<div class="info-card small">Mốc tham chiếu: trung bình lý thuyết ≈ 1% cho mỗi cặp 00–99. '
            'Nếu <b>p-value &lt; 0.05</b> ⇒ có lệch đáng kể so với đều; ngược lại thì coi như gần ngẫu nhiên.</div>',
            unsafe_allow_html=True
        )

        # Suggestions
        st.subheader("⭐ Top gợi ý theo posterior")
        suggest = ranked[ranked["post_p"] >= threshold].head(topk)
        st.dataframe(
            suggest[["cap", "count", "freq", "post_p"]]
            .rename(columns={"cap": "cặp", "count": "lượt", "freq": "tần suất", "post_p": "posterior"}),
            use_container_width=True,
            hide_index=True
        )

        # Full table in expander
        with st.expander("Xem bảng đầy đủ 00–99"):
            st.dataframe(
                ranked[["cap","count","freq","post_p"]]
                .rename(columns={"cap": "cặp", "count": "lượt", "freq": "tần suất", "post_p": "posterior"}),
                use_container_width=True, hide_index=True
            )

        # Save & downloads
        os.makedirs("data", exist_ok=True)
        ranked.to_csv("data/xsmb_ranked.csv", index=False, encoding="utf-8-sig")
        df.to_csv("data/xsmb_full.csv", index=False, encoding="utf-8-sig")

        cL, cR = st.columns([1,1])
        with cL:
            st.download_button("⬇️ Tải bảng xếp hạng (CSV)",
                               data=open("data/xsmb_ranked.csv","rb").read(),
                               file_name="xsmb_ranked.csv", mime="text/csv", use_container_width=True)
        with cR:
            st.download_button("⬇️ Tải bảng đầy đủ (CSV)",
                               data=open("data/xsmb_full.csv","rb").read(),
                               file_name="xsmb_full.csv", mime="text/csv", use_container_width=True)
    else:
        st.markdown(
            '<div class="info-card">Chọn tham số trong <b>Sidebar</b> rồi nhấn <b>🚀 Crawl & Phân tích</b> để bắt đầu.</div>',
            unsafe_allow_html=True
        )

# ============ TAB 2: QUICK VIEW FROM /data ============
with tabs[1]:
    st.write("Hiển thị nhanh các file CSV đã được lưu trong thư mục `data/` (do workflow hoặc lần chạy trước tạo).")
    colA, colB = st.columns([1,1])
    with colA:
        path_ranked = "data/xsmb_ranked.csv"
        if os.path.exists(path_ranked):
            st.success("Đã tìm thấy `data/xsmb_ranked.csv`")
            df_ranked = pd.read_csv(path_ranked)
            st.dataframe(df_ranked.head(20), use_container_width=True, hide_index=True)
        else:
            st.warning("Chưa có `data/xsmb_ranked.csv` trong repo.")
    with colB:
        path_full = "data/xsmb_full.csv"
        if os.path.exists(path_full):
            st.success("Đã tìm thấy `data/xsmb_full.csv`")
            df_full = pd.read_csv(path_full)
            st.dataframe(df_full.head(20), use_container_width=True, hide_index=True)
        else:
            st.warning("Chưa có `data/xsmb_full.csv` trong repo.")

# ============ TAB 3: UPLOAD YOUR OWN ============
with tabs[2]:
    st.write("Tải lên CSV tuỳ chỉnh (cần tối thiểu 2 cột: `cap` và `count`).")
    upl = st.file_uploader("Chọn file CSV của bạn", type=["csv"])
    alpha0_u = st.number_input("alpha0 cho file tải lên", min_value=1.0,
                               max_value=1000.0, value=200.0, step=10.0, key="alpha_u")
    if upl:
        dfu = pd.read_csv(upl)
        if not {"cap","count"}.issubset(set(dfu.columns)):
            st.error("CSV phải có cột 'cap' và 'count'.")
        else:
            N_u = int(dfu["count"].sum())
            dfu2 = dirichlet_smoothing(dfu, N_u, alpha0_u)
            chi2_stat_u, pval_u = chi_square_test(dfu2, N_u)

            c1, c2 = st.columns(2)
            c1.metric("Tổng lượt 2 số (upload)", f"{N_u:,}")
            c2.metric("p-value (upload)", f"{pval_u:.6f}")

            st.dataframe(rank(dfu2)[["cap","count","freq","post_p"]],
                         use_container_width=True, hide_index=True)
