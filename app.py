import time
from pathlib import Path

import pandas as pd
import streamlit as st

from model_utils import XSSPredictor


st.set_page_config(
    page_title="XMD-XSS Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

DATASETS = {
    "XSS_dataset": BASE_DIR / "artifacts" / "fmereani",
    "DeepXSS_Dataset_02": BASE_DIR / "artifacts" / "deepxss",
}

QR_PATH = BASE_DIR / "assets" / "qr.png"

EXAMPLES = {
    "XSS cơ bản": '<script>alert(1)</script>',
    "Event handler": '<img src=x onerror=alert(document.cookie)>',
    "javascript: URL": '<a href="javascript:alert(1)">click</a>',
    "Encoded": '%3Cscript%3Ealert%281%29%3C%2Fscript%3E',
    "Benign": 'Đây là nội dung an toàn',
}


@st.cache_resource(show_spinner=False)
def load_predictor(artifact_path: str):
    return XSSPredictor(artifact_path)


def get_metric_value(df: pd.DataFrame, names):
    if df.empty:
        return None

    lower_map = {c.lower(): c for c in df.columns}

    for name in names:
        if name.lower() in lower_map:
            try:
                return float(df[lower_map[name.lower()]].iloc[0])
            except Exception:
                return df[lower_map[name.lower()]].iloc[0]

    if set(["Metric", "Value"]).issubset(df.columns):
        for name in names:
            row = df[df["Metric"].astype(str).str.lower() == name.lower()]
            if not row.empty:
                try:
                    return float(row["Value"].iloc[0])
                except Exception:
                    return row["Value"].iloc[0]

    return None


def pct(x):
    if x is None:
        return "N/A"
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return str(x)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #667085;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .risk-card {
        padding: 1.2rem;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        background: linear-gradient(135deg, #ffffff, #f8fafc);
    }
    .small-note {
        color: #667085;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">XMD-XSS: Demo phát hiện payload XSS bằng học sâu đa phương thức</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Nhập payload → mô hình Text, Structural, Char dự đoán XSS / Benign, hiển thị xác suất, đặc trưng, kết quả thực nghiệm và giải thích XAI.</div>',
    unsafe_allow_html=True,
)

if "payload_input" not in st.session_state:
    st.session_state.payload_input = '<script>alert(1)</script>'

with st.sidebar:
    st.header("Cấu hình demo")

    dataset_name = st.selectbox("Chọn dataset/model", list(DATASETS.keys()))
    artifact_path = DATASETS[dataset_name]

    st.divider()
    st.subheader("Payload mẫu")

    selected_example = st.selectbox("Chọn ví dụ nhanh", list(EXAMPLES.keys()))

    if st.button("Đưa ví dụ vào ô nhập", type="secondary", use_container_width=True):
        st.session_state.payload_input = EXAMPLES[selected_example]
        st.rerun()

    st.divider()
    st.subheader("Truy cập nhanh")

    if QR_PATH.exists():
        st.image(
            str(QR_PATH),
            caption="Quét mã QR để mở website demo",
            use_container_width=True,
        )
        st.caption("Giảng viên có thể quét QR bằng điện thoại để truy cập trực tiếp hệ thống.")
    else:
        st.info("Chưa tìm thấy ảnh QR tại assets/qr.png.")

predictor = load_predictor(str(artifact_path))

payload = st.text_area(
    "Payload cần kiểm tra",
    key="payload_input",
    height=145,
    placeholder="Ví dụ: <img src=x onerror=alert(1)>",
)

col_a, col_b = st.columns([1, 3])

with col_a:
    analyze = st.button("Kiểm tra payload", type="primary", use_container_width=True)

with col_b:
    st.caption(
        "Lưu ý: Đây là website demo học thuật cho KLTN, không thay thế hệ thống WAF/IDS thực tế."
    )

if analyze:
    if not payload.strip():
        st.warning("Vui lòng nhập payload trước khi kiểm tra.")
    else:
        start = time.perf_counter()
        result = predictor.predict(payload)
        elapsed_ms = (time.perf_counter() - start) * 1000

        prob_xss = result["prob_xss"]
        risk_level = result["risk_level"]

        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Kết luận", result["label"])
        c2.metric("Mức cảnh báo", risk_level)
        c3.metric("Xác suất XSS", f"{prob_xss * 100:.2f}%")
        c4.metric("Thời gian suy luận", f"{elapsed_ms:.1f} ms")

        st.progress(
            min(max(prob_xss, 0.0), 1.0),
            text=f"P(XSS) = {prob_xss:.4f}",
        )

        left, right = st.columns([1, 1])

        with left:
            st.subheader("Giải thích nhanh")
            st.write(
                "Các tín hiệu dưới đây kết hợp giữa dấu hiệu bảo mật trong payload "
                "và thống kê XAI toàn cục đã sinh từ notebook."
            )
            st.dataframe(
                result["top_local_signals"],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Token hóa payload")
            token_df = pd.DataFrame(
                {
                    "STT": range(1, len(result["tokens"]) + 1),
                    "Token": result["tokens"],
                }
            )
            st.dataframe(token_df, use_container_width=True, hide_index=True)

        with right:
            st.subheader("Đặc trưng cấu trúc nổi bật")
            feature_df = pd.DataFrame(
                [
                    {"Feature": k, "Value": v}
                    for k, v in result["structural_features"].items()
                ]
            )
            feature_df = feature_df[
                feature_df["Value"].astype(float).abs() > 0
            ].head(30)
            st.dataframe(feature_df, use_container_width=True, hide_index=True)

            st.subheader("Xác suất hai lớp")
            chart_df = pd.DataFrame(
                {
                    "Lớp": ["Benign", "XSS"],
                    "Xác suất": [result["prob_benign"], result["prob_xss"]],
                }
            )
            st.bar_chart(chart_df, x="Lớp", y="Xác suất", use_container_width=True)

st.divider()
st.header("Kết quả nghiên cứu được tích hợp trong demo")

m1, m2, m3, m4 = st.columns(4)

metrics = predictor.metrics

m1.metric("Accuracy", pct(get_metric_value(metrics, ["accuracy", "test_accuracy"])))
m2.metric("Precision", pct(get_metric_value(metrics, ["precision", "test_precision"])))
m3.metric("Recall", pct(get_metric_value(metrics, ["recall", "test_recall"])))
m4.metric("F1-score", pct(get_metric_value(metrics, ["f1", "f1_score", "test_f1"])))

tab1, tab2, tab3, tab4 = st.tabs(
    ["So sánh mô hình", "Confusion matrix", "XAI toàn cục", "Thông tin dataset"]
)

with tab1:
    if not predictor.all_results.empty:
        st.dataframe(predictor.all_results, use_container_width=True, hide_index=True)
    else:
        st.info("Không tìm thấy file all_single_and_multimodal_results.csv.")

with tab2:
    if not predictor.confusion.empty:
        st.dataframe(predictor.confusion, use_container_width=True, hide_index=True)
    else:
        st.info("Không tìm thấy confusion matrix.")

with tab3:
    xai_cols = st.columns(2)

    with xai_cols[0]:
        st.write("Text IG token summary")
        if not predictor.global_ig.empty:
            st.dataframe(
                predictor.global_ig.head(30),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Không có global_text_ig_token_summary_v2_full_test.csv.")

    with xai_cols[1]:
        st.write("Structural SHAP summary")
        if not predictor.global_struct.empty:
            st.dataframe(
                predictor.global_struct.head(30),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Không có global_structural_shap_summary_full_test.csv.")

with tab4:
    if not predictor.dataset_summary.empty:
        st.write("Dataset summary")
        st.dataframe(
            predictor.dataset_summary,
            use_container_width=True,
            hide_index=True,
        )

    if not predictor.label_distribution.empty:
        st.write("Label distribution")
        st.dataframe(
            predictor.label_distribution,
            use_container_width=True,
            hide_index=True,
        )

    if predictor.dataset_summary.empty and predictor.label_distribution.empty:
        st.info("Không có file dataset_summary.csv hoặc label_distribution.csv.")

st.caption(
    "Thiết kế tham khảo ý tưởng kiểm tra rủi ro theo mức cảnh báo giống các công cụ kiểm tra an toàn trực tuyến; "
    "demo này tập trung vào payload XSS và mô hình nghiên cứu của KLTN."
)
