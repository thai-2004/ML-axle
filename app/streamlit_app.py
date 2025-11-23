"""
Demo Streamlit cho dự án Online Shoppers Intention.

Cho phép nhập thủ công các đặc trưng cấp phiên gốc để các bộ phân loại
Decision Tree và Random Forest đã được huấn luyện có thể được sử dụng để dự đoán
liệu một phiên người dùng có kết thúc bằng giao dịch mua hàng (`Revenue`) hay không.
"""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATHS = {
    "Rừng Ngẫu Nhiên": MODEL_DIR / "random_forest.pkl",
    "Cây Quyết Định": MODEL_DIR / "decision_tree.pkl",
}

FEATURE_ORDER: List[str] = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

MONTH_ORDER = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "June",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

MONTH_ENCODING: Dict[str, int] = {
    "Aug": 0,
    "Dec": 1,
    "Feb": 2,
    "Jul": 3,
    "June": 4,
    "Mar": 5,
    "May": 6,
    "Nov": 7,
    "Oct": 8,
    "Sep": 9,
}

VISITOR_ORDER = ["New_Visitor", "Other", "Returning_Visitor"]

VISITOR_ENCODING: Dict[str, int] = {
    "New_Visitor": 0,
    "Other": 1,
    "Returning_Visitor": 2,
}

WEEKEND_ENCODING: Dict[bool, int] = {False: 0, True: 1}

NUMERIC_FEATURES: List[str] = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]


@st.cache_resource
def load_models() -> Dict[str, Any]:
    """Tải các mô hình đã serialize một lần để giữ giao diện phản hồi nhanh."""

    loaded = {}
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy mô hình: {path}")
        loaded[name] = joblib.load(path)
    return loaded


@st.cache_data
def load_reference_data():
    """Đọc các tập dữ liệu thô và đã xử lý cùng với tính toán các giá trị mặc định/tùy chọn."""

    raw_path = DATA_DIR / "raw" / "online_shoppers_intention.csv"
    encoded_path = DATA_DIR / "processed" / "df_encoded.csv"
    raw_df = pd.read_csv(raw_path)
    encoded_df = pd.read_csv(encoded_path)

    defaults = {
        feature: float(raw_df[feature].median())
        for feature in NUMERIC_FEATURES
    }
    defaults.update(
        {
            "Month": raw_df["Month"].mode().iloc[0],
            "VisitorType": raw_df["VisitorType"].mode().iloc[0],
            "Weekend": bool(raw_df["Weekend"].mode().iloc[0]),
            "OperatingSystems": int(raw_df["OperatingSystems"].mode().iloc[0]),
            "Browser": int(raw_df["Browser"].mode().iloc[0]),
            "Region": int(raw_df["Region"].mode().iloc[0]),
            "TrafficType": int(raw_df["TrafficType"].mode().iloc[0]),
        }
    )

    month_options = [
        month for month in MONTH_ORDER if month in raw_df["Month"].unique()
    ]
    visitor_options = [
        visitor
        for visitor in VISITOR_ORDER
        if visitor in raw_df["VisitorType"].unique()
    ]
    options = {
        "Month": month_options,
        "OperatingSystems": sorted(raw_df["OperatingSystems"].unique().astype(int)),
        "Browser": sorted(raw_df["Browser"].unique().astype(int)),
        "Region": sorted(raw_df["Region"].unique().astype(int)),
        "TrafficType": sorted(raw_df["TrafficType"].unique().astype(int)),
        "VisitorType": visitor_options,
    }

    return raw_df, encoded_df, defaults, options


def encode_payload(payload: Dict[str, float]) -> Dict[str, float]:
    """Chuyển đổi đầu vào thành các giá trị số mà các mô hình mong đợi."""

    encoded = payload.copy()
    encoded["Month"] = MONTH_ENCODING[encoded["Month"]]
    encoded["VisitorType"] = VISITOR_ENCODING[encoded["VisitorType"]]
    encoded["Weekend"] = WEEKEND_ENCODING[encoded["Weekend"]]
    return encoded


def predict(model, input_df: pd.DataFrame) -> Dict[str, float]:
    """Chạy mô hình đã chọn và trả về dự đoán + xác suất."""

    prediction = model.predict(input_df)[0]
    probability = float(model.predict_proba(input_df)[0][1])
    return {"prediction": bool(prediction), "probability": probability}


def main():
    st.set_page_config(
        page_title="Demo Dự Đoán Doanh Thu Người Mua Hàng Trực Tuyến", layout="wide"
    )
    st.title("Dự Đoán `Revenue` cho Phiên Truy Cập Website")
    st.markdown(
        """
        Sử dụng biểu mẫu bên dưới để điều chỉnh các chỉ số phiên (lượt xem trang, tỷ lệ thoát,
        lựa chọn thiết bị/trình duyệt, v.v.) và so sánh cách các mô hình Decision Tree và Random
        Forest phản ứng.
        """
    )

    models = load_models()
    raw_df, encoded_df, defaults, options = load_reference_data()

    with st.sidebar:
        st.header("Cài đặt mô hình")
        model_name = st.selectbox("Chọn mô hình", list(MODEL_PATHS.keys()))
        st.caption("Chạy `streamlit run app/streamlit_app.py` từ thư mục gốc của repo.")
        st.divider()
        st.markdown(
            "Bản snapshot tập dữ liệu đến từ `data/processed/df_encoded.csv`."
        )

    with st.form("session_form"):
        st.subheader("Tín hiệu cấp phiên")
        col1, col2, col3 = st.columns(3)
        with col1:
            admin = st.number_input(
                "Administrative",
                value=float(defaults["Administrative"]),
                min_value=0.0,
                step=1.0,
                help="Số trang quản trị đã truy cập",
            )
            admin_duration = st.number_input(
                "Thời gian Administrative",
                value=float(defaults["Administrative_Duration"]),
                min_value=0.0,
                step=0.5,
                help="Thời gian truy cập các trang quản trị (giây)",
            )
            informational = st.number_input(
                "Informational",
                value=float(defaults["Informational"]),
                min_value=0.0,
                step=1.0,
                help="Số trang thông tin đã truy cập",
            )
            informational_duration = st.number_input(
                "Thời gian Informational",
                value=float(defaults["Informational_Duration"]),
                min_value=0.0,
                step=0.5,
                help="Thời gian truy cập các trang thông tin (giây)",
            )
        with col2:
            product = st.number_input(
                "ProductRelated",
                value=float(defaults["ProductRelated"]),
                min_value=0.0,
                step=1.0,
                help="Số trang sản phẩm đã truy cập",
            )
            product_duration = st.number_input(
                "Thời gian ProductRelated",
                value=float(defaults["ProductRelated_Duration"]),
                min_value=0.0,
                step=1.0,
                help="Thời gian truy cập các trang sản phẩm (giây)",
            )
            bounce = st.number_input(
                "Tỷ lệ Bounce",
                value=float(defaults["BounceRates"]),
                min_value=0.0,
                max_value=1.0,
                step=0.001,
                help="Tỷ lệ người dùng rời khỏi sau một trang",
            )
            exit_rate = st.number_input(
                "Tỷ lệ Thoát",
                value=float(defaults["ExitRates"]),
                min_value=0.0,
                max_value=1.0,
                step=0.001,
                help="Tỷ lệ thoát khỏi trang",
            )
        with col3:
            page_values = st.number_input(
                "Giá trị Trang",
                value=float(defaults["PageValues"]),
                min_value=0.0,
                step=0.1,
                help="Giá trị trung bình của trang trong phiên",
            )
            special_day = st.number_input(
                "Ngày Đặc Biệt",
                value=float(defaults["SpecialDay"]),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                help="Độ gần với ngày lễ/đặc biệt (0-1)",
            )
            st.selectbox(
                "Tháng",
                options=options["Month"],
                index=options["Month"].index(defaults["Month"]),
                key="month_select",
            )
            st.selectbox(
                "Loại Khách Truy Cập",
                options=options["VisitorType"],
                index=options["VisitorType"].index(defaults["VisitorType"]),
                key="visitor_select",
            )

        st.markdown("### Siêu dữ liệu Kỹ thuật")
        ops_col, browser_col, region_col, traffic_col = st.columns(4)
        with ops_col:
            st.selectbox(
                "Hệ Điều Hành",
                options=options["OperatingSystems"],
                index=options["OperatingSystems"].index(
                    int(defaults["OperatingSystems"])
                ),
                key="os_select",
            )
        with browser_col:
            st.selectbox(
                "Trình Duyệt",
                options=options["Browser"],
                index=options["Browser"].index(int(defaults["Browser"])),
                key="browser_select",
            )
        with region_col:
            st.selectbox(
                "Khu vực",
                options=options["Region"],
                index=options["Region"].index(int(defaults["Region"])),
                key="region_select",
            )
        with traffic_col:
            st.selectbox(
                "Loại Lưu Lượng",
                options=options["TrafficType"],
                index=options["TrafficType"].index(
                    int(defaults["TrafficType"])
                ),
                key="traffic_select",
            )
        weekend_flag = st.checkbox("Phiên Cuối Tuần", value=defaults["Weekend"])

        submitted = st.form_submit_button("Chạy Dự Đoán")

    session_payload = {
        "Administrative": admin,
        "Administrative_Duration": admin_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product,
        "ProductRelated_Duration": product_duration,
        "BounceRates": bounce,
        "ExitRates": exit_rate,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": st.session_state["month_select"],
        "OperatingSystems": st.session_state["os_select"],
        "Browser": st.session_state["browser_select"],
        "Region": st.session_state["region_select"],
        "TrafficType": st.session_state["traffic_select"],
        "VisitorType": st.session_state["visitor_select"],
        "Weekend": weekend_flag,
    }

    if submitted:
        encoded_input = encode_payload(session_payload)
        input_df = pd.DataFrame([encoded_input], columns=FEATURE_ORDER)
        result = predict(models[model_name], input_df)
        st.success("Dự đoán đã sẵn sàng")
        col_a, col_b = st.columns(2)
        col_a.metric(
            label="Doanh thu dự đoán",
            value="Có" if result["prediction"] else "Không",
            delta=f"{result['probability'] * 100:.1f}% xác suất",
        )
        col_b.progress(result["probability"])
        with st.expander("Dòng đã mã hóa + siêu dữ liệu"):
            st.dataframe(input_df, use_container_width=True)
        with st.expander("Bản snapshot tập dữ liệu (đã mã hóa)"):
            st.dataframe(encoded_df.head(), use_container_width=True)
    else:
        st.info("Điều chỉnh các trường và nhấp **Chạy Dự Đoán** để xem kết quả.")

    st.markdown("---")
    st.caption("Nguồn dữ liệu: `data/raw/online_shoppers_intention.csv`")


if __name__ == "__main__":
    main()

