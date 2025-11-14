# Dự đoán Khả năng Mua Hàng trên Website

Dự án học máy dự đoán khả năng mua hàng của người dùng trên website dựa trên dữ liệu phiên truy cập.

## 📋 Mô tả

Dự án này sử dụng machine learning để dự đoán biến `Revenue` (TRUE/FALSE) nhằm nhận biết phiên truy cập có kết thúc bằng mua hàng hay không. Đây là bài toán phân loại nhị phân với mức mất cân bằng lớp vừa phải (15.5% phiên mua hàng).

## 🎯 Mục tiêu

- Dự đoán chính xác khả năng mua hàng của người dùng
- Tối ưu chiến dịch marketing
- Cá nhân hóa ưu đãi cho khách hàng tiềm năng
- Đánh giá chất lượng nguồn traffic

## 📁 Cấu trúc dự án

```
bai-cuoi-ky/
├── data/                    # Dữ liệu
│   ├── raw/                 # Dữ liệu gốc
│   ├── processed/           # Dữ liệu đã xử lý
│   └── features/            # Đặc trưng đã tạo
├── notebooks/               # Jupyter notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Training.ipynb
│   ├── 05_Model_Evaluation.ipynb
│   └── 06_Model_Explanation.ipynb
├── src/                     # Source code
│   ├── data/                # Xử lý dữ liệu
│   ├── features/            # Feature engineering
│   ├── models/              # Mô hình ML
│   ├── evaluation/          # Đánh giá mô hình
│   └── utils/               # Utilities
├── app/                     # Demo ứng dụng
├── models/                  # Models đã huấn luyện
├── results/                 # Kết quả
│   ├── figures/             # Biểu đồ
│   ├── reports/             # Báo cáo
│   └── metrics/             # Metrics
└── config/                  # Cấu hình
```

## 🚀 Cài đặt

1. Clone repository hoặc tải source code

2. Tạo môi trường ảo (khuyến nghị):

```bash
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
```

3. Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

## 📊 Dữ liệu

Dataset `online_shoppers_intention.csv` chứa:

- **12,330 phiên truy cập**
- **18 features** bao gồm:
  - Hoạt động phiên (Administrative, Informational, ProductRelated)
  - Chỉ số hiệu suất (BounceRates, ExitRates, PageValues)
  - Bối cảnh phiên (Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend)
- **Target variable**: `Revenue` (TRUE/FALSE)

## 🔄 Workflow

1. **EDA** (`01_EDA.ipynb`): Khám phá dữ liệu, phân tích phân phối
2. **Tiền xử lý** (`02_Data_Preprocessing.ipynb`): Làm sạch dữ liệu, xử lý outliers
3. **Feature Engineering** (`03_Feature_Engineering.ipynb`): Tạo đặc trưng mới
4. **Huấn luyện** (`04_Model_Training.ipynb`): Training Decision Tree và Random Forest
5. **Đánh giá** (`05_Model_Evaluation.ipynb`): So sánh hiệu suất models
6. **Giải thích** (`06_Model_Explanation.ipynb`): SHAP/LIME để giải thích predictions

## 🎓 Mô hình sử dụng

- **Decision Tree**: Mô hình đơn giản, dễ giải thích
- **Random Forest**: Ensemble method cho hiệu suất cao hơn

## 📈 Metrics đánh giá

- Precision, Recall, F1-score
- ROC-AUC, PR-AUC
- Confusion Matrix

## 🛠️ Công nghệ sử dụng

- Python 3.8+
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- SHAP (model explanation)

## 📝 Đề cương

Xem chi tiết trong file `de_cuong_du_an.md`

## 👤 Tác giả

Dự án cuối kỳ - Machine Learning

## 📄 License

MIT License
