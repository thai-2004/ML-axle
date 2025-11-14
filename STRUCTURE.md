# Cấu trúc file dự án

## Cấu trúc thư mục đề xuất

```
bai-cuoi-ky/
├── data/                          # Thư mục dữ liệu
│   ├── raw/                       # Dữ liệu gốc
│   │   └── online_shoppers_intention.csv
│   ├── processed/                 # Dữ liệu đã xử lý
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── features/                  # File đặc trưng đã tạo
│       └── feature_engineering.py
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_EDA.ipynb              # Khám phá dữ liệu (Exploratory Data Analysis)
│   ├── 02_Data_Preprocessing.ipynb  # Tiền xử lý dữ liệu
│   ├── 03_Feature_Engineering.ipynb  # Tạo đặc trưng
│   ├── 04_Model_Training.ipynb   # Huấn luyện mô hình
│   ├── 05_Model_Evaluation.ipynb # Đánh giá mô hình
│   └── 06_Model_Explanation.ipynb  # Giải thích mô hình (SHAP/LIME)
│
├── src/                           # Source code Python
│   ├── __init__.py
│   ├── data/                      # Module xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── load_data.py          # Load và kiểm tra dữ liệu
│   │   └── preprocess.py         # Tiền xử lý
│   ├── features/                  # Module tạo đặc trưng
│   │   ├── __init__.py
│   │   └── engineering.py        # Feature engineering
│   ├── models/                    # Module mô hình
│   │   ├── __init__.py
│   │   ├── decision_tree.py      # Decision Tree model
│   │   ├── random_forest.py      # Random Forest model
│   │   └── train.py              # Training pipeline
│   ├── evaluation/                # Module đánh giá
│   │   ├── __init__.py
│   │   ├── metrics.py            # Tính toán metrics
│   │   └── visualization.py      # Vẽ biểu đồ đánh giá
│   └── utils/                     # Utilities
│       ├── __init__.py
│       └── helpers.py            # Helper functions
│
├── app/                           # Ứng dụng demo (nếu có)
│   ├── streamlit_app.py          # Streamlit app
│   ├── templates/                 # HTML templates (nếu dùng Flask)
│   └── static/                    # CSS, JS (nếu dùng Flask)
│
├── models/                        # Models đã huấn luyện
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── preprocessor.pkl          # Preprocessing pipeline
│
├── results/                       # Kết quả phân tích
│   ├── figures/                   # Hình ảnh, biểu đồ
│   │   ├── eda_*.png
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   └── feature_importance.png
│   ├── reports/                   # Báo cáo
│   │   └── model_report.md
│   └── metrics/                   # Metrics JSON
│       └── evaluation_results.json
│
├── config/                        # File cấu hình
│   └── config.yaml                # Cấu hình hyperparameters, paths
│
├── requirements.txt               # Python dependencies
├── README.md                      # Hướng dẫn dự án
├── de_cuong_du_an.md             # Đề cương dự án
└── STRUCTURE.md                   # File này

```

## Mô tả các file chính

### 1. Data (`data/`)

- **raw/**: Dữ liệu gốc chưa xử lý
- **processed/**: Dữ liệu sau khi chia train/validation/test
- **features/**: Code tạo đặc trưng

### 2. Notebooks (`notebooks/`)

- **01_EDA.ipynb**: Khám phá dữ liệu, phân tích phân phối, correlation
- **02_Data_Preprocessing.ipynb**: Làm sạch, xử lý missing values, outliers
- **03_Feature_Engineering.ipynb**: Tạo đặc trưng mới (ratios, interactions, etc.)
- **04_Model_Training.ipynb**: Huấn luyện Decision Tree và Random Forest
- **05_Model_Evaluation.ipynb**: Đánh giá hiệu suất, so sánh models
- **06_Model_Explanation.ipynb**: SHAP/LIME để giải thích predictions

### 3. Source Code (`src/`)

- **data/**: Functions để load và preprocess data
- **features/**: Feature engineering logic
- **models/**: Model definitions và training pipelines
- **evaluation/**: Metrics calculation và visualization
- **utils/**: Helper functions dùng chung

### 4. App (`app/`)

- **streamlit_app.py**: Demo ứng dụng cho phép nhập features và nhận prediction
- Hoặc Flask app nếu muốn REST API

### 5. Models (`models/`)

- Các file `.pkl` chứa mô hình đã huấn luyện
- Preprocessing pipeline để áp dụng cho data mới

### 6. Results (`results/`)

- **figures/**: Tất cả biểu đồ, hình ảnh
- **reports/**: Báo cáo kết quả dạng markdown
- **metrics/**: Metrics dạng JSON để tracking

### 7. Config (`config/`)

- **config.yaml**: Cấu hình hyperparameters, file paths, random seed

## File quan trọng khác

- **requirements.txt**: Danh sách thư viện cần thiết (pandas, scikit-learn, shap, streamlit, etc.)
- **README.md**: Hướng dẫn cài đặt, chạy dự án
- **de_cuong_du_an.md**: Đề cương chi tiết dự án

## Workflow đề xuất

1. Bắt đầu với `01_EDA.ipynb` để hiểu dữ liệu
2. Tiền xử lý trong `02_Data_Preprocessing.ipynb`
3. Tạo features trong `03_Feature_Engineering.ipynb`
4. Huấn luyện models trong `04_Model_Training.ipynb`
5. Đánh giá và so sánh trong `05_Model_Evaluation.ipynb`
6. Giải thích predictions trong `06_Model_Explanation.ipynb`
7. Tạo demo app trong `app/streamlit_app.py`
