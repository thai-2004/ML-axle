# Hướng dẫn chạy dự án

## 📋 Tổng quan

Dự án này sử dụng Jupyter Notebooks để thực hiện pipeline từ khám phá dữ liệu đến đánh giá mô hình. Các bước được thực hiện tuần tự từ notebook 01 đến 06.

---

## 🚀 Các bước chạy

### **Bước 1: Cài đặt môi trường**

```powershell
# Di chuyển vào thư mục dự án
cd D:\code\machine\bai-cuoi-ky

# Tạo môi trường ảo (nếu chưa có)
python -m venv venv

# Kích hoạt môi trường ảo
.\venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

✅ **Kiểm tra**: Chạy `pip list` để xác nhận đã cài đủ các thư viện.

---

### **Bước 2: Khởi động Jupyter**

Có 2 cách:

#### **Cách 1: Jupyter Notebook (Giao diện cổ điển)**

```powershell
jupyter notebook
```

→ Mở trình duyệt tại `http://localhost:8888`

#### **Cách 2: Jupyter Lab (Giao diện hiện đại - Khuyến nghị)**

```powershell
jupyter lab
```

→ Mở trình duyệt tại `http://localhost:8888`

---

### **Bước 3: Chạy các notebooks theo thứ tự**

#### **📊 Notebook 1: EDA (Exploratory Data Analysis)**

**File**: `notebooks/01_EDA.ipynb`

**Mục đích**: Khám phá dữ liệu, hiểu cấu trúc và phân phối

**Thực hiện**:

1. Mở notebook `01_EDA.ipynb` trong Jupyter
2. Chạy tất cả các cells (Cell → Run All)
3. Xem kết quả:
   - Tổng quan dữ liệu (số dòng, số cột, kiểu dữ liệu)
   - Phân phối biến mục tiêu `Revenue`
   - Missing values
   - Correlation matrix
   - Biểu đồ phân phối các features

**Kết quả mong đợi**:

- Hiểu được cấu trúc dữ liệu
- Nhận biết class imbalance (≈15.5% mua hàng)
- Xác định các features quan trọng

---

#### **🧹 Notebook 2: Data Preprocessing**

**File**: `notebooks/02_Data_Preprocessing.ipynb`

**Mục đích**: Làm sạch dữ liệu, xử lý outliers

**Thực hiện**:

1. Mở notebook `02_Data_Preprocessing.ipynb`
2. Chạy tất cả các cells
3. Kết quả:
   - Dữ liệu đã được làm sạch
   - Xử lý outliers (nếu có)
   - Dữ liệu được lưu vào `data/processed/`

**Kết quả mong đợi**:

- Dataset sạch, sẵn sàng cho feature engineering

---

#### **⚙️ Notebook 3: Feature Engineering**

**File**: `notebooks/03_Feature_Engineering.ipynb`

**Mục đích**: Tạo các đặc trưng mới

**Thực hiện**:

1. Mở notebook `03_Feature_Engineering.ipynb`
2. Chạy tất cả các cells
3. Tạo các features:
   - `total_duration`, `total_pages`
   - `admin_ratio`, `informational_ratio`, `product_ratio`
   - `duration_per_page`
   - Biến mùa vụ (`is_q4`, `is_weekend`)
   - Tương tác (`PageValues * ProductRelated`)

**Kết quả mong đợi**:

- Dataset với các features mới
- Features được lưu vào `data/features/`

---

#### **🎯 Notebook 4: Model Training**

**File**: `notebooks/04_Model_Training.ipynb`

**Mục đích**: Huấn luyện Decision Tree và Random Forest

**Thực hiện**:

1. Mở notebook `04_Model_Training.ipynb`
2. Chạy tất cả các cells
3. Models được huấn luyện:
   - Decision Tree với `class_weight='balanced'`
   - Random Forest với hyperparameter tuning
4. Models được lưu vào `models/`:
   - `decision_tree.pkl`
   - `random_forest.pkl`
   - `preprocessor.pkl`

**Kết quả mong đợi**:

- 2 models đã được train
- Models đã được lưu để sử dụng sau

**⏱️ Lưu ý**: Training có thể mất vài phút, đặc biệt là Random Forest với hyperparameter tuning.

---

#### **📈 Notebook 5: Model Evaluation**

**File**: `notebooks/05_Model_Evaluation.ipynb`

**Mục đích**: Đánh giá và so sánh hiệu suất models

**Thực hiện**:

1. Mở notebook `05_Model_Evaluation.ipynb`
2. Chạy tất cả các cells
3. Kết quả đánh giá:
   - Metrics: Precision, Recall, F1-score, ROC-AUC, PR-AUC
   - Confusion Matrix
   - ROC Curve
   - Precision-Recall Curve
   - So sánh 2 models

**Kết quả mong đợi**:

- Bảng metrics chi tiết
- Biểu đồ so sánh 2 models
- Kết quả được lưu vào `results/figures/` và `results/metrics/`

---

#### **🔍 Notebook 6: Model Explanation**

**File**: `notebooks/06_Model_Explanation.ipynb`

**Mục đích**: Giải thích predictions bằng SHAP

**Thực hiện**:

1. Mở notebook `06_Model_Explanation.ipynb`
2. Chạy tất cả các cells
3. Kết quả:
   - Feature importance
   - SHAP summary plot
   - SHAP waterfall plot cho từng prediction
   - Permutation importance

**Kết quả mong đợi**:

- Hiểu được features nào quan trọng nhất
- Giải thích được tại sao model đưa ra prediction cụ thể
- Kết quả được lưu vào `results/figures/`

**⏱️ Lưu ý**: SHAP có thể chạy chậm với Random Forest lớn. Nếu quá lâu, có thể giảm `n_samples` trong SHAP explainer.

---

## 🎯 Chạy nhanh (Quick Start)

Nếu bạn muốn chạy toàn bộ pipeline một lần:

```powershell
# 1. Khởi động Jupyter
jupyter lab

# 2. Trong Jupyter Lab, mở lần lượt các notebooks:
#    - 01_EDA.ipynb → Run All
#    - 02_Data_Preprocessing.ipynb → Run All
#    - 03_Feature_Engineering.ipynb → Run All
#    - 04_Model_Training.ipynb → Run All
#    - 05_Model_Evaluation.ipynb → Run All
#    - 06_Model_Explanation.ipynb → Run All
```

---

## ⚠️ Lưu ý quan trọng

1. **Chạy tuần tự**: Phải chạy các notebooks theo thứ tự 01 → 06 vì:

   - Notebook sau phụ thuộc vào kết quả notebook trước
   - Ví dụ: Notebook 04 cần dữ liệu đã được preprocess và feature engineering

2. **Thư mục output**: Đảm bảo các thư mục sau tồn tại:

   - `data/processed/`
   - `data/features/`
   - `models/`
   - `results/figures/`
   - `results/metrics/`
   - `results/reports/`

   Nếu chưa có, tạo bằng:

   ```powershell
   mkdir -p data/processed data/features models results/figures results/metrics results/reports
   ```

3. **Dữ liệu đầu vào**: Đảm bảo file `data/raw/online_shoppers_intention.csv` tồn tại

4. **Memory**: Với dataset lớn và Random Forest, có thể cần RAM ≥ 4GB

5. **Time**: Toàn bộ pipeline có thể mất 10-30 phút tùy máy tính

---

## 🐛 Xử lý lỗi thường gặp

### **Lỗi: ModuleNotFoundError**

```
ModuleNotFoundError: No module named 'src'
```

**Giải pháp**: Đảm bảo đang chạy notebook từ thư mục `notebooks/`, và file `src/` nằm ở cùng cấp với thư mục `notebooks/`

### **Lỗi: FileNotFoundError**

```
FileNotFoundError: data/raw/online_shoppers_intention.csv
```

**Giải pháp**: Kiểm tra đường dẫn file trong `config/config.yaml` hoặc đảm bảo file CSV tồn tại

### **Lỗi: MemoryError khi training**

**Giải pháp**:

- Giảm `n_estimators` trong Random Forest (ví dụ: 50 thay vì 100)
- Giảm kích thước dataset bằng cách lấy mẫu
- Đóng các ứng dụng khác để giải phóng RAM

### **Lỗi: SHAP chạy quá lâu**

**Giải pháp**:

- Giảm số lượng samples trong SHAP explainer
- Chỉ chạy SHAP cho Decision Tree (nhanh hơn Random Forest)

---

## 📊 Kết quả mong đợi

Sau khi chạy xong tất cả notebooks, bạn sẽ có:

✅ **Models**: `models/decision_tree.pkl`, `models/random_forest.pkl`

✅ **Kết quả đánh giá**:

- Metrics trong `results/metrics/evaluation_results.json`
- Biểu đồ trong `results/figures/`

✅ **Báo cáo**: Model report trong `results/reports/`

✅ **Hiểu biết về dữ liệu**:

- Features quan trọng nhất
- Insights về hành vi người dùng
- Giải thích được predictions

---

## 🎓 Tips

1. **Đọc kỹ comments**: Mỗi notebook có comments giải thích từng bước
2. **Chạy từng cell**: Nếu gặp lỗi, chạy từng cell để tìm lỗi cụ thể
3. **Lưu output**: Sau mỗi notebook, lưu kết quả để tránh phải chạy lại
4. **Experiment**: Thử thay đổi hyperparameters để cải thiện kết quả

---

## 📞 Hỗ trợ

Nếu gặp vấn đề, kiểm tra:

1. Đã cài đặt đủ dependencies chưa (`pip list`)
2. Đang ở đúng thư mục dự án chưa
3. File dữ liệu có tồn tại không
4. Các thư mục output đã được tạo chưa

---

**Chúc bạn thành công với dự án! 🚀**
