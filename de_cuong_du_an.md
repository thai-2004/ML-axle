## Đề cương: Dự đoán Khả năng Mua Hàng trên Website

### 1. Hiểu bài toán & dữ liệu

- **Mục tiêu**: dự đoán biến `Revenue` (`TRUE/FALSE`) để nhận biết phiên truy cập có kết thúc bằng mua hàng hay không.
- **Giá trị thực tiễn**: tối ưu chiến dịch marketing, cá nhân hóa ưu đãi, đánh giá chất lượng nguồn traffic.
- **Đặc điểm dữ liệu**: 12 330 phiên truy cập, 1 908 phiên mua (≈15,5%). Đây là bài toán chuyển đổi người dùng, mức mất cân bằng vừa phải.
  **Giải thích các cột trong dataset** (dữ liệu đã được làm sạch, không có missing values):

1. **Nhóm hoạt động phiên (session activity)**:

   - `Administrative` (int): số trang administrative mà người dùng truy cập (0-27)
   - `Administrative_Duration` (float): tổng thời gian (giây) trên các trang administrative
   - `Informational` (int): số trang thông tin truy cập (0-24)
   - `Informational_Duration` (float): tổng thời gian (giây) trên các trang thông tin
   - `ProductRelated` (int): số trang sản phẩm truy cập (0-705)
   - `ProductRelated_Duration` (float): tổng thời gian (giây) trên các trang sản phẩm

2. **Nhóm chỉ số hiệu suất (performance metrics)**:

   - `BounceRates` (float): tỷ lệ thoát ngay sau 1 trang (0.0-0.2), càng cao càng không tốt
   - `ExitRates` (float): tỷ lệ thoát tại mỗi trang (0.0-0.2), phản ánh chất lượng trải nghiệm
   - `PageValues` (float): giá trị trung bình của trang trong phiên (0-361.76), càng cao càng có tiềm năng mua hàng
   - `SpecialDay` (float): độ gần với ngày đặc biệt (0.0-1.0), ví dụ Black Friday, Valentine

3. **Nhóm bối cảnh phiên (session context)**:

   - `Month` (object): tháng của phiên (Feb, Mar, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec)
   - `OperatingSystems` (int): hệ điều hành (1-8), mã hóa số
   - `Browser` (int): trình duyệt (1-13), mã hóa số
   - `Region` (int): khu vực địa lý (1-9), mã hóa số
   - `TrafficType` (int): nguồn traffic (1-20), ví dụ organic search, referral, direct
   - `VisitorType` (object): loại visitor - `Returning_Visitor`, `New_Visitor`, hoặc `Other`
   - `Weekend` (bool): `TRUE` nếu là cuối tuần, `FALSE` nếu là ngày thường

4. **Biến mục tiêu (target variable)**:
   - `Revenue` (bool): `TRUE` nếu phiên kết thúc bằng mua hàng, `FALSE` nếu không mua

### 2. Tiền xử lý & tạo đặc trưng

- **Làm sạch dữ liệu**: kiểm tra giá trị thiếu, ngoại lai; xử lý phiên có thời lượng bất thường.
- **Chuyển đổi biến**: chuẩn hóa thời lượng/đếm, mã hóa biến phân loại (`Month`, `Region`, `TrafficType`, `VisitorType`) bằng One-Hot hoặc Target Encoding.
- **Đặc trưng mới**: `total_duration`, `total_pages`, tỷ lệ thời lượng hoặc số trang theo loại (`admin_ratio`, `informational_ratio`, `product_ratio`), `duration_per_page`, biến mùa vụ (`is_q4`, `is_weekend`), tương tác (`PageValues * ProductRelated`, `SpecialDay * PageValues`).
- **Chuẩn bị tập dữ liệu**: chia train/validation/test theo trình tự thời gian; scale biến liên tục khi cần.

### 3. Xử lý mất cân bằng lớp

- **Trọng số lớp**: sử dụng `class_weight='balanced'` cho Logistic Regression hoặc cây quyết định.
- **Tái lấy mẫu**: thử SMOTE/SMOTE-NC hoặc undersampling nhẹ nếu cần cải thiện Recall.
- **Đánh giá**: áp dụng cross-validation giữ trật tự thời gian để theo dõi tác động của cân bằng lớp.

### 4. Lựa chọn & huấn luyện mô hình

- **Cây quyết định đơn**: Decision Tree với `class_weight='balanced'`, tuning độ sâu, `min_samples_split`/`min_samples_leaf` để tránh overfit; xem xét cost-complexity pruning.
- **Rừng ngẫu nhiên**: tối ưu số cây (`n_estimators`), `max_depth`, `max_features`; tận dụng feature importance và permutation importance để khám phá yếu tố ảnh hưởng.
- **So sánh hai hướng**: đánh giá chênh lệch bias-variance giữa cây đơn và Random Forest, ghi nhận trường hợp cây đơn dễ giải thích và rừng ngẫu nhiên cho hiệu suất cao hơn.
- **Pipeline**: sử dụng `ColumnTransformer` chỉ để mã hóa biến phân loại; mô hình cây không cần scale; huấn luyện với cross-validation 5-fold theo thời gian.

### 5. Đánh giá mô hình

- **Metric chính**: Precision, Recall, F1-score cho lớp `Revenue=TRUE`.
- **Bổ sung**: ROC-AUC, PR-AUC để đánh giá khả năng phân biệt tổng quan; confusion matrix để quan sát FP/FN.
- **Phân tích chi tiết**: kiểm tra hiệu suất theo tháng, nguồn traffic, loại visitor; calibration plot nếu dùng xác suất cho quyết định marketing.

### 6. Triển khai & giải thích

- **Điểm chuyển đổi**: xuất xác suất 0–1 làm “conversion score” để ưu tiên chăm sóc khách hàng tiềm năng.
- **Ứng dụng demo**: xây dựng notebook hoặc app (Streamlit/Flask) cho phép nhập đặc trưng phiên và nhận khuyến nghị hành động (giữ lại, ưu tiên, giảm ưu đãi).
- **Giải thích mô hình**: sử dụng permutation importance, SHAP để diễn giải toàn cục; SHAP/LIME cho từng phiên (ví dụ “PageValues cao và SpecialDay góp phần tăng xác suất mua”).
- **Giám sát**: theo dõi drift phân phối đặc trưng, tỷ lệ mua theo tháng; thiết lập lịch cập nhật định kỳ (hàng quý).

### 7. Kế hoạch mở rộng & thu thập thêm dữ liệu

- **Bổ sung dữ liệu**: nếu có thể, thu thập ID người dùng để tạo đặc trưng hành vi theo lịch sử; thêm thông tin chiến dịch marketing hoặc chi phí traffic.
- **Hợp nhất nguồn**: kết hợp dữ liệu hậu mua (hoàn hàng, đánh giá) để tạo nhiệm vụ dự đoán giá trị vòng đời.
- **Triển khai thực tế**: đóng gói thành dịch vụ API realtime phục vụ hệ thống gợi ý/CRM; thiết lập dashboard giám sát hiệu suất.

### 8. Bước hành động tiếp theo

- Thực hiện EDA chi tiết và kiểm định các giả thuyết về hành vi mua.
- Xác định ngưỡng metric mục tiêu (ví dụ F1 ≥ 0,60, Precision ≥ 0,50).
- Thiết lập notebook/pipeline baseline với Decision Tree và báo cáo kết quả, sau đó so sánh với Random Forest.
- Ghi nhận thí nghiệm, điều chỉnh feature engineering và chiến lược cân bằng lớp dựa trên phản hồi.
