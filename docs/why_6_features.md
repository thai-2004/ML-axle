# Tại Sao Lại Tính Information Gain Cho 6 Features?

## 1. Bối Cảnh Ví Dụ

Ví dụ này sử dụng **dataset mẫu nhỏ** (chỉ 5 mẫu) để minh họa cách tính Information Gain. Đây là ví dụ **giáo dục** nhằm:

- Giải thích các trường hợp khác nhau khi tính Information Gain
- Minh họa cách xử lý numerical features (cần threshold)
- Minh họa cách xử lý categorical/binary features
- So sánh các feature với nhau để tìm best split

---

## 2. Lý Do Chọn 6 Features Cụ Thể

6 features được chọn để **minh họa các kịch bản khác nhau**:

### Feature 1: **Administrative** (Numerical)

- **Mục đích:** Minh họa numerical feature cần threshold
- **Đặc điểm:** Các giá trị [3, 2, 2, 0, 0] → cần thử nhiều threshold
- **Kết quả:** Gain = 0.171 bits (trung bình)

### Feature 2: **PageValues** (Numerical)

- **Mục đích:** Minh họa feature có nhiều threshold khác nhau cho kết quả khác nhau
- **Đặc điểm:**
  - Threshold = 0.5 → Gain = 0.322 bits
  - Threshold = 10.0 → Gain = 0.722 bits (tốt nhất!)
- **Kết quả:** Best Gain = 0.722 bits (rất cao)

### Feature 3: **VisitorType** (Categorical)

- **Mục đích:** Minh họa trường hợp **KHÔNG THỂ SPLIT**
- **Đặc điểm:** Tất cả 5 mẫu đều có giá trị = 2
- **Kết quả:** Gain = 0 bits (không phân chia được)
- **Bài học:** Feature không có variance → không hữu ích

### Feature 4: **Weekend** (Binary)

- **Mục đích:** Minh hĩa binary feature cho **split hoàn hảo**
- **Đặc điểm:**
  - Weekend = 0: Row 1 → Revenue = [1] (100% mua)
  - Weekend = 1: Rows 2-5 → Revenue = [0,0,0,0] (0% mua)
- **Kết quả:** Gain = 0.722 bits (split hoàn hảo, tạo pure nodes)

### Feature 5: **BounceRates** (Numerical)

- **Mục đích:** Minh họa numerical feature với threshold tối ưu
- **Đặc điểm:** Threshold = 0.01 tạo split hợp lý
- **Kết quả:** Gain = 0.322 bits (trung bình)

### Feature 6: **ProductRelated** (Numerical)

- **Mục đích:** Minh họa numerical feature không tối ưu lắm
- **Đặc điểm:** Threshold = 29.5 không tạo split tốt
- **Kết quả:** Gain = 0.171 bits (thấp)

---

## 3. Tóm Tắt So Sánh

| Feature            | Loại        | Best Gain | Ý Nghĩa                         |
| ------------------ | ----------- | --------- | ------------------------------- |
| **PageValues**     | Numerical   | 0.722     | Tốt nhất (với threshold = 10.0) |
| **Weekend**        | Binary      | 0.722     | Tốt nhất (split hoàn hảo)       |
| **BounceRates**    | Numerical   | 0.322     | Trung bình                      |
| **Administrative** | Numerical   | 0.171     | Thấp                            |
| **ProductRelated** | Numerical   | 0.171     | Thấp                            |
| **VisitorType**    | Categorical | 0.0       | Không split được                |

**Kết luận:** Cả **PageValues** và **Weekend** đều cho Gain cao nhất (0.722). Decision Tree sẽ chọn một trong hai làm root node.

---

## 4. Trong Thực Tế

### Dataset thực tế có **NHIỀU HƠN** 6 features:

Theo `de_cuong_du_an.md`, dataset thực tế có:

1. **Session Activity:** Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration (6 features)

2. **Performance Metrics:** BounceRates, ExitRates, PageValues, SpecialDay (4 features)

3. **Session Context:** Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend (7 features)

**Tổng cộng:** ~17 features ban đầu, cộng thêm các engineered features có thể lên đến 20-30 features.

### Quy Trình Thực Tế:

1. **Tính Information Gain cho TẤT CẢ features** trong dataset
2. **Với mỗi numerical feature:** Thử nhiều threshold candidates
3. **Với mỗi categorical feature:** Tính Gain trực tiếp
4. **Chọn feature + threshold có Gain CAO NHẤT** làm best split

---

## 5. Tại Sao Ví Dụ Chỉ Dùng 6 Features?

1. **Minh họa đủ các trường hợp:**

   - ✅ Numerical feature với threshold tốt (PageValues)
   - ✅ Numerical feature với threshold tệ (ProductRelated)
   - ✅ Binary feature (Weekend)
   - ✅ Categorical feature không dùng được (VisitorType)

2. **Dễ theo dõi:** Dataset nhỏ (5 mẫu) dễ tính toán thủ công

3. **Giáo dục:** Giúp người học hiểu các kịch bản khác nhau mà không bị quá tải

4. **So sánh rõ ràng:** Thấy được sự khác biệt giữa các feature (từ 0.0 đến 0.722)

---

## 6. Kết Luận

6 features trong ví dụ được chọn **có chủ đích** để:

- ✅ Minh họa đủ các loại feature (numerical, categorical, binary)
- ✅ Minh họa các trường hợp khác nhau (split tốt, split tệ, không split được)
- ✅ Dễ tính toán và theo dõi với dataset nhỏ
- ✅ Giúp hiểu được cơ chế Decision Tree chọn feature nào

**Trong thực tế:** Decision Tree sẽ tính Information Gain cho **TẤT CẢ features** trong dataset (có thể 20-30 features), không chỉ 6 features như trong ví dụ.

---

## Tài Liệu Tham Khảo

- Xem thêm: `docs/gini_explained.md` để hiểu cách tính Gini
- Code implementation: `src/utils/gain_calculator.py`
- Dataset thực tế: `de_cuong_du_an.md` (mục 1: Hiểu bài toán & dữ liệu)
