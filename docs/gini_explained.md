# Gini Impurity: Cách Tính và Ý Nghĩa

## 1. Công Thức Tính Gini Impurity

### Công thức cơ bản:

```
Gini(S) = 1 - Σ (p_i)²
```

Trong đó:

- `S` = tập dữ liệu tại một node
- `p_i` = tỷ lệ của class i trong node
- Tổng chạy qua tất cả các class

### Với bài toán Binary Classification (2 lớp):

```
Gini(S) = 1 - (p₀² + p₁²)
```

Với:

- `p₀` = tỷ lệ class 0 (ví dụ: False/Negative)
- `p₁` = tỷ lệ class 1 (ví dụ: True/Positive)
- `p₀ + p₁ = 1`

---

## 2. Ví Dụ Tính Toán

### Ví dụ 1: Node "Sạch" (Pure Node)

**Dữ liệu:** 100 mẫu, tất cả đều class 0 (False)

- `p₀ = 100/100 = 1.0`
- `p₁ = 0/100 = 0.0`

```
Gini = 1 - (1.0² + 0.0²)
     = 1 - (1 + 0)
     = 0
```

**Kết quả:** Gini = 0 → Node "sạch", không có tạp chất (impurity)

---

### Ví dụ 2: Node "Hoàn Toàn Lẫn Lộn"

**Dữ liệu:** 100 mẫu, 50 class 0 và 50 class 1

- `p₀ = 50/100 = 0.5`
- `p₁ = 50/100 = 0.5`

```
Gini = 1 - (0.5² + 0.5²)
     = 1 - (0.25 + 0.25)
     = 1 - 0.5
     = 0.5
```

**Kết quả:** Gini = 0.5 → Node có độ lẫn lộn tối đa (với 2 class)

---

### Ví dụ 3: Node "Tạp Chất Trung Bình"

**Dữ liệu:** 1000 mẫu, 850 class 0, 150 class 1

- `p₀ = 850/1000 = 0.85`
- `p₁ = 150/1000 = 0.15`

```
Gini = 1 - (0.85² + 0.15²)
     = 1 - (0.723 + 0.023)
     = 1 - 0.746
     = 0.254
```

**Kết quả:** Gini = 0.254 → Node có một ít tạp chất

---

## 3. Ý Nghĩa của Gini Impurity

### 3.1. Đo Lường Độ "Tạp Chất" (Impurity)

- **Gini = 0**: Node "sạch", chỉ có 1 class → Lý tưởng để dự đoán
- \*_Gini = 0.5_ (với 2 class): Node lẫn lộn nhất, 50-50 → Không\* phân biệt được
- **Gini càng cao**: Node càng lẫn lộn, khó phân loại

### 3.2. Phạm Vi Giá Trị

- **Với 2 class:** Gini ∈ [0, 0.5]

  - Tối thiểu: 0 (pure node)
  - Tối đa: 0.5 (50-50 split)

- **Với C class:** Gini ∈ [0, (C-1)/C]
  - Tối đa khi tất cả các class có tỷ lệ bằng nhau (1/C mỗi class)

### 3.3. Ý Nghĩa Thực Tế trong Decision Tree

1. **Tìm Best Split:**

   - Chọn feature và threshold làm giảm Gini nhiều nhất
   - Split tốt → Gini của các node con thấp hơn node cha

2. **Gini Gain:**

   ```
   Gini Gain = Gini(parent) - Weighted Gini(children)
   ```

   - Gain cao → Split tạo ra các nhóm "sạch" hơn
   - Chọn split có Gini Gain cao nhất

3. **Dừng Split:**
   - Khi Gini = 0 → Node pure, không cần split nữa
   - Hoặc khi đạt độ sâu/độ phức tạp giới hạn

---

## 4. So Sánh Gini vs Entropy

| Đặc điểm      | Gini Impurity                  | Entropy                     |
| ------------- | ------------------------------ | --------------------------- |
| **Công thức** | `1 - Σ(p_i)²`                  | `-Σ(p_i × log₂(p_i))`       |
| **Phạm vi**   | [0, 0.5] cho 2 class           | [0, 1] cho 2 class          |
| **Tính toán** | Nhanh hơn (không cần log)      | Chậm hơn (có log)           |
| **Kết quả**   | Thường tương đương với Entropy | Thường tương đương với Gini |
| **Ứng dụng**  | Random Forest (mặc định)       | ID3, C4.5                   |

**Lưu ý:** Cả hai đều là impurity measures tốt, Gini thường được dùng nhiều hơn vì tính toán nhanh hơn.

---

## 5. Ví Dụ Tính Gini Gain

**Node cha:** 1000 mẫu, 850 class 0, 150 class 1

```
Gini(parent) = 1 - (0.85² + 0.15²) = 0.254
```

**Sau khi split thành 2 nhóm:**

- **Nhóm 1:** 700 mẫu, 630 class 0, 70 class 1
  - Gini(group1) = 1 - (0.90² + 0.10²) = 0.18
- **Nhóm 2:** 300 mẫu, 220 class 0, 80 class 1
  - Gini(group2) = 1 - (0.73² + 0.27²) = 0.394

**Weighted Gini:**

```
Weighted Gini = (700/1000) × 0.18 + (300/1000) × 0.394
              = 0.7 × 0.18 + 0.3 × 0.394
              = 0.126 + 0.118
              = 0.244
```

**Gini Gain:**

```
Gini Gain = 0.254 - 0.244 = 0.010
```

**Kết luận:** Split này giảm Gini một chút (0.010), có thể không phải split tốt nhất.

---

## 6. Tóm Tắt

1. **Gini Impurity** đo lường độ "lẫn lộn" của một node trong Decision Tree
2. **Gini = 0** → Node sạch, chỉ có 1 class (lý tưởng)
3. **Gini càng cao** → Node càng lẫn lộn, khó phân loại
4. **Mục tiêu:** Tìm split làm giảm Gini nhiều nhất (Gini Gain cao)
5. **Gini nhanh hơn Entropy** trong tính toán, thường cho kết quả tương đương

---

## Tài Liệu Tham Khảo

- Xem thêm: `docs/gain_calculation_3_groups.md` để hiểu cách tính Gini Gain cho 3 nhóm
- Code implementation: `src/utils/gain_calculator.py`
