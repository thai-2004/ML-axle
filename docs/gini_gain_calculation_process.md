# Mô Hình Hóa Quy Trình Tính Gini Gain Cho Tất Cả Features

## 1. Tổng Quan Quy Trình

Khi Decision Tree cần chọn best split tại một node, nó phải:

1. **Tính Gini Gain cho TẤT CẢ features** trong dataset (34 features sau feature engineering)
2. **Với mỗi feature:** Thử tất cả các cách split khả thi
3. **So sánh** tất cả các Gini Gain
4. **Chọn** feature + threshold có **Gini Gain cao nhất**

---

## 2. Flowchart Quy Trình

```
START: Node cần split
  │
  ├─> Tính Gini(parent) của node hiện tại
  │
  ├─> FOR mỗi feature trong [34 features]:
  │     │
  │     ├─> IF feature là NUMERICAL:
  │     │     │
  │     │     ├─> Sắp xếp giá trị feature
  │     │     │
  │     │     ├─> Tạo các threshold candidates:
  │     │     │     - Trung bình giữa các giá trị liên tiếp
  │     │     │     - Ví dụ: [1, 3, 5] → thresholds: [2, 4]
  │     │     │
  │     │     ├─> FOR mỗi threshold:
  │     │     │     │
  │     │     │     ├─> Split: feature <= threshold vs feature > threshold
  │     │     │     │
  │     │     │     ├─> Tính Weighted Gini(children)
  │     │     │     │
  │     │     │     ├─> Tính Gini Gain = Gini(parent) - Weighted Gini(children)
  │     │     │     │
  │     │     │     └─> Lưu (feature, threshold, gini_gain)
  │     │     │
  │     │     └─> END FOR threshold
  │     │
  │     └─> IF feature là CATEGORICAL/BINARY:
  │           │
  │           ├─> FOR mỗi giá trị unique của feature:
  │           │     │
  │           │     ├─> Split: feature == value vs feature != value
  │           │     │     (hoặc split thành nhiều nhóm nếu nhiều giá trị)
  │           │     │
  │           │     ├─> Tính Weighted Gini(children)
  │           │     │
  │           │     ├─> Tính Gini Gain
  │           │     │
  │           │     └─> Lưu (feature, value, gini_gain)
  │           │
  │           └─> END FOR value
  │
  └─> END FOR feature
      │
      ├─> Tìm (feature, threshold/value) có Gini Gain CAO NHẤT
      │
      ├─> Thực hiện split với feature và threshold/value đó
      │
      └─> RETURN: 2 (hoặc nhiều) child nodes mới

END
```

---

## 3. Công Thức Tính Toán

### 3.1. Gini Impurity của một Node

```python
def gini_impurity(node):
    """
    Tính Gini Impurity của một node

    Args:
        node: List các labels (0 hoặc 1)

    Returns:
        Gini Impurity (0 ≤ Gini ≤ 0.5 cho binary classification)
    """
    if len(node) == 0:
        return 0

    # Đếm số lượng mỗi class
    count_0 = sum(1 for label in node if label == 0)
    count_1 = sum(1 for label in node if label == 1)
    total = len(node)

    # Tính tỷ lệ
    p_0 = count_0 / total
    p_1 = count_1 / total

    # Công thức Gini
    gini = 1 - (p_0**2 + p_1**2)

    return gini
```

### 3.2. Weighted Gini của Children Nodes

```python
def weighted_gini(children):
    """
    Tính Weighted Gini sau khi split thành nhiều nhóm

    Args:
        children: List các nhóm con, mỗi nhóm là list labels

    Returns:
        Weighted Gini Impurity
    """
    total_samples = sum(len(child) for child in children)

    if total_samples == 0:
        return 0

    weighted_sum = 0
    for child in children:
        weight = len(child) / total_samples
        gini_child = gini_impurity(child)
        weighted_sum += weight * gini_child

    return weighted_sum
```

### 3.3. Gini Gain

```python
def gini_gain(parent, children):
    """
    Tính Gini Gain = sự giảm Gini sau khi split

    Args:
        parent: List labels của node cha
        children: List các nhóm con sau split

    Returns:
        Gini Gain (càng cao càng tốt)
    """
    gini_parent = gini_impurity(parent)
    weighted_gini_children = weighted_gini(children)

    gain = gini_parent - weighted_gini_children

    return gain
```

---

## 4. Ví Dụ Minh Họa Cụ Thể

### Dataset Mẫu (5 mẫu để minh họa):

| Row | PageValues | Weekend | BounceRates | Revenue |
| --- | ---------- | ------- | ----------- | ------- |
| 1   | 0.0        | 0       | 0.2         | **1**   |
| 2   | 0.0        | 1       | 0.0         | 0       |
| 3   | 0.0        | 1       | 0.2         | 0       |
| 4   | 0.0        | 1       | 0.05        | 0       |
| 5   | 10.0       | 1       | 0.02        | 0       |

**Node cha:** `[1, 0, 0, 0, 0]` → 1 Revenue=1, 4 Revenue=0

### Bước 1: Tính Gini(parent)

```python
# Node cha: [1, 0, 0, 0, 0]
count_0 = 4
count_1 = 1
total = 5

p_0 = 4/5 = 0.8
p_1 = 1/5 = 0.2

Gini(parent) = 1 - (0.8² + 0.2²)
             = 1 - (0.64 + 0.04)
             = 1 - 0.68
             = 0.32
```

---

### Bước 2: Tính Gini Gain cho từng Feature

#### Feature 1: PageValues (Numerical)

**Giá trị:** `[0.0, 0.0, 0.0, 0.0, 10.0]`

**Threshold candidates:**

- Giữa 0.0 và 10.0 → threshold = `5.0`

**Với threshold = 5.0:**

Split:

- **Left (PageValues ≤ 5.0):** Rows 1,2,3,4 → Revenue: `[1, 0, 0, 0]`
- **Right (PageValues > 5.0):** Row 5 → Revenue: `[0]`

Tính toán:

```python
# Left node: [1, 0, 0, 0]
Gini(left) = 1 - (0.75² + 0.25²) = 1 - (0.5625 + 0.0625) = 0.375

# Right node: [0]
Gini(right) = 1 - (1² + 0²) = 0

# Weighted Gini
Weighted Gini = (4/5) × 0.375 + (1/5) × 0
               = 0.8 × 0.375 + 0
               = 0.3

# Gini Gain
Gini Gain(PageValues, threshold=5.0) = 0.32 - 0.3 = 0.02
```

**Kết quả:** Gini Gain = **0.02** (thấp)

---

#### Feature 2: Weekend (Binary/Categorical)

**Giá trị:** `[0, 1, 1, 1, 1]`

**Split theo Weekend = 0 vs Weekend = 1:**

- **Weekend = 0:** Row 1 → Revenue: `[1]`
- **Weekend = 1:** Rows 2,3,4,5 → Revenue: `[0, 0, 0, 0]`

Tính toán:

```python
# Weekend = 0 node: [1]
Gini(weekend_0) = 1 - (0² + 1²) = 0  # Pure node!

# Weekend = 1 node: [0, 0, 0, 0]
Gini(weekend_1) = 1 - (1² + 0²) = 0  # Pure node!

# Weighted Gini
Weighted Gini = (1/5) × 0 + (4/5) × 0
              = 0

# Gini Gain
Gini Gain(Weekend) = 0.32 - 0 = 0.32
```

**Kết quả:** Gini Gain = **0.32** (rất cao!)

---

#### Feature 3: BounceRates (Numerical)

**Giá trị:** `[0.2, 0.0, 0.2, 0.05, 0.02]`

**Sắp xếp:** `[0.0, 0.02, 0.05, 0.2, 0.2]`

**Threshold candidates:**

- Giữa 0.0 và 0.02 → threshold = `0.01`
- Giữa 0.02 và 0.05 → threshold = `0.035`
- Giữa 0.05 và 0.2 → threshold = `0.125`
- Giữa 0.2 và 0.2 → bỏ qua (giá trị trùng)

**Thử threshold = 0.01:**

Split:

- **Left (BounceRates ≤ 0.01):** Rows 2,5 → Revenue: `[0, 0]`
- **Right (BounceRates > 0.01):** Rows 1,3,4 → Revenue: `[1, 0, 0]`

```python
# Left node: [0, 0]
Gini(left) = 1 - (1² + 0²) = 0

# Right node: [1, 0, 0]
Gini(right) = 1 - (0.67² + 0.33²) = 1 - (0.45 + 0.11) = 0.44

# Weighted Gini
Weighted Gini = (2/5) × 0 + (3/5) × 0.44
               = 0.264

# Gini Gain
Gini Gain(BounceRates, threshold=0.01) = 0.32 - 0.264 = 0.056
```

**Kết quả:** Gini Gain = **0.056** (trung bình)

---

### Bước 3: So Sánh và Chọn Best Split

| Feature     | Threshold/Value | Gini Gain | Ranking |
| ----------- | --------------- | --------- | ------- |
| **Weekend** | 0 vs 1          | **0.32**  | 🥇 #1   |
| BounceRates | 0.01            | 0.056     | #2      |
| PageValues  | 5.0             | 0.02      | #3      |

**Kết luận:**

- ✅ **Best Split:** `Weekend == 0` vs `Weekend == 1`
- ✅ **Gini Gain cao nhất:** 0.32
- ✅ **Tạo ra 2 pure nodes:** Perfect split!

---

## 5. Quy Trình Với Dataset Thực Tế (34 Features)

### 5.1. Ước Tính Số Lượng Tính Toán

**Dataset thực tế:**

- 34 features (sau feature engineering)
- 12,330 mẫu
- Khoảng 20 numerical features, 14 categorical features

**Với mỗi numerical feature:**

- Số threshold candidates ≈ số unique values / 2
- Trung bình: ~50-200 threshold mỗi feature
- Tổng: 20 features × 100 threshold ≈ **2,000 cách split**

**Với mỗi categorical feature:**

- Binary: 1 cách split
- Multi-class (ví dụ Month có 12 giá trị): 12 cách split
- Trung bình: ~5 cách split mỗi feature
- Tổng: 14 features × 5 cách ≈ **70 cách split**

**Tổng số tính toán:**

- 2,000 + 70 = **~2,070 lần tính Gini Gain** cho 1 split!

#### 5.1.1. Giải Thích Chi Tiết: "~20 numerical features × ~100 threshold ≈ 2,000 cách split"

##### 1. **~20 numerical features**

Theo file này:
- **34 features** sau feature engineering
- **~14 categorical features** (như `Month`, `VisitorType`, `Weekend`, `is_q4`, `quarter`, ...)
- **~20 numerical features** = 34 - 14 (ví dụ: `PageValues`, `BounceRates`, `total_duration`, `admin_duration_ratio`, ...)

##### 2. **~100 threshold cho mỗi numerical feature**

Cách tính threshold:
- Với numerical feature, Decision Tree tạo threshold candidates tại điểm giữa các giá trị liên tiếp
- Ví dụ với `BounceRates`: `[0.0, 0.02, 0.05, 0.2, 0.2]` → thresholds: `[0.01, 0.035, 0.125]` = **3 threshold candidates**

**Tại sao ~100 threshold?**
- Nếu một feature có khoảng 200 unique values → có thể tạo ~200 threshold candidates
- Nếu có 50 unique values → ~50 threshold candidates
- **Trung bình**: nếu mỗi numerical feature có khoảng 100 unique values → **~100 threshold candidates**
- Scikit-learn có thể giới hạn số threshold (ví dụ tối đa 256) để tối ưu tốc độ

##### 3. **20 × 100 = 2,000 cách split**

Logic:
- Với mỗi numerical feature, thử mỗi threshold → mỗi threshold = **1 cách split**
- 20 features × 100 threshold/feature = **2,000 cách split cần thử**

**Ví dụ minh họa:**
```
Feature: PageValues
- Threshold 1: PageValues <= 5.0 → Split #1
- Threshold 2: PageValues <= 10.0 → Split #2
- Threshold 3: PageValues <= 15.0 → Split #3
...
- Threshold 100: PageValues <= 500.0 → Split #100
→ Tổng: 100 cách split cho PageValues

Làm tương tự cho 20 numerical features → 20 × 100 = 2,000 cách split
```

##### 4. **Đây chỉ là ước tính**

Trong thực tế:
- Feature có ít unique values → ít threshold hơn
- Feature có nhiều unique values → nhiều threshold hơn
- Scikit-learn có thể giới hạn số threshold để tối ưu

**~100 threshold/feature** là ước tính trung bình dựa trên:
- Dataset có 12,330 mẫu
- Các numerical features có phân bố đa dạng
- Có thể áp dụng giới hạn tối đa (ví dụ 256 thresholds)

##### Tóm tắt

| Thành phần | Giải thích |
|------------|------------|
| **20 numerical features** | 34 tổng features - 14 categorical ≈ 20 numerical |
| **~100 threshold/feature** | Ước tính trung bình (dựa trên số unique values) |
| **2,000 cách split** | 20 × 100 = 2,000 cách split cần thử và tính Gini Gain |

Con số này thể hiện độ phức tạp: để chọn best split tại 1 node, Decision Tree cần thử và tính Gini Gain cho khoảng **2,000 cách split khác nhau**.

---

### 5.2. Pseudo-code Đầy Đủ

```python
def find_best_split(X, y):
    """
    Tìm best split cho một node

    Args:
        X: DataFrame với 34 features
        y: Series labels (Revenue: 0 hoặc 1)

    Returns:
        (best_feature, best_threshold, best_gain)
    """
    # Bước 1: Tính Gini của node cha
    parent_labels = y.tolist()
    gini_parent = gini_impurity(parent_labels)

    # Bước 2: Khởi tạo
    best_gain = -1
    best_feature = None
    best_threshold = None

    # Bước 3: Duyệt qua tất cả features
    for feature_name in X.columns:  # 34 features
        feature_values = X[feature_name].values

        # Kiểm tra loại feature
        if is_numerical(feature_values):
            # NUMERICAL FEATURE
            # Sắp xếp và tạo thresholds
            sorted_values = np.sort(np.unique(feature_values))
            thresholds = [(sorted_values[i] + sorted_values[i+1]) / 2
                         for i in range(len(sorted_values)-1)]

            # Thử từng threshold
            for threshold in thresholds:
                # Split
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                left_labels = y[left_mask].tolist()
                right_labels = y[right_mask].tolist()

                # Tính Gini Gain
                children = [left_labels, right_labels]
                weighted_gini = weighted_gini_impurity(children)
                gain = gini_parent - weighted_gini

                # Cập nhật best
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_name
                    best_threshold = threshold

        else:
            # CATEGORICAL FEATURE
            unique_values = np.unique(feature_values)

            # Thử split theo từng giá trị
            for value in unique_values:
                # Split
                left_mask = feature_values == value
                right_mask = ~left_mask

                left_labels = y[left_mask].tolist()
                right_labels = y[right_mask].tolist()

                # Tính Gini Gain
                children = [left_labels, right_labels]
                weighted_gini = weighted_gini_impurity(children)
                gain = gini_parent - weighted_gini

                # Cập nhật best
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_name
                    best_threshold = value

    # Bước 4: Trả về best split
    return best_feature, best_threshold, best_gain
```

---

## 6. Tối Ưu Hóa (Optimization)

Trong thực tế, các thư viện như scikit-learn sử dụng nhiều kỹ thuật tối ưu:

### 6.1. Chỉ Thử Một Số Threshold Candidates

- Không thử tất cả threshold, chỉ thử một số giá trị đại diện
- Ví dụ: chỉ thử 256 threshold cho mỗi numerical feature

### 6.2. Early Stopping

- Nếu tìm thấy Gini Gain = Gini(parent) → Perfect split, dừng ngay

### 6.3. Parallel Processing

- Tính toán song song cho nhiều features/thresholds

### 6.4. Sampling (trong Random Forest)

- Chỉ xét một tập con features (ví dụ: √34 ≈ 6 features)
- Giảm đáng kể số lượng tính toán

---

## 7. Tóm Tắt

### Quy Trình Chính:

1. ✅ **Tính Gini(parent)** của node cần split
2. ✅ **Với mỗi feature (34 features):**
   - Nếu numerical: thử nhiều threshold → tính Gini Gain
   - Nếu categorical: thử các cách split → tính Gini Gain
3. ✅ **So sánh** tất cả Gini Gain
4. ✅ **Chọn** feature + threshold có **Gini Gain cao nhất**
5. ✅ **Split** node thành children nodes

### Độ Phức Tạp:

- **Số tính toán:** ~2,000-3,000 lần tính Gini Gain cho 1 split
- **Time complexity:** O(n_features × n_samples × log(n_samples))
- **Space complexity:** O(n_samples)

### Kết Quả:

- Decision Tree chọn được **best feature + threshold** tại mỗi node
- Build tree từ root đến leaves
- Mỗi split giảm Gini Impurity nhiều nhất có thể

---

## Tài Liệu Tham Khảo

- `docs/gini_explained.md`: Giải thích chi tiết về Gini Impurity
- `docs/gain_calculation_3_groups.md`: Cách tính Gain cho nhiều nhóm
- `docs/why_6_features.md`: Tại sao ví dụ chỉ dùng 6 features
