# Cách Tính Gain cho 3 Nhóm Dữ Liệu

## Tổng Quan

Khi split một node trong Decision Tree/Random Forest tạo ra **3 nhóm con** (ví dụ: feature `VisitorType` có 3 giá trị), chúng ta cần tính:

1. **Information Gain** (dùng Entropy)
2. **Gini Gain** (dùng Gini Impurity)

---

## 1. INFORMATION GAIN (Entropy-based)

### Công thức tổng quát:

```
Information Gain = Entropy(parent) - Weighted Entropy(children)
```

### Các bước tính toán:

#### Bước 1: Tính Entropy của Node Cha (Parent)

```
Entropy(S) = -Σ (p_i × log₂(p_i))
```

Trong đó:
- `p_i` = tỷ lệ của class i trong node cha
- Với bài toán binary classification (2 lớp): `p_False` và `p_True`

**Ví dụ:**
- Node cha có 1000 mẫu
- False: 850 mẫu → p_False = 0.85
- True: 150 mẫu → p_True = 0.15

```
Entropy(parent) = -(0.85 × log₂(0.85) + 0.15 × log₂(0.15))
                = -(0.85 × (-0.234) + 0.15 × (-2.737))
                = -(0.85 × (-0.234) + 0.15 × (-2.737))
                = -(-0.199 - 0.411)
                = 0.610 bits
```

#### Bước 2: Tính Entropy cho từng nhóm con (3 nhóm)

**Nhóm 1 (Returning_Visitor):**
- Tổng: 700 mẫu
- False: 630 (90%), True: 70 (10%)
```
Entropy(group1) = -(0.90 × log₂(0.90) + 0.10 × log₂(0.10))
                = -(0.90 × (-0.152) + 0.10 × (-3.322))
                = -(-0.137 - 0.332)
                = 0.469 bits
```

**Nhóm 2 (New_Visitor):**
- Tổng: 250 mẫu
- False: 180 (72%), True: 70 (28%)
```
Entropy(group2) = -(0.72 × log₂(0.72) + 0.28 × log₂(0.28))
                = -(0.72 × (-0.473) + 0.28 × (-1.837))
                = -(-0.341 - 0.514)
                = 0.855 bits
```

**Nhóm 3 (Other):**
- Tổng: 50 mẫu
- False: 40 (80%), True: 10 (20%)
```
Entropy(group3) = -(0.80 × log₂(0.80) + 0.20 × log₂(0.20))
                = -(0.80 × (-0.322) + 0.20 × (-2.322))
                = -(-0.258 - 0.464)
                = 0.722 bits
```

#### Bước 3: Tính Weighted Entropy (Trung bình có trọng số)

```
Weighted Entropy = Σ (|S_i| / |S|) × Entropy(S_i)
```

Trong đó:
- `|S_i|` = số mẫu trong nhóm i
- `|S|` = tổng số mẫu trong node cha

```
Weighted Entropy = (700/1000) × 0.469 + (250/1000) × 0.855 + (50/1000) × 0.722
                 = 0.7 × 0.469 + 0.25 × 0.855 + 0.05 × 0.722
                 = 0.328 + 0.214 + 0.036
                 = 0.578 bits
```

#### Bước 4: Tính Information Gain

```
Information Gain = Entropy(parent) - Weighted Entropy
                 = 0.610 - 0.578
                 = 0.032 bits
```

---

## 2. GINI GAIN (Gini Impurity-based)

### Công thức tổng quát:

```
Gini Gain = Gini(parent) - Weighted Gini(children)
```

### Các bước tính toán:

#### Bước 1: Tính Gini Impurity của Node Cha

```
Gini(S) = 1 - Σ (p_i)²
```

**Ví dụ:**
- Node cha có 1000 mẫu
- False: 850 (85%), True: 150 (15%)

```
Gini(parent) = 1 - (0.85² + 0.15²)
             = 1 - (0.723 + 0.023)
             = 1 - 0.746
             = 0.254
```

#### Bước 2: Tính Gini Impurity cho từng nhóm con

**Nhóm 1 (Returning_Visitor):**
- False: 630 (90%), True: 70 (10%)
```
Gini(group1) = 1 - (0.90² + 0.10²)
             = 1 - (0.81 + 0.01)
             = 1 - 0.82
             = 0.18
```

**Nhóm 2 (New_Visitor):**
- False: 180 (72%), True: 70 (28%)
```
Gini(group2) = 1 - (0.72² + 0.28²)
             = 1 - (0.518 + 0.078)
             = 1 - 0.596
             = 0.404
```

**Nhóm 3 (Other):**
- False: 40 (80%), True: 10 (20%)
```
Gini(group3) = 1 - (0.80² + 0.20²)
             = 1 - (0.64 + 0.04)
             = 1 - 0.68
             = 0.32
```

#### Bước 3: Tính Weighted Gini Impurity

```
Weighted Gini = Σ (|S_i| / |S|) × Gini(S_i)
```

```
Weighted Gini = (700/1000) × 0.18 + (250/1000) × 0.404 + (50/1000) × 0.32
              = 0.7 × 0.18 + 0.25 × 0.404 + 0.05 × 0.32
              = 0.126 + 0.101 + 0.016
              = 0.243
```

#### Bước 4: Tính Gini Gain

```
Gini Gain = Gini(parent) - Weighted Gini
          = 0.254 - 0.243
          = 0.011
```

---

## 3. CODE MÔ PHỎNG TÍNH TOÁN

```python
import numpy as np
import pandas as pd

def calculate_entropy(y):
    """Tính Entropy cho một nhóm dữ liệu"""
    if len(y) == 0:
        return 0
    p = np.bincount(y) / len(y)
    p = p[p > 0]  # Loại bỏ các giá trị 0 để tránh log(0)
    return -np.sum(p * np.log2(p))

def calculate_gini(y):
    """Tính Gini Impurity cho một nhóm dữ liệu"""
    if len(y) == 0:
        return 0
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p ** 2)

def calculate_information_gain(parent_y, groups_y):
    """
    Tính Information Gain khi split thành nhiều nhóm
    
    Parameters:
    -----------
    parent_y : array
        Labels của node cha
    groups_y : list of arrays
        List các arrays chứa labels của từng nhóm con
    """
    # Entropy của node cha
    parent_entropy = calculate_entropy(parent_y)
    
    # Tính weighted entropy của các nhóm con
    total_samples = len(parent_y)
    weighted_entropy = 0
    
    for group_y in groups_y:
        if len(group_y) > 0:
            group_weight = len(group_y) / total_samples
            group_entropy = calculate_entropy(group_y)
            weighted_entropy += group_weight * group_entropy
    
    # Information Gain
    information_gain = parent_entropy - weighted_entropy
    return information_gain

def calculate_gini_gain(parent_y, groups_y):
    """
    Tính Gini Gain khi split thành nhiều nhóm
    """
    # Gini của node cha
    parent_gini = calculate_gini(parent_y)
    
    # Tính weighted gini của các nhóm con
    total_samples = len(parent_y)
    weighted_gini = 0
    
    for group_y in groups_y:
        if len(group_y) > 0:
            group_weight = len(group_y) / total_samples
            group_gini = calculate_gini(group_y)
            weighted_gini += group_weight * group_gini
    
    # Gini Gain
    gini_gain = parent_gini - weighted_gini
    return gini_gain

# Ví dụ với 3 nhóm
# Giả sử: 0 = False, 1 = True

# Node cha: 1000 mẫu (850 False, 150 True)
parent_y = np.array([0]*850 + [1]*150)

# Nhóm 1: Returning_Visitor (700 mẫu: 630 False, 70 True)
group1_y = np.array([0]*630 + [1]*70)

# Nhóm 2: New_Visitor (250 mẫu: 180 False, 70 True)
group2_y = np.array([0]*180 + [1]*70)

# Nhóm 3: Other (50 mẫu: 40 False, 10 True)
group3_y = np.array([0]*40 + [1]*10)

groups_y = [group1_y, group2_y, group3_y]

# Tính Information Gain
info_gain = calculate_information_gain(parent_y, groups_y)
print(f"Information Gain: {info_gain:.4f}")

# Tính Gini Gain
gini_gain = calculate_gini_gain(parent_y, groups_y)
print(f"Gini Gain: {gini_gain:.4f}")

# In chi tiết
print("\n--- Chi tiết Entropy ---")
print(f"Entropy(parent): {calculate_entropy(parent_y):.4f}")
for i, group_y in enumerate(groups_y, 1):
    print(f"Entropy(group{i}): {calculate_entropy(group_y):.4f} (n={len(group_y)})")

print("\n--- Chi tiết Gini ---")
print(f"Gini(parent): {calculate_gini(parent_y):.4f}")
for i, group_y in enumerate(groups_y, 1):
    print(f"Gini(group{i}): {calculate_gini(group_y):.4f} (n={len(group_y)})")
```

---

## 4. Ý NGHĨA CỦA GAIN

### Information Gain / Gini Gain càng cao thì:
- Split càng tốt
- Tách biệt các lớp tốt hơn
- Giảm được nhiều độ "hỗn loạn" (impurity) hơn

### So sánh với 2 nhóm:
- Với **3 nhóm**, chúng ta tính weighted average của **3 giá trị impurity** thay vì 2
- Công thức tổng quát giữ nguyên, chỉ thay đổi số lượng nhóm con

---

## 5. LƯU Ý QUAN TRỌNG

1. **Trọng số (Weight)**: Mỗi nhóm con được tính theo tỷ lệ số mẫu của nó so với node cha
2. **Xử lý nhóm rỗng**: Nếu một nhóm không có mẫu nào (entropy/gini = 0), nó không đóng góp vào weighted average
3. **Log base 2**: Entropy thường dùng log₂, nhưng có thể dùng log tự nhiên (ln) - kết quả chỉ khác một hằng số
4. **Sklearn mặc định**: RandomForest trong sklearn dùng **Gini Impurity** (criterion='gini') chứ không phải Entropy

---

## 6. TRONG SKLEARN

```python
from sklearn.tree import DecisionTreeClassifier

# Dùng Gini (mặc định)
tree_gini = DecisionTreeClassifier(criterion='gini')
tree_gini.fit(X, y)

# Dùng Entropy
tree_entropy = DecisionTreeClassifier(criterion='entropy')
tree_entropy.fit(X, y)
```

Sklearn tự động tính gain cho tất cả các cách split có thể (bao gồm cả 3 nhóm) và chọn split có gain cao nhất.

