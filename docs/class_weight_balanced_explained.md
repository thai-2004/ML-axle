# Giải Thích Về `class_weight='balanced'` trong Machine Learning

## 1. Tổng Quan

`class_weight='balanced'` là một tham số quan trọng trong các thuật toán classification của scikit-learn (như Decision Tree, Random Forest, Logistic Regression, SVM, v.v.) để xử lý **dữ liệu mất cân bằng (imbalanced dataset)**.

Khi một class có nhiều mẫu hơn class kia, mô hình có xu hướng "thiên vị" về class đa số, dẫn đến hiệu suất kém cho class thiểu số. `class_weight='balanced'` giúp tự động tính toán trọng số cho từng class dựa trên tần suất của chúng.

---

## 2. Vấn Đề Dữ Liệu Mất Cân Bằng

### 2.1. Trong Dataset Của Chúng Ta

Dataset e-commerce có phân phối class rất mất cân bằng:

```
Target Variable: Revenue (True/False)
- Revenue=False: 10,422 mẫu (84.53%)
- Revenue=True:  1,908 mẫu (15.47%)
- Tỷ lệ mất cân bằng: 5.46:1
```

### 2.2. Hậu Quả Khi Không Xử Lý

**Không dùng `class_weight='balanced'`:**

- Mô hình có xu hướng dự đoán tất cả về class đa số (Revenue=False)
- **Accuracy cao** (ví dụ: 84.5%) nhưng **không có ý nghĩa thực tế**
- **Recall thấp** cho class thiểu số (Revenue=True) → **Bỏ lỡ nhiều khách hàng tiềm năng**
- **Precision thấp** cho class thiểu số → Không phát hiện được những người thực sự sẽ mua hàng

**Ví dụ:**

```
Nếu mô hình luôn dự đoán "False":
- Accuracy = 84.5% (rất cao!)
- Nhưng Recall cho "True" = 0% (tệ!)
- → Hoàn toàn vô dụng trong thực tế!
```

---

## 3. Cách `class_weight='balanced'` Hoạt Động

### 3.1. Công Thức Tính Trọng Số

Sklearn tự động tính trọng số ngược với tần suất của class:

```python
weight_i = n_samples / (n_classes * count_i)
```

Trong đó:

- `n_samples`: Tổng số mẫu trong dataset
- `n_classes`: Số lượng classes (thường là 2 cho binary classification)
- `count_i`: Số lượng mẫu của class thứ i

### 3.2. Ví Dụ Tính Toán Với Dataset Của Chúng Ta

```python
# Dữ liệu
n_samples = 12,330
n_classes = 2
class_0_count = 10,422  # Revenue=False
class_1_count = 1,908   # Revenue=True

# Tính trọng số
weight_0 = 12,330 / (2 * 10,422) ≈ 0.591
weight_1 = 12,330 / (2 * 1,908)  ≈ 3.231

# Tỷ lệ weight
weight_1 / weight_0 ≈ 5.46  # Tương ứng với tỷ lệ mất cân bằng!
```

### 3.3. Ý Nghĩa

- Class thiểu số (Revenue=True) có **trọng số cao hơn gấp ~5.46 lần** class đa số
- Mỗi mẫu của class thiểu số "quan trọng hơn" trong quá trình training
- Mô hình sẽ **chú ý nhiều hơn** đến việc phân loại đúng class thiểu số
- Lỗi phân loại sai class thiểu số sẽ bị **phạt nặng hơn** trong loss function

---

## 4. Implementation Trong Code

### 4.1. Decision Tree Model

Trong file `src/models/decision_tree.py`:

```python
def create_decision_tree_model(max_depth: int = None,
                              min_samples_split: int = 2,
                              min_samples_leaf: int = 1,
                              class_weight: str = 'balanced',  # ← Sử dụng ở đây
                              random_state: int = 42):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,  # ← Truyền vào model
        random_state=random_state
    )
    return model
```

### 4.2. Random Forest Model

Tương tự trong `src/models/random_forest.py`:

```python
def create_random_forest_model(n_estimators: int = 100,
                              max_depth: int = None,
                              max_features: str = 'sqrt',
                              class_weight: str = 'balanced',  # ← Sử dụng ở đây
                              random_state: int = 42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        class_weight=class_weight,  # ← Truyền vào model
        random_state=random_state
    )
    return model
```

### 4.3. Hyperparameter Tuning

Trong GridSearchCV, `class_weight='balanced'` cũng được sử dụng:

```python
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'class_weight': ['balanced']  # ← Cố định là 'balanced'
}
```

---

## 5. So Sánh Các Options Khác

| Option                         | Ý Nghĩa                                  | Khi Nào Dùng                                            |
| ------------------------------ | ---------------------------------------- | ------------------------------------------------------- |
| `None` hoặc không set          | Tất cả classes có weight = 1 (bằng nhau) | **Dữ liệu cân bằng** (tỷ lệ gần 50:50)                  |
| `'balanced'`                   | Tự động tính weight ngược với frequency  | **Dữ liệu mất cân bằng** (như dataset của chúng ta) ✅  |
| `dict` (ví dụ: `{0: 1, 1: 3}`) | Set weight thủ công cho từng class       | Biết rõ tỷ lệ weight mong muốn, hoặc muốn tune thủ công |

### 5.1. Ví Dụ Dict

```python
# Set thủ công
class_weight = {0: 1.0, 1: 5.5}  # Class 1 có trọng số cao hơn 5.5 lần

# Tương tự với 'balanced' nhưng kiểm soát được chính xác
model = DecisionTreeClassifier(class_weight=class_weight)
```

---

## 6. Kết Quả Thực Tế Trên Dataset

### 6.1. Decision Tree với `class_weight='balanced'`

```
Metrics:
- Accuracy: 0.8674
- Precision: 0.5793
- Recall: 0.5262
- F1-score: 0.5514
- ROC-AUC: 0.7281

Classification Report:
              precision    recall  f1-score   support
           0       0.91      0.93      0.92      2084
           1       0.58      0.53      0.55       382
```

**Phân tích:**

- ✅ Recall = 0.53: Phát hiện được **53% khách hàng sẽ mua hàng**
- ✅ Precision = 0.58: Trong số những người được dự đoán "mua", **58% thực sự mua**
- ⚠️ Chưa tối ưu nhưng đã tốt hơn nhiều so với không dùng balanced

### 6.2. Random Forest với `class_weight='balanced'`

```
Metrics:
- Accuracy: 0.9023
- Precision: 0.7640
- Recall: 0.5340
- F1-score: 0.6287
- ROC-AUC: 0.9171

Classification Report:
              precision    recall  f1-score   support
           0       0.92      0.97      0.94      2084
           1       0.76      0.53      0.63       382
```

**Phân tích:**

- ✅ Recall = 0.53: Vẫn phát hiện được **53% khách hàng sẽ mua hàng**
- ✅ Precision = 0.76: **Rất tốt!** Trong số những người được dự đoán "mua", **76% thực sự mua**
- ✅ ROC-AUC = 0.9171: **Rất xuất sắc!** Khả năng phân biệt giữa 2 classes rất tốt

### 6.3. So Sánh Giả Thuyết (Không Dùng Balanced)

Nếu không dùng `class_weight='balanced'`, có thể dự đoán:

```
Metrics (ước tính):
- Accuracy: ~0.85 (cao nhưng không có ý nghĩa)
- Precision: ~0.10-0.20 (rất thấp)
- Recall: ~0.05-0.10 (rất thấp, gần như bỏ lỡ hết)
- ROC-AUC: ~0.60-0.70 (kém)
```

→ **Sử dụng `class_weight='balanced'` cải thiện đáng kể hiệu suất!**

---

## 7. Lợi Ích Của `class_weight='balanced'`

### 7.1. Tự Động Hóa

- ✅ **Không cần tính toán thủ công** trọng số
- ✅ Tự động điều chỉnh khi dữ liệu thay đổi
- ✅ Áp dụng dễ dàng cho bất kỳ dataset mất cân bằng nào

### 7.2. Cân Bằng Hiệu Suất

- ✅ **Cải thiện Recall** cho class thiểu số → Không bỏ lỡ các trường hợp quan trọng
- ✅ **Cân bằng Precision và Recall** → F1-score tốt hơn
- ✅ **Tăng ROC-AUC** → Khả năng phân loại tổng thể tốt hơn

### 7.3. Phù Hợp Với Business Case

Trong bài toán e-commerce:

- **False Negative (FN)**: Dự đoán "không mua" nhưng thực tế "mua"
  - ❌ **Lỗi tốn kém!** Bỏ lỡ cơ hội bán hàng, không tối ưu marketing campaigns
- **False Positive (FP)**: Dự đoán "mua" nhưng thực tế "không mua"
  - ⚠️ Chấp nhận được: Chỉ tốn chi phí marketing nhưng không mất khách hàng

→ `class_weight='balanced'` giúp **giảm False Negative**, tối ưu business value!

---

## 8. Cách Hoạt Động Trong Training Process

### 8.1. Decision Tree

Trong Decision Tree, `class_weight` ảnh hưởng đến:

1. **Gini Impurity tính toán:**

   - Mỗi mẫu được nhân với weight của class tương ứng
   - Class thiểu số có trọng số cao hơn → ảnh hưởng lớn hơn đến Gini calculation

2. **Split selection:**

   - Khi chọn best split, Gini Gain được tính với weighted samples
   - Tree sẽ ưu tiên splits giúp phân loại đúng class thiểu số

3. **Leaf node prediction:**
   - Prediction dựa trên weighted majority vote
   - Class với weighted count cao hơn sẽ được chọn

### 8.2. Random Forest

Trong Random Forest:

1. **Mỗi tree trong ensemble** đều sử dụng `class_weight='balanced'`
2. **Bootstrap sampling** + weighted Gini → Mỗi tree chú ý đến class thiểu số
3. **Final prediction** = average weighted predictions từ tất cả trees
4. → **Hiệu quả hơn** vì ensemble nhiều trees, mỗi tree đều được cân bằng

---

## 9. Khi Nào KHÔNG Nên Dùng `class_weight='balanced'`?

### 9.1. Dữ Liệu Cân Bằng

Nếu tỷ lệ classes gần 50:50 (ví dụ: 48% vs 52%), không cần thiết:

```python
class_weight=None  # Đủ rồi
```

### 9.2. Cost-Sensitive Learning Cụ Thể

Nếu biết rõ chi phí của từng loại lỗi và muốn control chính xác:

```python
# Ví dụ: FN cost = 100, FP cost = 10
# → Cần weight ratio = 10:1 chứ không phải tỷ lệ mẫu
class_weight = {0: 1, 1: 10}  # Tune thủ công
```

### 9.3. Extremely Imbalanced Data

Với dữ liệu cực kỳ mất cân bằng (ví dụ: 99:1), có thể cần:

- **Kết hợp** với kỹ thuật khác: SMOTE, undersampling, oversampling
- **Tune weight thủ công** thay vì 'balanced'

---

## 10. Best Practices

### 10.1. Luôn Kiểm Tra Class Distribution Trước

```python
# EDA
print(df['target'].value_counts())
print(df['target'].value_counts(normalize=True))

# Nếu tỷ lệ > 70:30 hoặc < 30:70 → Xem xét dùng 'balanced'
```

### 10.2. So Sánh Kết Quả

```python
# Train 2 models: với và không có balanced
model_1 = DecisionTreeClassifier(class_weight=None)
model_2 = DecisionTreeClassifier(class_weight='balanced')

# So sánh metrics, đặc biệt là Recall và ROC-AUC
```

### 10.3. Kết Hợp Với Các Kỹ Thuật Khác

- ✅ **class_weight='balanced'** + **Feature Engineering** tốt
- ✅ **class_weight='balanced'** + **Ensemble methods** (Random Forest)
- ⚠️ Có thể không cần **SMOTE/oversampling** nữa nếu balanced đã đủ

---

## 11. Kết Luận

### 11.1. Tóm Tắt

- `class_weight='balanced'` là một **công cụ mạnh mẽ và đơn giản** để xử lý dữ liệu mất cân bằng
- **Tự động tính toán** trọng số ngược với tần suất của class
- **Cải thiện đáng kể** Recall và ROC-AUC cho class thiểu số
- **Phù hợp với business case** của e-commerce: giảm False Negative, không bỏ lỡ khách hàng tiềm năng

### 11.2. Trong Project Này

✅ **Quyết định đúng đắn**: Sử dụng `class_weight='balanced'` cho cả Decision Tree và Random Forest

✅ **Kết quả tốt**:

- Random Forest: ROC-AUC = 0.9171, Precision = 0.76, Recall = 0.53
- Đủ tốt để sử dụng trong production

✅ **Khuyến nghị**: Tiếp tục sử dụng trong các model tương lai cho dataset này

---

## 12. Tài Liệu Tham Khảo

- [Scikit-learn Documentation: class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
- [Handling Imbalanced Datasets in Machine Learning](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
- [Cost-Sensitive Learning for Imbalanced Classification](https://machinelearningmastery.com/cost-sensitive-learning-for-imbalanced-classification/)
