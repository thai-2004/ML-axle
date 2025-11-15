# Giải thích Confusion Matrix

## Confusion Matrix là gì?

**Confusion Matrix** (Ma trận nhầm lẫn) là một bảng hiển thị số lượng dự đoán đúng/sai của mô hình, so sánh giữa nhãn thực tế (`y_true`) và nhãn dự đoán (`y_pred`). Đây là công cụ cơ bản và quan trọng nhất để đánh giá hiệu suất mô hình phân loại.

## Cấu trúc Confusion Matrix cho Binary Classification

Với bài toán 2 lớp (Class 0: False/Negative, Class 1: True/Positive), Confusion Matrix có dạng 2×2:

```
                    Predicted (Dự đoán)
                  ┌──────────┬──────────┐
                  │  Class 0 │  Class 1 │
        ┌─────────┼──────────┼──────────┤
Actual  │ Class 0 │    TN    │    FP    │
(Thực)  ├─────────┼──────────┼──────────┤
        │ Class 1 │    FN    │    TP    │
        └─────────┴──────────┴──────────┘
```

### 4 ô quan trọng:

1. **TN (True Negative)**:

   - Thực tế là Class 0, dự đoán là Class 0 → **Đúng**
   - Mô hình dự đoán đúng khi không có sự kiện xảy ra

2. **FP (False Positive)**:

   - Thực tế là Class 0, dự đoán là Class 1 → **Sai**
   - Còn gọi là **Type I Error** hoặc **False Alarm**
   - Mô hình dự đoán sai khi báo có sự kiện nhưng thực tế không có

3. **FN (False Negative)**:

   - Thực tế là Class 1, dự đoán là Class 0 → **Sai**
   - Còn gọi là **Type II Error** hoặc **Miss**
   - Mô hình dự đoán sai khi báo không có sự kiện nhưng thực tế có

4. **TP (True Positive)**:
   - Thực tế là Class 1, dự đoán là Class 1 → **Đúng**
   - Mô hình dự đoán đúng khi có sự kiện xảy ra

## Ví dụ từ dự án Online Shoppers Intention

### Test Set của dự án:

- **Class 0** (False/Negative - Không mua hàng): 2,084 mẫu
- **Class 1** (True/Positive - Mua hàng): 382 mẫu
- **Tổng**: 2,466 mẫu

### Decision Tree Confusion Matrix

Từ kết quả evaluation trong notebook `05_Model_Evaluation.ipynb`:

**Metrics đã có:**

- **Accuracy** = 0.8674
- **Recall** (Class 1) = 0.5262
- **Precision** (Class 1) = 0.5793
- **F1-Score** = 0.5514

**Confusion Matrix tính toán:**

```
                    Predicted
                  ┌──────────┬──────────┐
                  │  Class 0 │  Class 1 │
        ┌─────────┼──────────┼──────────┤
        │ Class 0 │   1,939  │    145   │
Actual  ├─────────┼──────────┼──────────┤
        │ Class 1 │    181   │    201   │
        └─────────┴──────────┴──────────┘
```

**Giải thích chi tiết:**

- **TN = 1,939**: 1,939 khách hàng thực tế không mua hàng, mô hình dự đoán đúng là không mua
- **FP = 145**: 145 khách hàng thực tế không mua hàng, mô hình dự đoán sai là sẽ mua (False Positive)
- **FN = 181**: 181 khách hàng thực tế mua hàng, mô hình dự đoán sai là không mua (False Negative)
- **TP = 201**: 201 khách hàng thực tế mua hàng, mô hình dự đoán đúng là sẽ mua

**Kiểm tra:**

- Tổng đúng = TN + TP = 1,939 + 201 = 2,140 ✓
- Tổng sai = FP + FN = 145 + 181 = 326 ✓
- Tổng Class 0 thực tế = TN + FP = 1,939 + 145 = 2,084 ✓
- Tổng Class 1 thực tế = FN + TP = 181 + 201 = 382 ✓

### Random Forest Confusion Matrix

**Metrics đã có:**

- **Accuracy** = 0.9023
- **Recall** (Class 1) = 0.5340
- **Precision** (Class 1) = 0.7640
- **F1-Score** = 0.6287

**Confusion Matrix tính toán:**

```
                    Predicted
                  ┌──────────┬──────────┐
                  │  Class 0 │  Class 1 │
        ┌─────────┼──────────┼──────────┤
        │ Class 0 │   2,020  │     64   │
Actual  ├─────────┼──────────┼──────────┤
        │ Class 1 │    178   │    204   │
        └─────────┴──────────┴──────────┘
```

**So sánh với Decision Tree:**

- Random Forest có ít False Positive hơn (64 vs 145) → Precision cao hơn
- False Negative gần tương đương (178 vs 181) → Recall gần bằng nhau
- Random Forest tổng thể tốt hơn với ít lỗi hơn

## Từ Confusion Matrix tính các Metrics

### 1. Accuracy (Độ chính xác tổng thể)

```
Accuracy = (TN + TP) / (TN + FP + FN + TP)
         = Tổng số dự đoán đúng / Tổng số mẫu
```

**Ví dụ Decision Tree:**

```
Accuracy = (1,939 + 201) / 2,466
         = 2,140 / 2,466
         = 0.8674 = 86.74%
```

**Ý nghĩa**: 86.74% tổng số mẫu được dự đoán đúng.

**Hạn chế**: Accuracy không phản ánh tốt khi dữ liệu mất cân bằng (imbalanced data). Với dữ liệu mất cân bằng như dự án này (2,084 vs 382), một mô hình "ngây thơ" chỉ dự đoán Class 0 sẽ có Accuracy = 84.5% nhưng không có giá trị.

### 2. Precision (Độ chính xác - Class 1)

```
Precision = TP / (TP + FP)
          = Số dự đoán đúng Class 1 / Tổng số dự đoán Class 1
```

**Ví dụ Decision Tree:**

```
Precision = 201 / (201 + 145)
          = 201 / 346
          = 0.5793 = 57.93%
```

**Ý nghĩa**: Trong số các mẫu mô hình dự đoán là Class 1, 57.93% là đúng.

**Trong bài toán Online Shoppers Intention:**

- 57.93% khách hàng được dự đoán sẽ mua hàng thực sự mua hàng.
- Precision cao → Giảm chi phí marketing lãng phí (ít False Positive)

### 3. Recall (Độ nhạy - Sensitivity - Class 1)

```
Recall = TP / (TP + FN)
       = Số phát hiện đúng Class 1 / Tổng số Class 1 thực tế
```

**Ví dụ Decision Tree:**

```
Recall = 201 / (201 + 181)
       = 201 / 382
       = 0.5262 = 52.62%
```

**Ý nghĩa**: Trong số các mẫu thực tế là Class 1, mô hình phát hiện được 52.62%.

**Trong bài toán Online Shoppers Intention:**

- 52.62% khách hàng thực sự mua hàng được mô hình phát hiện.
- Recall cao → Bỏ lỡ ít khách hàng tiềm năng (ít False Negative)

### 4. Specificity (Độ đặc hiệu - Class 0)

```
Specificity = TN / (TN + FP)
            = Số dự đoán đúng Class 0 / Tổng số Class 0 thực tế
```

**Ví dụ Decision Tree:**

```
Specificity = 1,939 / (1,939 + 145)
            = 1,939 / 2,084
            = 0.9304 = 93.04%
```

**Ý nghĩa**: 93.04% mẫu thực tế là Class 0 được mô hình dự đoán đúng.

**Trong bài toán Online Shoppers Intention:**

- 93.04% khách hàng không mua hàng được dự đoán đúng là không mua.

### 5. F1-Score (Cân bằng Precision và Recall)

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = Harmonic Mean của Precision và Recall
```

**Ví dụ Decision Tree:**

```
F1 = 2 × (0.5793 × 0.5262) / (0.5793 + 0.5262)
   = 2 × 0.3048 / 1.1055
   = 0.5514 = 55.14%
```

**Ý nghĩa**: Cân bằng giữa Precision và Recall. F1-Score cao khi cả Precision và Recall đều cao.

**Khi nào dùng F1:**

- Khi cần cân bằng giữa Precision và Recall
- Khi dữ liệu mất cân bằng và Accuracy không phản ánh tốt hiệu suất

### Tóm tắt công thức từ Confusion Matrix

```
                    Predicted
                  ┌──────────┬──────────┐
                  │  Class 0 │  Class 1 │
        ┌─────────┼──────────┼──────────┤
        │ Class 0 │    TN    │    FP    │ → Specificity = TN/(TN+FP)
Actual  ├─────────┼──────────┼──────────┤
        │ Class 1 │    FN    │    TP    │ → Recall = TP/(TP+FN)
        └─────────┴──────────┴──────────┘
                   ↓           ↓
              Precision = TP/(TP+FP)
              Accuracy = (TN+TP)/(TN+FP+FN+TP)
```

## So sánh Decision Tree vs Random Forest

### Decision Tree:

```
Accuracy:  86.74%
Precision: 57.93%
Recall:    52.62%
F1:        55.14%
```

### Random Forest:

```
Accuracy:  90.23%
Precision: 76.40%
Recall:    53.40%
F1:        62.87%
```

**Nhận xét:**

- **Random Forest có Accuracy cao hơn**: 90.23% vs 86.74% (+3.49%)
- **Random Forest có Precision cao hơn nhiều**: 76.40% vs 57.93% (+18.47%)
  - → Ít False Positive hơn (64 vs 145)
  - → Giảm chi phí marketing lãng phí đáng kể
- **Recall gần tương đương**: 53.40% vs 52.62% (+0.78%)
  - → Phát hiện được gần như số lượng khách hàng mua hàng tương đương
- **Random Forest có F1 cao hơn**: 62.87% vs 55.14% (+7.73%)
  - → Cân bằng tốt hơn giữa Precision và Recall

**Kết luận**: Random Forest tốt hơn Decision Tree trong bài toán này, đặc biệt ở khả năng giảm False Positive (tăng Precision).

## Ý nghĩa thực tế trong bài toán Online Shoppers Intention

### False Positive (FP)

**Với Decision Tree: FP = 145**

- 145 khách hàng không mua hàng nhưng được dự đoán sẽ mua
- **Hậu quả**:
  - Tốn chi phí marketing, quảng cáo không cần thiết
  - Giảm ROI của chiến dịch marketing
  - Lãng phí nguồn lực

**Với Random Forest: FP = 64**

- Chỉ 64 khách hàng không mua hàng được dự đoán sai
- **Lợi ích**: Tiết kiệm 81 khách hàng × chi phí marketing = Tiết kiệm đáng kể

### False Negative (FN)

**Với Decision Tree: FN = 181**

- 181 khách hàng sẽ mua hàng nhưng không được phát hiện
- **Hậu quả**:
  - Bỏ lỡ cơ hội bán hàng
  - Mất doanh thu tiềm năng
  - Không tận dụng được cơ hội chuyển đổi

**Với Random Forest: FN = 178**

- Chỉ giảm được 3 False Negative so với Decision Tree
- Vẫn còn nhiều khách hàng tiềm năng bị bỏ lỡ

### Trade-off Precision vs Recall

**Precision cao (Random Forest: 76.40%):**

- ✅ Ít tốn chi phí marketing sai
- ✅ Hiệu quả marketing cao hơn
- ❌ Vẫn có thể bỏ lỡ một số khách hàng tiềm năng

**Recall cao (hơn 70%):**

- ✅ Phát hiện được nhiều khách hàng tiềm năng
- ✅ Không bỏ lỡ cơ hội bán hàng
- ❌ Tốn nhiều chi phí marketing cho những khách hàng không mua

**Trong bài toán này:**

- Random Forest đạt được Precision cao (76.40%) với Recall chấp nhận được (53.40%)
- Đây là lựa chọn tốt vì giảm chi phí marketing lãng phí là ưu tiên
- Vẫn phát hiện được hơn 50% khách hàng thực sự mua hàng

## Cách Visualize Confusion Matrix trong code

### Sử dụng function có sẵn:

```python
from evaluation.visualization import plot_confusion_matrix

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred_dt,
                     labels=['Không mua', 'Mua'],
                     title='Decision Tree - Confusion Matrix')
```

### Hoặc sử dụng trực tiếp với sklearn và seaborn:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Tính confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Vẽ heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Không mua', 'Mua'],
            yticklabels=['Không mua', 'Mua'])
plt.ylabel('Nhãn thực tế')
plt.xlabel('Nhãn dự đoán')
plt.title('Confusion Matrix')
plt.show()
```

### Output:

- **Heatmap màu xanh** (Blues) để dễ nhìn
- **Số trong mỗi ô** hiển thị số lượng mẫu
- **Trục Y**: True Label (nhãn thực tế)
- **Trục X**: Predicted Label (nhãn dự đoán)

## Khi nào cần quan tâm đến từng metric?

### Ưu tiên Precision khi:

- ✅ Chi phí của False Positive cao (ví dụ: spam detection, fraud detection)
- ✅ Muốn đảm bảo những gì mô hình báo là tích cực thực sự là tích cực
- ✅ Tránh lãng phí nguồn lực cho dự đoán sai

**Ví dụ trong bài toán này:**

- Chi phí marketing cho khách hàng không mua hàng là lãng phí
- → Cần Precision cao (Random Forest: 76.40%)

### Ưu tiên Recall khi:

- ✅ Hậu quả của False Negative nghiêm trọng (ví dụ: chẩn đoán bệnh, phát hiện an ninh)
- ✅ Không muốn bỏ lỡ bất kỳ trường hợp tích cực nào
- ✅ Chấp nhận một số False Positive để không bỏ sót

**Ví dụ:**

- Chẩn đoán ung thư: Bỏ sót bệnh nhân ung thư là nghiêm trọng hơn nhiều so với dự đoán sai
- → Cần Recall cao (>90%)

### Cần cả Precision và Recall:

- ✅ Dùng F1-Score để cân bằng
- ✅ Sử dụng Precision-Recall Curve để xem trade-off
- ✅ Điều chỉnh threshold của mô hình để cân bằng

## Tóm tắt

1. **Confusion Matrix** là công cụ cơ bản nhất để đánh giá mô hình phân loại, hiển thị số lượng dự đoán đúng/sai.

2. **4 ô chính**:

   - **TN (True Negative)**: Dự đoán đúng Class 0
   - **FP (False Positive)**: Dự đoán sai - Type I Error
   - **FN (False Negative)**: Dự đoán sai - Type II Error
   - **TP (True Positive)**: Dự đoán đúng Class 1

3. **Các metrics từ Confusion Matrix**:

   - **Accuracy**: Tỷ lệ dự đoán đúng tổng thể
   - **Precision**: Độ chính xác khi dự đoán Class 1
   - **Recall**: Độ nhạy - Phát hiện được bao nhiêu Class 1
   - **Specificity**: Độ đặc hiệu - Phát hiện được bao nhiêu Class 0
   - **F1-Score**: Cân bằng giữa Precision và Recall

4. **Precision vs Recall Trade-off**:

   - Tăng Precision → Giảm FP → Ít tốn chi phí sai
   - Tăng Recall → Giảm FN → Bỏ lỡ ít trường hợp tích cực
   - Tùy vào bài toán để chọn ưu tiên

5. **Trong dự án này**:

   - Random Forest tốt hơn Decision Tree
   - Đặc biệt tốt ở Precision (76.40%) → Giảm chi phí marketing lãng phí
   - Recall chấp nhận được (53.40%) → Vẫn phát hiện được hơn một nửa khách hàng mua hàng

6. **Confusion Matrix giúp**:
   - Hiểu rõ mô hình sai ở đâu
   - Đánh giá hiệu suất theo từng lớp
   - Quyết định điều chỉnh mô hình hoặc threshold
   - So sánh các mô hình một cách trực quan

---

**Tài liệu tham khảo:**

- Notebook: `05_Model_Evaluation.ipynb`
- Code: `src/evaluation/metrics.py`, `src/evaluation/visualization.py`
- Dữ liệu: Test set với 2,466 mẫu (2,084 Class 0, 382 Class 1)
