"""
Demo script minh họa quy trình tính Gini Gain cho TẤT CẢ features
để chọn best split trong Decision Tree.

Script này mô hình hóa quy trình được mô tả trong:
docs/gini_gain_calculation_process.md
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from gain_calculator import calculate_gini, calculate_gini_gain


def find_best_split(X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> Tuple[str, Optional[float], float, dict]:
    """
    Tìm best split cho một node bằng cách tính Gini Gain cho TẤT CẢ features.
    
    Đây là mô hình hóa quy trình thực tế của Decision Tree:
    1. Tính Gini Gain cho mỗi feature
    2. Với numerical features: thử nhiều threshold
    3. Với categorical features: thử các cách split khác nhau
    4. Chọn feature + threshold có Gini Gain cao nhất
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features (các cột là features)
    y : pd.Series
        Labels (target variable)
    verbose : bool
        In chi tiết quá trình tính toán
    
    Returns:
    --------
    best_feature : str
        Tên feature tốt nhất
    best_threshold : float hoặc None
        Threshold tốt nhất (None nếu là categorical)
    best_gain : float
        Gini Gain cao nhất
    results : dict
        Chi tiết kết quả cho tất cả features
    """
    if verbose:
        print("=" * 80)
        print("QUY TRÌNH TÌM BEST SPLIT: TÍNH GINI GAIN CHO TẤT CẢ FEATURES")
        print("=" * 80)
        print(f"\nDataset: {len(X)} mẫu, {len(X.columns)} features")
        print(f"Features: {list(X.columns)}")
        
        # Hiển thị class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"Class Distribution: {class_dist}")
    
    # Bước 1: Tính Gini của node cha
    parent_labels = y.values
    gini_parent = calculate_gini(parent_labels)
    
    if verbose:
        print(f"\n{'='*80}")
        print("BƯỚC 1: TÍNH GINI CỦA NODE CHA")
        print(f"{'='*80}")
        print(f"Gini(parent) = {gini_parent:.6f}")
    
    # Bước 2: Khởi tạo
    best_gain = -1
    best_feature = None
    best_threshold = None
    best_split_type = None
    
    # Lưu kết quả cho tất cả features
    results = {
        'parent_gini': gini_parent,
        'features': {}
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print("BƯỚC 2: TÍNH GINI GAIN CHO TỪNG FEATURE")
        print(f"{'='*80}\n")
    
    # Bước 3: Duyệt qua tất cả features
    for feature_name in X.columns:
        feature_values = X[feature_name].values
        feature_type = None
        feature_results = []
        
        # Kiểm tra loại feature
        if pd.api.types.is_numeric_dtype(X[feature_name]):
            # NUMERICAL FEATURE
            feature_type = 'numerical'
            
            # Sắp xếp và tạo thresholds
            unique_values = np.unique(feature_values)
            
            if len(unique_values) == 1:
                # Tất cả giá trị giống nhau → không thể split
                if verbose:
                    print(f"[{feature_name}] (Numerical)")
                    print(f"  ⚠️  Tất cả giá trị = {unique_values[0]} → Không thể split (Gain = 0)")
                continue
            
            # Tạo thresholds: trung bình giữa các giá trị liên tiếp
            sorted_values = np.sort(unique_values)
            thresholds = [(sorted_values[i] + sorted_values[i+1]) / 2 
                         for i in range(len(sorted_values)-1)]
            
            if verbose:
                print(f"[{feature_name}] (Numerical)")
                print(f"  Giá trị: {sorted_values}")
                print(f"  Thresholds: {len(thresholds)} thresholds")
            
            # Thử từng threshold
            best_threshold_for_feature = None
            best_gain_for_feature = -1
            
            for threshold in thresholds:
                # Split
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Kiểm tra split hợp lệ (cả 2 nhóm phải có ít nhất 1 mẫu)
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                
                left_labels = y[left_mask].values
                right_labels = y[right_mask].values
                
                # Tính Gini Gain
                children = [left_labels, right_labels]
                gini_gain, details = calculate_gini_gain(parent_labels, children, verbose=False)
                
                # Lưu kết quả
                feature_results.append({
                    'threshold': threshold,
                    'gini_gain': gini_gain,
                    'left_samples': len(left_labels),
                    'right_samples': len(right_labels),
                    'left_gini': calculate_gini(left_labels),
                    'right_gini': calculate_gini(right_labels),
                    'details': details
                })
                
                # Cập nhật best cho feature này
                if gini_gain > best_gain_for_feature:
                    best_gain_for_feature = gini_gain
                    best_threshold_for_feature = threshold
                
                # Cập nhật best tổng thể
                if gini_gain > best_gain:
                    best_gain = gini_gain
                    best_feature = feature_name
                    best_threshold = threshold
                    best_split_type = 'numerical'
            
            # Lưu kết quả feature
            results['features'][feature_name] = {
                'type': feature_type,
                'best_threshold': best_threshold_for_feature,
                'best_gain': best_gain_for_feature,
                'all_results': feature_results
            }
            
            if verbose:
                if best_gain_for_feature > 0:
                    best_result = max(feature_results, key=lambda x: x['gini_gain'])
                    print(f"  ✅ Best Threshold: {best_result['threshold']:.4f}")
                    print(f"     Left:  {best_result['left_samples']} mẫu, Gini = {best_result['left_gini']:.4f}")
                    print(f"     Right: {best_result['right_samples']} mẫu, Gini = {best_result['right_gini']:.4f}")
                    print(f"     Gini Gain = {best_result['gini_gain']:.6f}")
                else:
                    print(f"  ❌ Không tìm được split hợp lệ")
                print()
        
        else:
            # CATEGORICAL FEATURE
            feature_type = 'categorical'
            unique_values = np.unique(feature_values)
            
            if len(unique_values) == 1:
                # Tất cả giá trị giống nhau → không thể split
                if verbose:
                    print(f"[{feature_name}] (Categorical)")
                    print(f"  ⚠️  Tất cả giá trị = {unique_values[0]} → Không thể split (Gain = 0)")
                continue
            
            if verbose:
                print(f"[{feature_name}] (Categorical)")
                print(f"  Giá trị unique: {unique_values}")
                print(f"  Số cách split: {len(unique_values)}")
            
            # Thử split theo từng giá trị (binary split)
            best_value_for_feature = None
            best_gain_for_feature = -1
            
            for value in unique_values:
                # Split: feature == value vs feature != value
                left_mask = feature_values == value
                right_mask = ~left_mask
                
                # Kiểm tra split hợp lệ
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                
                left_labels = y[left_mask].values
                right_labels = y[right_mask].values
                
                # Tính Gini Gain
                children = [left_labels, right_labels]
                gini_gain, details = calculate_gini_gain(parent_labels, children, verbose=False)
                
                # Lưu kết quả
                feature_results.append({
                    'value': value,
                    'gini_gain': gini_gain,
                    'left_samples': len(left_labels),
                    'right_samples': len(right_labels),
                    'left_gini': calculate_gini(left_labels),
                    'right_gini': calculate_gini(right_labels),
                    'details': details
                })
                
                # Cập nhật best cho feature này
                if gini_gain > best_gain_for_feature:
                    best_gain_for_feature = gini_gain
                    best_value_for_feature = value
                
                # Cập nhật best tổng thể
                if gini_gain > best_gain:
                    best_gain = gini_gain
                    best_feature = feature_name
                    best_threshold = value
                    best_split_type = 'categorical'
            
            # Lưu kết quả feature
            results['features'][feature_name] = {
                'type': feature_type,
                'best_value': best_value_for_feature,
                'best_gain': best_gain_for_feature,
                'all_results': feature_results
            }
            
            if verbose:
                if best_gain_for_feature > 0:
                    best_result = max(feature_results, key=lambda x: x['gini_gain'])
                    print(f"  ✅ Best Split: {feature_name} == {best_result['value']}")
                    print(f"     Left:  {best_result['left_samples']} mẫu, Gini = {best_result['left_gini']:.4f}")
                    print(f"     Right: {best_result['right_samples']} mẫu, Gini = {best_result['right_gini']:.4f}")
                    print(f"     Gini Gain = {best_result['gini_gain']:.6f}")
                else:
                    print(f"  ❌ Không tìm được split hợp lệ")
                print()
    
    # Bước 4: Tóm tắt kết quả
    if verbose:
        print(f"\n{'='*80}")
        print("BƯỚC 3: SO SÁNH VÀ CHỌN BEST SPLIT")
        print(f"{'='*80}\n")
        
        # Sắp xếp features theo Gini Gain
        sorted_features = sorted(
            [(name, info['best_gain']) for name, info in results['features'].items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        print("Bảng xếp hạng Gini Gain:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Feature':<25} {'Type':<15} {'Gini Gain':<15}")
        print("-" * 80)
        
        for rank, (feature_name, gain) in enumerate(sorted_features, 1):
            feature_type = results['features'][feature_name]['type']
            symbol = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{symbol} {rank:<4} {feature_name:<25} {feature_type:<15} {gain:<15.6f}")
        
        print("-" * 80)
        
        print(f"\n{'='*80}")
        print("KẾT QUẢ: BEST SPLIT")
        print(f"{'='*80}")
        print(f"✅ Best Feature: {best_feature}")
        print(f"✅ Best Threshold/Value: {best_threshold}")
        print(f"✅ Split Type: {best_split_type}")
        print(f"✅ Best Gini Gain: {best_gain:.6f}")
        print(f"{'='*80}\n")
    
    results['best_feature'] = best_feature
    results['best_threshold'] = best_threshold
    results['best_gain'] = best_gain
    results['best_split_type'] = best_split_type
    
    return best_feature, best_threshold, best_gain, results


def demo_with_sample_data():
    """
    Demo với dataset mẫu nhỏ (5 mẫu) để minh họa quy trình.
    Dataset này giống ví dụ trong docs/gini_gain_calculation_process.md
    """
    print("\n" + "=" * 80)
    print("DEMO: MÔ HÌNH HÓA QUY TRÌNH TÍNH GINI GAIN")
    print("=" * 80)
    print("\nDataset mẫu (5 mẫu):")
    print("Dùng để minh họa quy trình, giống ví dụ trong:")
    print("docs/gini_gain_calculation_process.md\n")
    
    # Tạo dataset mẫu
    data = {
        'PageValues': [0.0, 0.0, 0.0, 0.0, 10.0],
        'Weekend': [0, 1, 1, 1, 1],
        'BounceRates': [0.2, 0.0, 0.2, 0.05, 0.02],
        'Revenue': [1, 0, 0, 0, 0]  # Target
    }
    
    df = pd.DataFrame(data)
    X = df[['PageValues', 'Weekend', 'BounceRates']]
    y = df['Revenue']
    
    print("Dataset:")
    print(df.to_string(index=True))
    print()
    
    # Tìm best split
    best_feature, best_threshold, best_gain, results = find_best_split(
        X, y, verbose=True
    )
    
    # Hiển thị chi tiết best split
    print("\n" + "=" * 80)
    print("CHI TIẾT BEST SPLIT")
    print("=" * 80)
    
    if best_feature:
        best_feature_info = results['features'][best_feature]
        best_result = max(
            best_feature_info['all_results'],
            key=lambda x: x['gini_gain']
        )
        
        print(f"\nFeature: {best_feature} ({best_feature_info['type']})")
        
        if best_feature_info['type'] == 'numerical':
            print(f"Threshold: {best_threshold}")
            print(f"\nSplit:")
            print(f"  Left (<= {best_threshold}):")
            print(f"    Samples: {best_result['left_samples']}")
            print(f"    Gini: {best_result['left_gini']:.6f}")
            
            # Hiển thị labels của left
            left_mask = X[best_feature] <= best_threshold
            left_labels = y[left_mask].tolist()
            print(f"    Labels: {left_labels}")
            
            print(f"\n  Right (> {best_threshold}):")
            print(f"    Samples: {best_result['right_samples']}")
            print(f"    Gini: {best_result['right_gini']:.6f}")
            
            # Hiển thị labels của right
            right_mask = X[best_feature] > best_threshold
            right_labels = y[right_mask].tolist()
            print(f"    Labels: {right_labels}")
        
        else:
            print(f"Split Value: {best_threshold}")
            print(f"\nSplit:")
            print(f"  Left ({best_feature} == {best_threshold}):")
            print(f"    Samples: {best_result['left_samples']}")
            print(f"    Gini: {best_result['left_gini']:.6f}")
            
            # Hiển thị labels của left
            left_mask = X[best_feature] == best_threshold
            left_labels = y[left_mask].tolist()
            print(f"    Labels: {left_labels}")
            
            print(f"\n  Right ({best_feature} != {best_threshold}):")
            print(f"    Samples: {best_result['right_samples']}")
            print(f"    Gini: {best_result['right_gini']:.6f}")
            
            # Hiển thị labels của right
            right_mask = X[best_feature] != best_threshold
            right_labels = y[right_mask].tolist()
            print(f"    Labels: {right_labels}")
        
        print(f"\nGini(parent): {results['parent_gini']:.6f}")
        print(f"Weighted Gini(children): {best_result['details']['weighted_gini']:.6f}")
        print(f"Gini Gain: {best_gain:.6f}")
        
        if best_result['left_gini'] == 0 and best_result['right_gini'] == 0:
            print("\n🎉 Perfect Split! Tạo ra 2 pure nodes (Gini = 0)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    demo_with_sample_data()

