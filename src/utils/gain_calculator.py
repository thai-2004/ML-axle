"""
Utility functions for calculating Information Gain and Gini Gain
when splitting data into multiple groups.
"""

import numpy as np


def calculate_entropy(y):
    """
    Tính Entropy cho một nhóm dữ liệu.
    
    Parameters:
    -----------
    y : array-like
        Labels của nhóm dữ liệu (0 hoặc 1 cho binary classification)
    
    Returns:
    --------
    float
        Entropy value (bits)
    """
    if len(y) == 0:
        return 0
    
    # Tính tỷ lệ của mỗi class
    unique, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    
    # Loại bỏ các giá trị 0 để tránh log(0)
    p = p[p > 0]
    
    # Tính entropy
    entropy = -np.sum(p * np.log2(p))
    return entropy


def calculate_gini(y):
    """
    Tính Gini Impurity cho một nhóm dữ liệu.
    
    Parameters:
    -----------
    y : array-like
        Labels của nhóm dữ liệu (0 hoặc 1 cho binary classification)
    
    Returns:
    --------
    float
        Gini Impurity value (0-1)
    """
    if len(y) == 0:
        return 0
    
    # Tính tỷ lệ của mỗi class
    unique, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    
    # Tính Gini impurity
    gini = 1 - np.sum(p ** 2)
    return gini


def calculate_information_gain(parent_y, groups_y, verbose=False):
    """
    Tính Information Gain khi split node cha thành nhiều nhóm con.
    
    Parameters:
    -----------
    parent_y : array-like
        Labels của node cha
    groups_y : list of arrays
        List các arrays chứa labels của từng nhóm con
    verbose : bool
        In chi tiết quá trình tính toán
    
    Returns:
    --------
    float
        Information Gain value (bits)
    dict
        Chi tiết quá trình tính toán (nếu verbose=True)
    """
    # Entropy của node cha
    parent_entropy = calculate_entropy(parent_y)
    total_samples = len(parent_y)
    
    # Tính weighted entropy của các nhóm con
    weighted_entropy = 0
    details = {
        'parent_entropy': parent_entropy,
        'parent_samples': total_samples,
        'groups': []
    }
    
    for i, group_y in enumerate(groups_y):
        if len(group_y) > 0:
            group_weight = len(group_y) / total_samples
            group_entropy = calculate_entropy(group_y)
            weighted_entropy += group_weight * group_entropy
            
            # Tính phân phối class
            unique, counts = np.unique(group_y, return_counts=True)
            class_dist = dict(zip(unique, counts))
            
            details['groups'].append({
                'group_id': i + 1,
                'samples': len(group_y),
                'weight': group_weight,
                'entropy': group_entropy,
                'class_distribution': class_dist,
                'contribution': group_weight * group_entropy
            })
    
    # Information Gain
    information_gain = parent_entropy - weighted_entropy
    
    details['weighted_entropy'] = weighted_entropy
    details['information_gain'] = information_gain
    
    if verbose:
        print("=" * 60)
        print("INFORMATION GAIN CALCULATION")
        print("=" * 60)
        print(f"\nParent Node:")
        print(f"  Samples: {total_samples}")
        print(f"  Entropy: {parent_entropy:.4f} bits")
        
        unique, counts = np.unique(parent_y, return_counts=True)
        print(f"  Class Distribution: {dict(zip(unique, counts))}")
        
        print(f"\nChild Groups:")
        for group in details['groups']:
            print(f"\n  Group {group['group_id']}:")
            print(f"    Samples: {group['samples']} ({group['weight']*100:.1f}%)")
            print(f"    Class Distribution: {group['class_distribution']}")
            print(f"    Entropy: {group['entropy']:.4f} bits")
            print(f"    Contribution: {group['contribution']:.4f} bits")
        
        print(f"\nWeighted Entropy: {weighted_entropy:.4f} bits")
        print(f"Information Gain: {information_gain:.4f} bits")
        print("=" * 60)
    
    return information_gain, details


def calculate_gini_gain(parent_y, groups_y, verbose=False):
    """
    Tính Gini Gain khi split node cha thành nhiều nhóm con.
    
    Parameters:
    -----------
    parent_y : array-like
        Labels của node cha
    groups_y : list of arrays
        List các arrays chứa labels của từng nhóm con
    verbose : bool
        In chi tiết quá trình tính toán
    
    Returns:
    --------
    float
        Gini Gain value
    dict
        Chi tiết quá trình tính toán (nếu verbose=True)
    """
    # Gini của node cha
    parent_gini = calculate_gini(parent_y)
    total_samples = len(parent_y)
    
    # Tính weighted gini của các nhóm con
    weighted_gini = 0
    details = {
        'parent_gini': parent_gini,
        'parent_samples': total_samples,
        'groups': []
    }
    
    for i, group_y in enumerate(groups_y):
        if len(group_y) > 0:
            group_weight = len(group_y) / total_samples
            group_gini = calculate_gini(group_y)
            weighted_gini += group_weight * group_gini
            
            # Tính phân phối class
            unique, counts = np.unique(group_y, return_counts=True)
            class_dist = dict(zip(unique, counts))
            
            details['groups'].append({
                'group_id': i + 1,
                'samples': len(group_y),
                'weight': group_weight,
                'gini': group_gini,
                'class_distribution': class_dist,
                'contribution': group_weight * group_gini
            })
    
    # Gini Gain
    gini_gain = parent_gini - weighted_gini
    
    details['weighted_gini'] = weighted_gini
    details['gini_gain'] = gini_gain
    
    if verbose:
        print("=" * 60)
        print("GINI GAIN CALCULATION")
        print("=" * 60)
        print(f"\nParent Node:")
        print(f"  Samples: {total_samples}")
        print(f"  Gini Impurity: {parent_gini:.4f}")
        
        unique, counts = np.unique(parent_y, return_counts=True)
        print(f"  Class Distribution: {dict(zip(unique, counts))}")
        
        print(f"\nChild Groups:")
        for group in details['groups']:
            print(f"\n  Group {group['group_id']}:")
            print(f"    Samples: {group['samples']} ({group['weight']*100:.1f}%)")
            print(f"    Class Distribution: {group['class_distribution']}")
            print(f"    Gini Impurity: {group['gini']:.4f}")
            print(f"    Contribution: {group['contribution']:.4f}")
        
        print(f"\nWeighted Gini: {weighted_gini:.4f}")
        print(f"Gini Gain: {gini_gain:.4f}")
        print("=" * 60)
    
    return gini_gain, details


def example_3_groups():
    """
    Ví dụ tính Gain cho trường hợp split thành 3 nhóm.
    """
    print("\n" + "=" * 70)
    print("VÍ DỤ: TÍNH GAIN CHO 3 NHÓM DỮ LIỆU")
    print("=" * 70)
    
    # Giả sử: 0 = False (không mua), 1 = True (mua)
    
    # Node cha: 1000 mẫu (850 không mua, 150 mua)
    parent_y = np.array([0] * 850 + [1] * 150)
    
    # Nhóm 1: Returning_Visitor (700 mẫu: 630 không mua, 70 mua)
    group1_y = np.array([0] * 630 + [1] * 70)
    
    # Nhóm 2: New_Visitor (250 mẫu: 180 không mua, 70 mua)
    group2_y = np.array([0] * 180 + [1] * 70)
    
    # Nhóm 3: Other (50 mẫu: 40 không mua, 10 mua)
    group3_y = np.array([0] * 40 + [1] * 10)
    
    groups_y = [group1_y, group2_y, group3_y]
    
    # Tính Information Gain
    print("\n")
    info_gain, info_details = calculate_information_gain(
        parent_y, groups_y, verbose=True
    )
    
    # Tính Gini Gain
    print("\n")
    gini_gain, gini_details = calculate_gini_gain(
        parent_y, groups_y, verbose=True
    )
    
    # So sánh
    print("\n" + "=" * 70)
    print("SO SÁNH")
    print("=" * 70)
    print(f"Information Gain: {info_gain:.4f} bits")
    print(f"Gini Gain:       {gini_gain:.4f}")
    print("\nLưu ý: Random Forest (sklearn) mặc định dùng Gini Gain")
    print("=" * 70)


if __name__ == "__main__":
    example_3_groups()

