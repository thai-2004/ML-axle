"""
Script to create Jupyter notebook templates for the project.
"""

import json
from pathlib import Path

def create_notebook_cell(cell_type, source, metadata=None):
    """Create a notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source if isinstance(source, list) else [source]
    }
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell

def create_eda_notebook():
    """Create 01_EDA.ipynb"""
    cells = [
        create_notebook_cell("markdown", "# 01. Exploratory Data Analysis (EDA)\n\nNotebook này thực hiện phân tích khám phá dữ liệu để hiểu về:\n- Cấu trúc và kích thước dataset\n- Phân phối các biến\n- Mối quan hệ giữa features và target\n- Missing values và outliers\n- Insights ban đầu về dữ liệu"),
        create_notebook_cell("code", "# Import libraries\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Set style\nsns.set_style(\"whitegrid\")\nplt.rcParams['figure.figsize'] = (12, 6)\n\n# Import project modules\nimport sys\nfrom pathlib import Path\nproject_root = Path().resolve().parent.parent\nsys.path.append(str(project_root / 'src'))\n\nfrom data.load_data import load_raw_data, inspect_data"),
        create_notebook_cell("markdown", "## 1. Load Data"),
        create_notebook_cell("code", "# Load raw data\ndf = load_raw_data()\n\n# Basic inspection\ninspect_data(df)"),
        create_notebook_cell("markdown", "## 2. Data Overview"),
        create_notebook_cell("code", "# Check data types\nprint(\"Data Types:\")\nprint(df.dtypes)\nprint(\"\\n\" + \"=\"*50)\n\n# Check for missing values\nprint(\"\\nMissing Values:\")\nmissing = df.isnull().sum()\nmissing_pct = (missing / len(df)) * 100\nmissing_df = pd.DataFrame({\n    'Missing Count': missing,\n    'Missing Percentage': missing_pct\n})\nprint(missing_df[missing_df['Missing Count'] > 0])"),
        create_notebook_cell("markdown", "## 3. Target Variable Analysis"),
        create_notebook_cell("code", "# Analyze target variable (Revenue)\nprint(\"Target Variable Distribution:\")\nprint(df['Revenue'].value_counts())\nprint(f\"\\nPercentage Distribution:\")\nprint(df['Revenue'].value_counts(normalize=True) * 100)\n\n# Visualize target distribution\nfig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n# Count plot\nsns.countplot(data=df, x='Revenue', ax=axes[0])\naxes[0].set_title('Revenue Distribution (Count)')\naxes[0].set_xlabel('Revenue')\n\n# Percentage plot\ndf['Revenue'].value_counts(normalize=True).plot(kind='bar', ax=axes[1], color=['skyblue', 'salmon'])\naxes[1].set_title('Revenue Distribution (Percentage)')\naxes[1].set_xlabel('Revenue')\naxes[1].set_ylabel('Percentage')\naxes[1].set_xticklabels(['False', 'True'], rotation=0)\n\nplt.tight_layout()\nplt.show()\n\nprint(f\"\\nClass Imbalance Ratio: {len(df[df['Revenue']==False]) / len(df[df['Revenue']==True]):.2f}:1\")"),
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

def create_preprocessing_notebook():
    """Create 02_Data_Preprocessing.ipynb"""
    cells = [
        create_notebook_cell("markdown", "# 02. Data Preprocessing\n\nNotebook này thực hiện tiền xử lý dữ liệu:\n- Xử lý missing values\n- Xử lý outliers\n- Encoding categorical variables\n- Chia tập train/validation/test\n- Lưu dữ liệu đã xử lý"),
        create_notebook_cell("code", "# Import libraries\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Import project modules\nimport sys\nfrom pathlib import Path\nproject_root = Path().resolve().parent.parent\nsys.path.append(str(project_root / 'src'))\n\nfrom data.load_data import load_raw_data\nfrom data.preprocess import (\n    check_missing_values, \n    check_outliers, \n    encode_categorical,\n    split_data\n)"),
        create_notebook_cell("markdown", "## 1. Load Raw Data"),
        create_notebook_cell("code", "# Load data\ndf = load_raw_data()\nprint(f\"Initial shape: {df.shape}\")\nprint(f\"\\nColumns: {list(df.columns)}\")\ndf.head()"),
        create_notebook_cell("markdown", "## 2. Check Missing Values"),
        create_notebook_cell("code", "# Check for missing values\nmissing_df = check_missing_values(df)\nprint(\"Missing Values Summary:\")\nprint(missing_df[missing_df['Missing Count'] > 0])\n\n# If no missing values, dataset is clean\nif missing_df['Missing Count'].sum() == 0:\n    print(\"\\n✓ No missing values found!\")\nelse:\n    print(\"\\nNeed to handle missing values...\")"),
        create_notebook_cell("markdown", "## 3. Outlier Detection"),
        create_notebook_cell("code", "# Check outliers\nnumerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()\noutlier_summary = check_outliers(df, columns=numerical_cols)\n\nprint(\"Outlier Summary:\")\nprint(outlier_summary)\nprint(\"\\nNote: Outliers are kept for now as they may contain valuable information.\")"),
        create_notebook_cell("markdown", "## 4. Encode Categorical Variables"),
        create_notebook_cell("code", "# Identify categorical columns (excluding target)\ncategorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()\nif 'Revenue' in categorical_cols:\n    categorical_cols.remove('Revenue')\n\nprint(f\"Categorical columns to encode: {categorical_cols}\")\n\n# Encode categorical variables using LabelEncoder\ndf_encoded, label_encoders = encode_categorical(df, columns=categorical_cols)\n\nprint(\"\\nEncoded data shape:\", df_encoded.shape)\nprint(\"\\nEncoded data types:\")\nprint(df_encoded.dtypes)"),
        create_notebook_cell("markdown", "## 5. Encode Target Variable"),
        create_notebook_cell("code", "# Encode Revenue (target variable)\nif df_encoded['Revenue'].dtype == 'bool':\n    df_encoded['Revenue'] = df_encoded['Revenue'].astype(int)\nelif df_encoded['Revenue'].dtype == 'object':\n    df_encoded['Revenue'] = (df_encoded['Revenue'] == 'True').astype(int)\n\nprint(\"Target variable distribution:\")\nprint(df_encoded['Revenue'].value_counts())"),
        create_notebook_cell("markdown", "## 6. Train/Validation/Test Split"),
        create_notebook_cell("code", "# Split data\nX_train, X_val, X_test, y_train, y_val, y_test = split_data(\n    df_encoded,\n    target_col='Revenue',\n    test_size=0.2,\n    val_size=0.1,\n    random_state=42,\n    stratify=True\n)\n\nprint(\"Data Split Summary:\")\nprint(f\"Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(df_encoded)*100:.1f}%)\")\nprint(f\"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df_encoded)*100:.1f}%)\")\nprint(f\"Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(df_encoded)*100:.1f}%)\")"),
        create_notebook_cell("markdown", "## 7. Save Processed Data"),
        create_notebook_cell("code", "# Save processed datasets\nprocessed_dir = project_root / \"data\" / \"processed\"\nprocessed_dir.mkdir(parents=True, exist_ok=True)\n\n# Save as CSV\nX_train.to_csv(processed_dir / \"X_train.csv\", index=False)\nX_val.to_csv(processed_dir / \"X_val.csv\", index=False)\nX_test.to_csv(processed_dir / \"X_test.csv\", index=False)\n\ny_train.to_csv(processed_dir / \"y_train.csv\", index=False)\ny_val.to_csv(processed_dir / \"y_val.csv\", index=False)\ny_test.to_csv(processed_dir / \"y_test.csv\", index=False)\n\ndf_encoded.to_csv(processed_dir / \"df_encoded.csv\", index=False)\n\nprint(\"✓ Processed data saved successfully!\")"),
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

def create_feature_engineering_notebook():
    """Create 03_Feature_Engineering.ipynb"""
    cells = [
        create_notebook_cell("markdown", "# 03. Feature Engineering\n\nNotebook này tạo các đặc trưng mới để cải thiện hiệu suất mô hình:\n- Tổng hợp features (total duration, total pages)\n- Tỷ lệ features (duration ratios, page ratios)\n- Duration per page features\n- Seasonal features (quarter, Q4 indicator)\n- Interaction features"),
        create_notebook_cell("code", "# Import libraries\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Import project modules\nimport sys\nfrom pathlib import Path\nproject_root = Path().resolve().parent.parent\nsys.path.append(str(project_root / 'src'))\n\nfrom features.engineering import engineer_all_features"),
        create_notebook_cell("markdown", "## 1. Load Data"),
        create_notebook_cell("code", "# Load preprocessed data\nprocessed_dir = project_root / \"data\" / \"processed\"\ndf = pd.read_csv(processed_dir / \"df_encoded.csv\")\n\nprint(f\"Data shape: {df.shape}\")\ndf.head()"),
        create_notebook_cell("markdown", "## 2. Apply Feature Engineering"),
        create_notebook_cell("code", "# Apply all feature engineering\ndf_engineered = engineer_all_features(df)\n\nprint(f\"Original features: {len(df.columns)}\")\nprint(f\"Engineered features: {len(df_engineered.columns)}\")\nprint(f\"New features added: {len(df_engineered.columns) - len(df.columns)}\")\n\n# Show new features\nnew_features = [col for col in df_engineered.columns if col not in df.columns]\nprint(f\"\\nNew features: {new_features}\")"),
        create_notebook_cell("markdown", "## 3. Save Engineered Features"),
        create_notebook_cell("code", "# Save engineered dataset\nfeatures_dir = project_root / \"data\" / \"features\"\nfeatures_dir.mkdir(parents=True, exist_ok=True)\n\ndf_engineered.to_csv(features_dir / \"df_engineered.csv\", index=False)\n\nprint(\"Engineered features saved successfully!\")"),
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return notebook

def create_model_training_notebook():
    """Create 04_Model_Training.ipynb"""
    cells = [
        create_notebook_cell("markdown", "# 04. Model Training\n\nNotebook này huấn luyện các mô hình:\n- Decision Tree\n- Random Forest\n- Hyperparameter tuning\n- Model evaluation"),
        create_notebook_cell("code", "# Import libraries\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Import project modules\nimport sys\nfrom pathlib import Path\nproject_root = Path().resolve().parent.parent\nsys.path.append(str(project_root / 'src'))\n\nfrom models.decision_tree import create_decision_tree_model, tune_decision_tree, save_model\nfrom models.random_forest import create_random_forest_model, tune_random_forest, save_model as save_rf_model\nfrom models.train import train_model"),
        create_notebook_cell("markdown", "## 1. Load Data"),
        create_notebook_cell("code", "# Load processed data\nprocessed_dir = project_root / \"data\" / \"processed\"\n\nX_train = pd.read_csv(processed_dir / \"X_train.csv\")\nX_val = pd.read_csv(processed_dir / \"X_val.csv\")\nX_test = pd.read_csv(processed_dir / \"X_test.csv\")\n\ny_train = pd.read_csv(processed_dir / \"y_train.csv\").squeeze()\ny_val = pd.read_csv(processed_dir / \"y_val.csv\").squeeze()\ny_test = pd.read_csv(processed_dir / \"y_test.csv\").squeeze()\n\nprint(f\"Training set: {X_train.shape}\")\nprint(f\"Validation set: {X_val.shape}\")\nprint(f\"Test set: {X_test.shape}\")"),
        create_notebook_cell("markdown", "## 2. Train Decision Tree"),
        create_notebook_cell("code", "# Create and train Decision Tree\ndt_model = create_decision_tree_model(\n    max_depth=None,\n    min_samples_split=2,\n    min_samples_leaf=1,\n    class_weight='balanced'\n)\n\ndt_model = train_model(dt_model, X_train, y_train)\n\n# Save model\nsave_model(dt_model, 'decision_tree.pkl')\nprint(\"Decision Tree model trained and saved!\")"),
        create_notebook_cell("markdown", "## 3. Train Random Forest"),
        create_notebook_cell("code", "# Create and train Random Forest\nrf_model = create_random_forest_model(\n    n_estimators=100,\n    max_depth=None,\n    max_features='sqrt',\n    class_weight='balanced'\n)\n\nrf_model = train_model(rf_model, X_train, y_train)\n\n# Save model\nsave_rf_model(rf_model, 'random_forest.pkl')\nprint(\"Random Forest model trained and saved!\")"),
        create_notebook_cell("markdown", "## 4. Hyperparameter Tuning (Optional)"),
        create_notebook_cell("code", "# Uncomment to run hyperparameter tuning\n# print(\"Tuning Decision Tree...\")\n# dt_grid = tune_decision_tree(X_train, y_train, cv=5, scoring='f1')\n# print(f\"Best parameters: {dt_grid.best_params_}\")\n# print(f\"Best score: {dt_grid.best_score_}\")\n\n# print(\"\\nTuning Random Forest...\")\n# rf_grid = tune_random_forest(X_train, y_train, cv=5, scoring='f1')\n# print(f\"Best parameters: {rf_grid.best_params_}\")\n# print(f\"Best score: {rf_grid.best_score_}\")"),
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return notebook

def create_model_evaluation_notebook():
    """Create 05_Model_Evaluation.ipynb"""
    cells = [
        create_notebook_cell("markdown", "# 05. Model Evaluation\n\nNotebook này đánh giá hiệu suất mô hình:\n- Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)\n- Confusion Matrix\n- ROC Curve\n- Precision-Recall Curve\n- So sánh models"),
        create_notebook_cell("code", "# Import libraries\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Import project modules\nimport sys\nfrom pathlib import Path\nproject_root = Path().resolve().parent.parent\nsys.path.append(str(project_root / 'src'))\n\nfrom models.decision_tree import load_model as load_dt_model\nfrom models.random_forest import load_model as load_rf_model\nfrom models.train import predict, predict_proba\nfrom evaluation.metrics import evaluate_model, print_classification_report\nfrom evaluation.visualization import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve"),
        create_notebook_cell("markdown", "## 1. Load Models and Data"),
        create_notebook_cell("code", "# Load models\ndt_model = load_dt_model('decision_tree.pkl')\nrf_model = load_rf_model('random_forest.pkl')\n\n# Load test data\nprocessed_dir = project_root / \"data\" / \"processed\"\nX_test = pd.read_csv(processed_dir / \"X_test.csv\")\ny_test = pd.read_csv(processed_dir / \"y_test.csv\").squeeze()\n\nprint(\"Models and data loaded successfully!\")"),
        create_notebook_cell("markdown", "## 2. Evaluate Decision Tree"),
        create_notebook_cell("code", "# Evaluate Decision Tree\nprint(\"=== Decision Tree Evaluation ===\\n\")\n\ndt_metrics = evaluate_model(dt_model, X_test, y_test)\nprint(\"Metrics:\")\nfor metric, value in dt_metrics.items():\n    print(f\"{metric}: {value:.4f}\")\n\n# Predictions\ny_pred_dt = predict(dt_model, X_test)\ny_proba_dt = predict_proba(dt_model, X_test)[:, 1] if hasattr(dt_model, 'predict_proba') else None\n\n# Classification report\nprint(\"\\nClassification Report:\")\nprint_classification_report(y_test, y_pred_dt)"),
        create_notebook_cell("markdown", "## 3. Evaluate Random Forest"),
        create_notebook_cell("code", "# Evaluate Random Forest\nprint(\"=== Random Forest Evaluation ===\\n\")\n\nrf_metrics = evaluate_model(rf_model, X_test, y_test)\nprint(\"Metrics:\")\nfor metric, value in rf_metrics.items():\n    print(f\"{metric}: {value:.4f}\")\n\n# Predictions\ny_pred_rf = predict(rf_model, X_test)\ny_proba_rf = predict_proba(rf_model, X_test)[:, 1] if hasattr(rf_model, 'predict_proba') else None\n\n# Classification report\nprint(\"\\nClassification Report:\")\nprint_classification_report(y_test, y_pred_rf)"),
        create_notebook_cell("markdown", "## 4. Visualizations"),
        create_notebook_cell("code", "# Plot confusion matrices\nfig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n# Decision Tree CM\nfrom sklearn.metrics import confusion_matrix\ncm_dt = confusion_matrix(y_test, y_pred_dt)\nsns.heatmap(cm_dt, annot=True, fmt='d', ax=axes[0], cmap='Blues')\naxes[0].set_title('Decision Tree - Confusion Matrix')\n\n# Random Forest CM\ncm_rf = confusion_matrix(y_test, y_pred_rf)\nsns.heatmap(cm_rf, annot=True, fmt='d', ax=axes[1], cmap='Blues')\naxes[1].set_title('Random Forest - Confusion Matrix')\n\nplt.tight_layout()\nplt.show()"),
        create_notebook_cell("code", "# ROC Curves\nif y_proba_dt is not None and y_proba_rf is not None:\n    from sklearn.metrics import roc_curve, auc\n    \n    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)\n    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)\n    \n    plt.figure(figsize=(10, 6))\n    plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc(fpr_dt, tpr_dt):.3f})')\n    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc(fpr_rf, tpr_rf):.3f})')\n    plt.plot([0, 1], [0, 1], 'k--', label='Random')\n    plt.xlabel('False Positive Rate')\n    plt.ylabel('True Positive Rate')\n    plt.title('ROC Curves Comparison')\n    plt.legend()\n    plt.grid(alpha=0.3)\n    plt.show()"),
        create_notebook_cell("markdown", "## 5. Model Comparison"),
        create_notebook_cell("code", "# Compare models\ncomparison = pd.DataFrame({\n    'Decision Tree': dt_metrics,\n    'Random Forest': rf_metrics\n})\nprint(\"Model Comparison:\")\nprint(comparison)"),
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return notebook

def create_model_explanation_notebook():
    """Create 06_Model_Explanation.ipynb"""
    cells = [
        create_notebook_cell("markdown", "# 06. Model Explanation\n\nNotebook này giải thích mô hình sử dụng:\n- Feature Importance\n- SHAP values (if available)\n- Sample predictions"),
        create_notebook_cell("code", "# Import libraries\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Import project modules\nimport sys\nfrom pathlib import Path\nproject_root = Path().resolve().parent.parent\nsys.path.append(str(project_root / 'src'))\n\nfrom models.random_forest import load_model, get_feature_importance\nfrom evaluation.visualization import plot_feature_importance"),
        create_notebook_cell("markdown", "## 1. Load Model and Data"),
        create_notebook_cell("code", "# Load model\nrf_model = load_model('random_forest.pkl')\n\n# Load test data\nprocessed_dir = project_root / \"data\" / \"processed\"\nX_test = pd.read_csv(processed_dir / \"X_test.csv\")\ny_test = pd.read_csv(processed_dir / \"y_test.csv\").squeeze()\n\nprint(\"Model and data loaded!\")"),
        create_notebook_cell("markdown", "## 2. Feature Importance"),
        create_notebook_cell("code", "# Get feature importance\nfeature_names = X_test.columns.tolist()\nimportance_dict = get_feature_importance(rf_model, feature_names)\n\n# Sort by importance\nimportance_sorted = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))\n\nprint(\"Top 10 Most Important Features:\")\nfor i, (feat, imp) in enumerate(list(importance_sorted.items())[:10], 1):\n    print(f\"{i}. {feat}: {imp:.4f}\")\n\n# Visualize\nimportances = [importance_dict[feat] for feat in feature_names]\nplot_feature_importance(importances, feature_names, top_n=20, title='Random Forest - Feature Importance')"),
        create_notebook_cell("markdown", "## 3. SHAP Values (Optional)"),
        create_notebook_cell("code", "# Uncomment to use SHAP\n# import shap\n# \n# # Create SHAP explainer\n# explainer = shap.TreeExplainer(rf_model)\n# shap_values = explainer.shap_values(X_test[:100])  # Use sample for speed\n# \n# # Summary plot\n# shap.summary_plot(shap_values[1], X_test[:100], show=False)\n# plt.title('SHAP Summary Plot')\n# plt.show()"),
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return notebook

def main():
    """Create all notebook templates."""
    notebooks_dir = Path(__file__).parent / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)
    
    notebooks = {
        "01_EDA.ipynb": create_eda_notebook(),
        "02_Data_Preprocessing.ipynb": create_preprocessing_notebook(),
        "03_Feature_Engineering.ipynb": create_feature_engineering_notebook(),
        "04_Model_Training.ipynb": create_model_training_notebook(),
        "05_Model_Evaluation.ipynb": create_model_evaluation_notebook(),
        "06_Model_Explanation.ipynb": create_model_explanation_notebook(),
    }
    
    for filename, notebook in notebooks.items():
        filepath = notebooks_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        print(f"Created: {filepath}")
    
    print("\nAll notebook templates created successfully!")

if __name__ == "__main__":
    main()

