# Predicting Future Purchase Intent from Early eCommerce Session Behavior

## Project Title and Description

This project builds a leakage-safe machine learning pipeline to predict whether an eCommerce customer session will result in a future purchase after observing only the first two session events.

The project follows an end-to-end AI/ML workflow:

1. Dataset overview and data dictionary
2. Exploratory data analysis
3. Data preprocessing, feature engineering, feature selection, and PCA
4. Supervised model implementation and model comparison
5. Ethical AI, explainability, and bias auditing
6. Final technical and business presentations

The final model is a LightGBM binary classification model that produces a purchase-propensity score for each eligible customer session.

---

## Problem Statement

eCommerce websites generate millions of browsing, cart, and purchase events. However, not every session has the same likelihood of converting into a purchase. The business goal is to identify high-intent sessions early so marketing or customer-engagement actions can be prioritized more efficiently.

This project asks:

> Can early customer-session behavior predict whether a user will purchase later in the same session?

To avoid data leakage, the model only uses information from the first two events of each session. The target variable, `future_purchase`, indicates whether a purchase occurs after those first two events.

This is framed as a supervised binary classification problem:

| Component | Definition |
|---|---|
| Prediction target | `future_purchase` |
| Positive class | Session purchases after the first two events |
| Observation window | First two session events only |
| Target window | Events after the second session event |
| Primary metric | PR-AUC / Average Precision |
| Final selected model | LightGBM |

PR-AUC was prioritized because future-purchase sessions are a minority class and accuracy alone can be misleading.

---

## Dataset Source and Description

The dataset used in this project is the **eCommerce behavior data from multi category store** dataset published on Kaggle by Michael Kechinov.

Dataset page:

```text
https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
```

The dataset contains user-event behavior from a large multi-category online store. Each row represents an event related to a product and a user. The available monthly files cover the period from October 2019 to April 2020.

This project uses:

```text
2020-Apr.csv
```

### Original Event-Level Fields

The raw dataset contains event-level fields similar to:

| Column | Description |
|---|---|
| `event_time` | Timestamp of the user event |
| `event_type` | Type of event, such as view, cart, or purchase |
| `product_id` | Product identifier |
| `category_id` | Product category identifier |
| `category_code` | Product category path or code |
| `brand` | Product brand |
| `price` | Product price |
| `user_id` | User identifier |
| `user_session` | Session identifier |

### Working Dataset Summary

| Processing Stage | Count |
|---|---:|
| Raw rows scanned | 66,589,268 |
| Session-preserved sample | 982,585 event records |
| Unique sampled sessions | 173,519 sessions |
| Leakage-safe model-ready sessions | 92,799 sessions |
| Training sessions | 64,959 |
| Validation sessions | 13,920 |
| Test sessions | 13,920 |

The raw dataset file is large and is not included in this repository. Download it from Kaggle and place it in the expected local folder before running the notebooks.

---

## Project Structure

```text
project_folder/
├── data
│   ├── img
│   │   └── slide1.png
│   ├── processed
│   │   ├── archive_old_step3_step4
│   │   │   ├── baseline_logistic_regression_validation_predictions.csv
│   │   │   ├── event_cleaning_audit.csv
│   │   │   ├── event_sample_cleaned_for_eda_feature_engineering.csv
│   │   │   ├── event_sample_session_preserved_approx_1m.csv
│   │   │   ├── final_model_test_evaluation_summary.csv
│   │   │   ├── final_random_forest_raw_feature_importance.csv
│   │   │   ├── final_random_forest_test_predictions.csv
│   │   │   ├── final_random_forest_transformed_feature_importance.csv
│   │   │   ├── initial_validation_model_comparison.csv
│   │   │   ├── initial_vs_tuned_random_forest_ranking_comparison.csv
│   │   │   ├── initial_vs_tuned_random_forest_threshold_comparison.csv
│   │   │   ├── leakage_safe_feature_inventory.csv
│   │   │   ├── leakage_safe_modeling_dataset_ready_for_split.csv
│   │   │   ├── random_forest_raw_feature_importance.csv
│   │   │   ├── random_forest_time_series_tuning_results.csv
│   │   │   ├── random_forest_validation_predictions.csv
│   │   │   ├── step4_final_model_artifact_manifest.csv
│   │   │   ├── step4_final_model_comparison_report.csv
│   │   │   ├── step4_final_tuned_random_forest_raw_feature_importance.csv
│   │   │   ├── step4_final_tuned_random_forest_test_evaluation.csv
│   │   │   ├── step4_final_tuned_random_forest_test_predictions.csv
│   │   │   ├── step4_final_tuned_random_forest_transformed_feature_importance.csv
│   │   │   ├── step4_validation_model_comparison.csv
│   │   │   ├── training_only_embedded_l1_feature_selection_report.csv
│   │   │   └── training_only_raw_feature_retention_summary.csv
│   │   ├── data_dictionary.csv
│   │   ├── event_cleaning_audit.csv
│   │   ├── event_sample_cleaned_for_eda_feature_engineering.csv
│   │   ├── event_sample_session_preserved_approx_1m.csv
│   │   ├── leakage_safe_feature_inventory.csv
│   │   ├── leakage_safe_feature_inventory_with_selection_status.csv
│   │   ├── leakage_safe_modeling_dataset_ready_for_split.csv
│   │   ├── missing_value_summary.csv
│   │   ├── step3_model_ready_dataset_validation.csv
│   │   ├── step3_training_numeric_correlation_matrix.csv
│   │   ├── step3_training_numeric_target_correlation.csv
│   │   ├── step3_training_pca_explained_variance.csv
│   │   ├── step3_training_pca_plot_sample.csv
│   │   ├── step4_final_lightgbm_raw_feature_importance.csv
│   │   ├── step4_final_lightgbm_test_evaluation_summary.csv
│   │   ├── step4_final_lightgbm_test_predictions.csv
│   │   ├── step4_final_lightgbm_transformed_feature_importance.csv
│   │   ├── step4_final_model_artifact_manifest.csv
│   │   ├── step4_final_model_comparison_report.csv
│   │   ├── step4_initial_validation_model_comparison.csv
│   │   ├── step4_lightgbm_time_series_tuning_results.csv
│   │   ├── step4_top_model_ranking_comparison.csv
│   │   ├── step4_top_model_threshold_comparison.csv
│   │   ├── step5_ethical_ai_artifact_manifest.csv
│   │   ├── step5_ethical_limitations_and_mitigations.csv
│   │   ├── step5_high_risk_segment_groups.csv
│   │   ├── step5_model_explainability_summary.csv
│   │   ├── step5_pdp_ice_explainability_summary.csv
│   │   ├── step5_responsible_use_guidelines.csv
│   │   ├── step5_segment_level_model_audit_results.csv
│   │   ├── step5_segment_level_model_audit_summary.csv
│   │   ├── training_only_embedded_l1_feature_selection_report.csv
│   │   └── training_only_raw_feature_retention_summary.csv
│   └── raw
│       ├── 2020-Apr.csv
├── models
│   ├── archive_old_step3_step4
│   │   ├── final_random_forest_model_metadata.json
│   │   ├── final_random_forest_purchase_propensity_pipeline.joblib
│   │   ├── step4_final_tuned_random_forest_metadata.json
│   │   └── step4_final_tuned_random_forest_pipeline.joblib
│   ├── step4_final_lightgbm_model_metadata.json
│   └── step4_final_lightgbm_purchase_propensity_pipeline.joblib
├── notebooks
│   ├── 01_problem understanding & framing.ipynb
│   ├── 02_data collection & understanding.ipynb
│   ├── 03_data preprocessing, applied EDA & feature engineering.ipynb
│   ├── 04_model implementation.ipynb
│   ├── 05_ethical thinking ethical AI & bias auditing.ipynb
│   ├── 06_technical presentation.css
│   ├── 06_technical presentation.ipynb
│   └── 06_technical presentation.slides.html
├── reports
│   ├── archive_old_step3_step4
│   │   ├── final_random_forest_feature_importance.png
│   │   └── step4_final_tuned_random_forest_feature_importance.png
│   ├── 06_business presentation.pdf
│   ├── 06_business_presentation.pptx
│   ├── 06_technical presentation.pdf
│   ├── step4_final_lightgbm_feature_importance.png
│   ├── step5_pdp_ice_early_cart_count.png
│   ├── step5_pdp_ice_early_event_gap_log1p.png
│   ├── step5_pdp_ice_early_unique_products.png
│   ├── step5_pdp_ice_price_for_aggregation_first.png
│   ├── step5_pdp_ice_price_for_aggregation_second.png
│   ├── step6_final_test_confusion_matrix.png
│   └── step6_validation_pr_auc_model_comparison.png
├── README.md
└── requirements.txt
```

---

## Installation Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create a Virtual Environment

For Windows PowerShell:

```bash
python -m venv .venv
.venv\Scripts\activate
```

For macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies from `requirements.txt`

This repository includes a `requirements.txt` file that lists the Python dependencies used in the project.

Install all dependencies with:

```bash
pip install -r requirements.txt
```

The dependency file includes the main packages used for:

- data analysis: `pandas`, `numpy`
- machine learning: `scikit-learn`, `joblib`, `lightgbm`, `xgboost`
- visualization: `matplotlib`
- notebooks: `jupyter`, `notebook`, `ipykernel`
- slideshow support: `rise`
- optional explainability: `shap`, `lime`

### 4. Optional Manual Installation

If `requirements.txt` is unavailable, the main dependencies can be installed manually:

```bash
pip install pandas numpy scikit-learn matplotlib joblib jupyter lightgbm xgboost
```

Optional packages:

```bash
pip install rise lime shap
```

Notes:

- `lightgbm` is required for the final selected model.
- `xgboost` is required if rerunning the full model-comparison notebook.
- `rise` is optional and only needed for live Jupyter slideshow presentations.
- `shap` and `lime` are optional explainability packages. PDP/ICE and model-based LightGBM feature importance were used as the final explainability approach because SHAP had local compatibility issues in the project environment.

---

## How to Run the Code

### 1. Download the Dataset

Download `2020-Apr.csv` from the Kaggle dataset page and place it here:

```text
data/raw/2020-Apr.csv
```

Because the file is large, it should usually not be committed to GitHub.

### 2. Install Project Dependencies

After activating the virtual environment, run:

```bash
pip install -r requirements.txt
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

Open the notebooks in numerical order.

### 4. Recommended Notebook Run Order

Run the notebooks in this order:

```text
01_dataset_overview_data_dictionary.ipynb
02_exploratory_data_analysis.ipynb
03_data_preprocessing_eda_feature_engineering.ipynb
04_model_implementation.ipynb
05_ethical_thinking_ethical_ai_bias_auditing.ipynb
```

### 5. Main Outputs by Step

| Step | Main Outputs |
|---|---|
| Step 1 | Dataset overview and data dictionary |
| Step 2 | EDA summaries and visualizations |
| Step 3 | Cleaned event dataset, leakage-safe modeling dataset, feature inventory, PCA, feature selection |
| Step 4 | Model comparison, tuned models, final LightGBM model, test predictions, feature importance |
| Step 5 | Ethical AI audit, segment-level bias audit, PDP/ICE explainability, mitigation plan |

### 6. Final Model Artifacts

The final model and metadata are saved in:

```text
models/step4_final_lightgbm_purchase_propensity_pipeline.joblib
models/step4_final_lightgbm_model_metadata.json
```

Final test predictions and evaluation summaries are saved in:

```text
data/processed/step4_final_lightgbm_test_predictions.csv
data/processed/step4_final_lightgbm_test_evaluation_summary.csv
```

---

## Python Dependencies

The project dependencies are managed through:

```text
requirements.txt
```

Current dependency groups:

| Category | Packages |
|---|---|
| Core data analysis | `pandas`, `numpy` |
| Machine learning | `scikit-learn`, `joblib`, `lightgbm`, `xgboost` |
| Visualization | `matplotlib` |
| Notebook environment | `jupyter`, `notebook`, `ipykernel` |
| Slideshow support | `rise` |
| Optional explainability | `shap`, `lime` |

To reproduce the environment, run:

```bash
pip install -r requirements.txt
```

If using a newer Python version, some optional explainability packages may require version-specific compatibility checks. If SHAP installation or execution fails, the project can still be reproduced using LightGBM feature importance and PDP/ICE plots.

---

## Results Summary

### Model Comparison

Six supervised classification models were trained and compared using the same leakage-safe dataset and chronological validation split:

| Model | Validation PR-AUC |
|---|---:|
| LightGBM | 0.4877 |
| XGBoost | 0.4817 |
| Random Forest | 0.4730 |
| Decision Tree | 0.4517 |
| Extra Trees | 0.4494 |
| Logistic Regression | 0.4484 |

LightGBM achieved the highest validation PR-AUC and was selected as the final model candidate.

### Final Model

The selected model was retrained using the combined training and validation sets, then evaluated once on the untouched chronological test set.

| Metric | Test Result |
|---|---:|
| Final model | LightGBM |
| Selected threshold | 0.65 |
| Test PR-AUC | 0.4824 |
| Test ROC-AUC | 0.7802 |
| Accuracy | 0.8902 |
| Precision | 0.6138 |
| Recall | 0.4896 |
| F1-score | 0.5447 |
| Test sessions targeted | 1,489 |
| Percentage targeted | 10.70% |

### Final Confusion Matrix

| Actual / Predicted | No Future Purchase | Future Purchase |
|---|---:|---:|
| No Future Purchase | 11,478 | 575 |
| Future Purchase | 953 | 914 |

### Business Interpretation

The model targeted 1,489 out of 13,920 test sessions. Among those targeted sessions, 914 became future purchasers.

The model-targeted precision was **61.38%**, compared with the overall test future-purchase rate of **13.41%**. This indicates that the model can identify a much higher-intent group than random or broad targeting.

### Explainability Summary

The top LightGBM feature drivers were:

1. `early_event_gap_log1p`
2. `price_for_aggregation_second`
3. `price_for_aggregation_first`
4. `brand_clean_second`
5. `category_code_clean_second`

PDP/ICE analysis showed that early cart activity strongly increased the model's predicted purchase probability. Timing, product price, brand, and category also contributed to model behavior.

### Ethical AI and Bias Audit Summary

The dataset did not contain direct sensitive demographic attributes such as gender, race, age, income, or socioeconomic status. Because of this, formal demographic fairness metrics could not be calculated.

Instead, the project performed segment-level audits using available non-sensitive fields such as product category, brand, price band, event timing, early cart behavior, and early event-gap behavior.

The main responsible AI finding was that the model strongly favors sessions with immediate-intent signals, especially early cart activity. This is useful for targeting, but it may under-target slower or exploratory shoppers. Therefore, the model should be used as a prioritization tool, not as a system for denying offers, reducing service quality, or labeling customers as low value.

---

## Recommended Use

This model is suitable for:

- Marketing prioritization
- Remarketing campaign support
- Cart recovery prioritization
- Early-session customer engagement
- Purchase-propensity scoring experiments

This model should not be used for:

- Denying customer access to offers
- Reducing customer service quality
- Permanently classifying users as low value
- Making unsupported demographic fairness claims

A controlled pilot is recommended before any production deployment.

---

## Author Information

**Author:** Dominic Gacad  
**Program:** Post Graduate Diploma in AI and Machine Learning, Asian Institute of Management  
**Project Type:** Capstone Project  
**Topic:** eCommerce purchase intent prediction using early session behavior  
**Location:** Philippines  

---

## Acknowledgments

Dataset source: Michael Kechinov, *eCommerce behavior data from multi category store*, Kaggle.

This project was completed as part of an AI and Machine Learning capstone workflow covering data exploration, preprocessing, feature engineering, model implementation, ethical AI auditing, and final communication.
