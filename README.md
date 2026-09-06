# Corizo Minor Project

## Project Overview
This repository contains the **Data Science Minor Capstone Project** completed as part of the **Corizo Internship Program**. The project focuses on predicting product yield (Pass/Fail) in the **Semiconductor Manufacturing Process** using machine learning classification algorithms applied to high-dimensional sensor data (SECOM dataset).

- **Author:** Subrahmanya Manjunatha Bhat
- **Project Type:** Capstone Project 1 (Minor Project)
- **Domain:** Data Science / Semiconductor Manufacturing

---

## Problem Statement
In semiconductor fabrication, process entities are continuously monitored by hundreds of sensors. Identifying defective wafer yields (Fail) early in the manufacturing pipeline is critical for quality assurance and reducing production costs. However, semiconductor yield datasets present severe challenges:
1. **Extreme Class Imbalance:** Over 93% of manufactured units pass, while fewer than 7% fail.
2. **High Dimensionality:** Hundreds of sensor variables with high noise and redundancy.
3. **Missing Values:** Extensive missing sensor signal values across measurements.

The challenge is to build a robust model capable of maximizing minority **Fail-class Recall** without generating excessive false alarms.

---

## Objectives
- **Data Cleansing:** Identify and eliminate invalid sensor features with >70% missing values, duplicate sensor features, and zero-variance attributes.
- **Exploratory Data Analysis:** Analyze missingness patterns, class imbalance, sensor correlations, and feature distributions.
- **Class Imbalance Handling:** Apply SMOTE (Synthetic Minority Over-sampling Technique) to address minority class under-representation.
- **Feature Selection:** Perform Random Forest Feature Importance ranking to extract the top 100 process-critical sensor features out of 590 attributes.
- **Model Evaluation & Tuning:** Compare Logistic Regression, Support Vector Machine (SVM), and Random Forest models across standard metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC).
- **Threshold Optimization:** Optimize classification decision thresholds via 5-fold Stratified Cross-Validation to maximize Fail-class detection.

---

## Technologies Used
- **Programming Language:** Python 3.x
- **Data Manipulation:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning & Preprocessing:** Scikit-Learn
- **Class Imbalance Handling:** Imbalanced-Learn (SMOTE, Pipeline)
- **Model Persistence:** Joblib
- **Environment:** Jupyter Notebook / Antigravity IDE

---

## Dataset
- **Dataset Name:** Semiconductor Manufacturing Process (SECOM) Sensor Dataset (`signal-data.csv`)
- **Observations:** 1,567 manufacturing samples
- **Attributes:** 592 columns (590 numerical sensor signals, 1 timestamp, 1 target label)
- **Target Encoding:**
  - `-1` : **Pass** (1,463 samples | 93.36%)
  - `1` : **Fail** (104 samples | 6.64%)

---

## Methodology

```
+--------------------------+
|  Raw SECOM Sensor Data   |
|   (1567 x 592 Signals)   |
+------------+-------------+
             |
             v
+--------------------------+
| Data Cleansing & Filter  |
| - Drop >70% missing (8)  |
| - Drop duplicates (104)  |
| - Median Imputation      |
+------------+-------------+
             |
             v
+--------------------------+
|  Exploratory Data Analysis|
| - Class Imbalance Check  |
| - Correlation Heatmaps   |
+------------+-------------+
             |
             v
+--------------------------+
|  Preprocessing & SMOTE   |
| - Train-Test Split (80/20|
| - StandardScaler         |
| - SMOTE Resampling       |
+------------+-------------+
             |
             v
+--------------------------+
| Feature Selection & Models|
| - RF Top 100 Features    |
| - LogReg / SVM / RF      |
+------------+-------------+
             |
             v
+--------------------------+
| Threshold Optimization   |
| - Optimal Threshold: 0.40|
| - Final Model Evaluation |
+--------------------------+
```

1. **Cleansing:** Removed 8 high-missing columns (>70%) and 104 duplicate columns, reducing attributes from 592 to 480.
2. **Imputation & Scaling:** Applied median imputation (`SimpleImputer`) and feature standardization (`StandardScaler`).
3. **Resampling:** Used SMOTE within `imblearn.pipeline` to prevent data leakage during train/test splits.
4. **Feature Importance:** Extracted Gini importance rankings using Random Forest to select the top 100 sensor features.
5. **Decision Threshold Tuning:** Conducted cross-validation probability analysis to tune decision threshold from 0.50 down to 0.40.

---

## Results

### Model Performance Comparison

| Model Architecture | Features Used | Test Accuracy | Fail Precision | Fail Recall | Fail F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 480 | 79.94% | 0.12 | 0.38 | 0.18 | 0.6980 |
| Support Vector Machine (SVM + SMOTE) | 480 | 84.71% | 0.17 | 0.43 | 0.24 | 0.7240 |
| Random Forest (SMOTE) | 480 | **92.99%** | 0.00 | 0.00 | 0.00 | 0.7710 |
| **Final Logistic Regression (SMOTE + 100 Features)** | **100** | **78.34%** | **0.27** | **90.48%** | **0.4222** | **0.8507** |

> **Key Finding:** While default Random Forest achieved high overall accuracy (92.99%), it failed entirely on the minority Fail class (0% recall). The **Final Optimized Logistic Regression Model** (100 features, SMOTE, threshold=0.40) successfully detected **90.48% (19/21)** of defective yields on unseen test data with a **ROC-AUC of 0.8507**.

---

## How to Run

### Prerequisites
Clone the repository and install required packages:
```bash
git clone https://github.com/SUBRAHMANYA2726/NEW_CORIZO_DS_MINOR.git
cd NEW_CORIZO_DS_MINOR
pip install -r requirements.txt
```

### Dataset Setup
1. Download `signal-data.csv` (SECOM Dataset).
2. Place `signal-data.csv` in the root folder or `data/` directory.

### Running the Notebook
Launch Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook notebooks/Corizo_Capstone_1_Minor_DataScience.ipynb
```
Run all cells sequentially to execute data preprocessing, exploratory analysis, model training, feature selection, and evaluation.

---

## Project Structure

```
NEW_CORIZO_DS_MINOR/
├── Corizo_Capstone_1_Minor_DataScience.ipynb  # Main Jupyter Notebook
├── notebooks/
│   └── Corizo_Capstone_1_Minor_DataScience.ipynb # Categorized Notebook copy
├── data/
│   └── README.md                              # Dataset setup guidelines
├── models/
│   └── README.md                              # Model pipeline specs
├── results/                                   # Folder for generated plots/metrics
├── .gitignore                                 # Git exclusion file
├── LICENSE                                    # MIT License
├── README.md                                  # Project Documentation
└── requirements.txt                           # Python dependencies
```

---

## Conclusion
- High-dimensional sensor data in semiconductor manufacturing contains substantial redundancy (104 duplicate attributes and 8 invalid features dropped).
- Standard classification models defaulting to a 0.50 threshold fail on extreme class imbalance (93.4% Pass vs 6.6% Fail).
- **SMOTE resampling**, combined with **Random Forest Feature Selection (top 100 features)** and **Logistic Regression Threshold Tuning (0.40)**, achieved an exceptional **90.48% Fail Recall** and **0.8507 ROC-AUC**, providing a reliable industrial quality inspection pipeline.
