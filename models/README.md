# Trained Models Overview

This directory contains details about the machine learning model pipelines developed for predicting semiconductor yield Pass/Fail outcomes.

## Model Summary

- **Primary Architecture:** Logistic Regression with SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature Selection:** Top 100 Sensor Features selected via Random Forest Feature Importance
- **Optimal Threshold:** Decision threshold tuned to 0.40 to maximize minority Fail-class recall (90.48%) and F1-score (0.4222)
- **Saved Model Artifact:** `final_semiconductor_yield_model.pkl` (Generated during notebook execution)

## Benchmark Comparison

| Model | Sampling | Features | Accuracy | Fail Precision | Fail Recall | Fail F1-Score |
|---|---|---|---|---|---|---|
| Logistic Regression | Raw (Imbalanced) | 480 | 79.94% | 0.12 | 0.38 | 0.18 |
| SVM (RBF Kernel) | SMOTE | 480 | 84.71% | 0.17 | 0.43 | 0.24 |
| Random Forest | SMOTE | 480 | **92.99%** | 0.00 | 0.00 | 0.00 |
| **Final Logistic Regression** | **SMOTE** | **100** | **78.34%** | **0.27** | **0.90** | **0.42** |
