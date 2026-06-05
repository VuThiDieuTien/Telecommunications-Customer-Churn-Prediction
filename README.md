# Telecommunications Customer Churn Prediction

## Overview

This project aims to predict customer churn in the telecommunications industry using machine learning techniques. By analyzing customer demographics, service usage, contract information, and billing data, the project identifies customers who are likely to discontinue the service and provides actionable insights to improve customer retention strategies.

## Objectives

- Analyze customer behavior and churn patterns.
- Identify key factors influencing customer attrition.
- Develop and compare multiple machine learning models for churn prediction.
- Improve business decision-making through model explainability and data-driven insights.

## Dataset

- Industry: Telecommunications
- Records: 7,043 customers
- Features: 38 variables
- Target Variable: Customer Churn (Yes/No)

## Methodology

The project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** framework:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment Planning

## Technologies

- Python
- Scikit-learn
- XGBoost
- Random Forest
- Support Vector Machine (SVM)
- SHAP

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Investigated customer demographics and service usage patterns.
- Identified higher churn rates among:
  - Month-to-month contract customers
  - Customers with shorter tenure
  - Customers with higher monthly charges

### 2. Machine Learning Modeling

Built and compared multiple classification models:

- Support Vector Machine (SVM)
- Random Forest
- XGBoost

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### 3. Model Performance

- XGBoost achieved the best performance.
- ROC-AUC score exceeded **0.93**, demonstrating strong predictive capability and model generalization.

### 4. Model Explainability

Applied **SHAP (SHapley Additive exPlanations)** to interpret model predictions and identify the most influential churn drivers.

Key factors included:

- Tenure
- Contract Type
- Monthly Charges
- Referral Count
- Number of Dependents

## Results

- Successfully identified high-risk customers.
- Generated actionable customer retention insights.
- Provided business recommendations to reduce churn, particularly for:
  - Newly acquired customers
  - Month-to-month contract users
