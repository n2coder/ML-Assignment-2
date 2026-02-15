
# Breast Cancer Diagnosis Prediction - ML Assignment 2

## a. Problem Statement
The objective of this project is to develop and evaluate multiple machine learning classification models to predict whether a breast mass is Malignant (M) or Benign (B). This is achieved using features derived from digitized images of fine needle aspirate (FNA) of breast masses.

## b. Dataset Description
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)[cite: 68].
- **Features:** 30 numeric attributes (radius, texture, perimeter, area, etc.).
- **Instances:** 569 samples.
- **Target Variable:** 'diagnosis' (M = Malignant, B = Benign).

## c. Models Used and Comparison Table
The following 6 models were implemented and evaluated using the same dataset[cite: 33, 69]:

| ML Model Name       |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:--------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.9737 | 0.9974 |      0.9762 |   0.9535 | 0.9647 | 0.9439 |
| Decision Tree       |     0.9298 | 0.9253 |      0.907  |   0.907  | 0.907  | 0.8506 |
| KNN                 |     0.9474 | 0.982  |      0.9302 |   0.9302 | 0.9302 | 0.888  |
| Naive Bayes         |     0.9649 | 0.9974 |      0.9756 |   0.9302 | 0.9524 | 0.9253 |
| Random Forest       |     0.9649 | 0.9969 |      0.9756 |   0.9302 | 0.9524 | 0.9253 |
| XGBoost             |     0.9561 | 0.9908 |      0.9524 |   0.9302 | 0.9412 | 0.9064 |

## d. Performance Observations
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Often serves as a strong baseline for binary classification. |
| **Decision Tree** | Captures non-linear relationships but can be prone to overfitting. |
| **kNN** | Performance depends on feature scaling and the choice of K. |
| **Naive Bayes** | Fast and effective, even with the assumption of feature independence. |
| **Random Forest** | Provides a robust ensemble approach by averaging multiple trees. |
| **XGBoost** | Typically the most powerful model for tabular data due to gradient boosting. |



