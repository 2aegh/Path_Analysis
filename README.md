
# Customer Path Analysis

The objective of this project is to develop a predictive model that can effectively identify conversion paths leading to a high return on ad spend (ROI). This model aims to optimize advertising strategies by focusing on the most effective customer journeys for achieving a strong return on investment.


## Overview
The project involves the following key steps:

1. **Load Data:** Gather data from various sources representing customer paths and associated metrics.
(Line 1 to 5)

2. **Data Cleansing - Feature Engineering:** Clean and preprocess the data, and engineer relevant features for analysis. 
(Line 8 to 83)

3. **Data Visualization:** Generate visual representations of the data to gain insights and inform decision-making. 
(Line 86 to 228)

4. **Data Preparation:** Further prepare the data for model training and evaluation. 
(Line 231 to 281)

5. **Training Model (Buyer/Non-Buyer):** Develop a classification model to predict if a customer will make a purchase. 
(Line 284 to 352)

6. **Training Model (ROI Level):** Create a model to categorize the paths that lead to a purchase into different ROI levels. 
(Line 355 to 409)

7. **Machine Learning Models:** Apply various machine learning algorithms to analyze and evaluate the importance of each features. 
(Line 412 to 540)

## Feature Engineering

- **ROI (Return on Investment):** Calculated as the ratio of sales to costs.
- **ROI per Impression:** Calculated by dividing ROI by the number of impressions.
- **ROI Category:** Rows are categorized into two sections: 'Buyer' (rows with sales) and 'Non-Buyer' (rows without sales).
- **ROI Level:** Rows are categorized into two sections: 'High ROI' (ROI greater than 15) and 'Low ROI' (ROI between 15 and 0).

## Additional Details
The first approach treats the problem as a sequential prediction task, focusing on the initial 50 steps of each path.

Class balance is achieved to prevent overfitting on zero ROI data.

A classification model is designed to predict if a customer will make a purchase based on their path.

A supplementary model is constructed to assess ROI levels for paths leading to purchases.
## Classification Model with LSTM
In the initial phase, a classification model utilizing the LSTM algorithm was employed to predict the likelihood of a purchase along specific paths. The achieved accuracy rates range from an impressive 65% to 75%. To further enhance model performance, I applied Bayesian optimization with Gaussian Processes through scikit-optimize, resulting in optimal hyperparameters for the LSTM model.

The resulting model exhibited exceptional accuracy, achieving 82% accuracy in predicting buyers and 70% accuracy in identifying non-buyers. The accompanying confusion matrix confirms its effectiveness, with a minor occurrence of false positives.
## ROI Level Classification
In the subsequent phase, I categorized ROIs into high and low brackets, forecasting outcomes based on the customer's path. By employing the same hyperparameter optimization process, the model was fine-tuned to achieve accuracies ranging from 55% to 65%. Notably, the model demonstrated significant proficiency, particularly in predicting high ROIs, achieving an accuracy of 71%.
## Regression Model for ROI Prediction
The final phase involved constructing a regression model aimed at precisely predicting ROI figures. Utilizing both non-parametric (Random Forest regressor) and parametric models (Linear Regression, Ridge, and Lasso), I sought to assess the influence of path length on the outcome. Unfortunately, the results indicated a need for further refinement, as the predictability of ROI fell short of expectations.
## Further Improvements
To enhance the project, consider the following:

Incorporate real-time data feeds for dynamic model updates.
Implement a user interface for easy model interaction.

## Contact
For questions or feedback, please contact Amirali Eghtesad at amirali.egh@gmail.com .

