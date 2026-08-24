# PRODIGY_ML_01: House Price Prediction

This repository contains the implementation of a **Linear Regression** model to predict house prices based on square footage, number of bedrooms, and bathrooms.

## Dataset
- Kaggle House Prices: Advanced Regression Techniques Dataset

## Features Used
- `GrLivArea`: Above ground living area in square feet
- `BedroomAbvGr`: Number of bedrooms above ground
- `FullBath`: Number of full bathrooms
- `SalePrice`: Target variable (House Price)

## Methodology
1. **Data Loading & Preprocessing**: Cleaned missing records and isolated key predictive features.
2. **Train-Test Split**: Divided data into 80% training and 20% testing sets.
3. **Model Training**: Trained a Scikit-Learn `LinearRegression` model.
4. **Evaluation**: Evaluated performance using Root Mean Squared Error (RMSE) and R² Score.
