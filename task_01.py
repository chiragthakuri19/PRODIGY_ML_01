import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load Dataset directly via raw URL
url = "https://raw.githubusercontent.com/selva86/datasets/master/HousePrices_train.csv"
df = pd.read_csv(url)

# 2. Select Features & Target
# GrLivArea: Living Area Sq Ft, BedroomAbvGr: Bedrooms, FullBath: Bathrooms
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
target = "SalePrice"

# Drop missing values
data = df[features + [target]].dropna()

X = data[features]
y = data[target]

# 3. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Coefficients (Weights): {model.coef_}")
print(f"Intercept (Bias): {model.intercept_}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2 Score: {r2:.4f}")
