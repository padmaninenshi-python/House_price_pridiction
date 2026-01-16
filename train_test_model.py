import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# =========================
# Load dataset
# =========================
data = pd.read_csv("house_data.csv")  # Replace with your CSV file

# =========================
# Features and target
# =========================
X = data[['size_sqft', 'bedrooms', 'bathrooms', 'location_type', 'furnishing']].copy()
y = data['rent']

# =========================
# Encode categorical columns safely
# =========================
for col in ['location_type', 'furnishing']:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col])  # Fixes SettingWithCopyWarning

# =========================
# Split data: 80% train, 20% test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Show ratio info
total_samples = len(X_train) + len(X_test)
print(f"Training data: {len(X_train)} samples")
print(f"Testing data: {len(X_test)} samples")
print(f"Training ratio: {len(X_train)/total_samples:.2f}")
print(f"Testing ratio: {len(X_test)/total_samples:.2f}")

# =========================
# Train Random Forest
# Optimized for large datasets
# =========================
model = RandomForestRegressor(
    n_estimators=100,  # number of trees
    max_depth=20,      # limit depth to speed up training
    n_jobs=-1,         # use all CPU cores
    random_state=42
)
print("\nTraining model, please wait...")
model.fit(X_train, y_train)

# =========================
# Predictions & performance
# =========================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\n---Model Performance---")
print(f"Train R2 Score: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test R2 Score: {r2_score(y_test, y_test_pred):.4f}")

# RMSE (works on all sklearn versions)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# =========================
# Save model
# =========================
joblib.dump(model, 'house_model.pkl')
print("\nModel saved as house_model.pkl")
