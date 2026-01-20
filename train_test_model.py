import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# 1. Load Data
print("Loading data...")
df = pd.read_excel('house_data.xlsx') # Make sure this matches your file name

# 2. Prepare Features
# (Adjust these column names if your excel file is different)
X = df.drop('Price', axis=1) 
y = df['Price']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train "Lite" Model
# Reduced n_estimators and max_depth to make the file small (<100MB)
print("Training Lite Model (this is faster)...")
model = RandomForestRegressor(
    n_estimators=50,      # Reduced from 100 to 50
    max_depth=10,         # Limited depth to stop it from growing huge
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Model R2 Score: {r2:.4f}")

# 6. Save Model
print("Saving model...")
joblib.dump(model, 'house_model.pkl')
print("Done! Check the file size of 'house_model.pkl'. It should be small now.")