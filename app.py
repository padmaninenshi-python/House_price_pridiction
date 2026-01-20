import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
import os
import urllib.request
import zipfile

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_FILE = 'house_model.pkl'  # The file inside the zip
ZIP_FILE = 'house_model.zip'    # The downloaded zip name

# CORRECT URL PLACEMENT:
MODEL_URL = "https://github.com/padmaninenshi-python/House_price_pridiction/releases/download/v1/house_model.zip" 

# --- MODEL LOADER ---
def get_model():
    # 1. Check if the raw model file already exists
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)

    # 2. If not, check if we have the zip file; if not, download it
    if not os.path.exists(ZIP_FILE):
        print("Model not found. Downloading zip from GitHub...")
        try:
            urllib.request.urlretrieve(MODEL_URL, ZIP_FILE)
            print("Download complete!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None

    # 3. Unzip the file
    print("Unzipping model...")
    try:
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(".") # Extract to current folder
        print("Unzip successful!")
    except Exception as e:
        print(f"Error unzipping model: {e}")
        return None

    # 4. Load the extracted model
    return joblib.load(MODEL_FILE)

model = get_model()

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: Model could not be loaded."
        
    try:
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f'Estimated House Price is: ₹{output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error in prediction: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)