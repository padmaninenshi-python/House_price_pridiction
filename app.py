import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
import joblib
import os

app = Flask(__name__)

# Load the model
model = joblib.load('house_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

# --- FIX: Allow both POST (button) and GET (direct link) ---
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # If someone clicks the link directly, send them back to Home
    if request.method == 'GET':
        return redirect(url_for('home'))

    # If it's a real prediction request (POST)
    try:
        # Get values from the form
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # Predict
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        
        return render_template('index.html', prediction_text=f'Estimated House Price is: ₹{output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)