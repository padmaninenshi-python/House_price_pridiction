import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Load the model directly (it is small enough now!)
model = joblib.load('house_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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