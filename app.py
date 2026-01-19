from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("house_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array([[
        data["size_sqft"],
        data["bedrooms"],
        data["bathrooms"],
        data["location_type"],
        data["furnishing"]
    ]])

    prediction = model.predict(features)[0]

    return jsonify({"prediction": float(prediction)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
