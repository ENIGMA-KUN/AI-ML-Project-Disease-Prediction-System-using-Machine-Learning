import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and feature names
model = joblib.load('C:\Users\chakr\Documents\GitHub\medical-ML\Logistic Regression.pkl')
feature_names = joblib.load('path/to/your/feature_names.pkl')  # Assuming feature names are saved separately

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data['symptoms']
    
    # Convert symptoms to a DataFrame with correct feature names
    symptoms_df = pd.DataFrame([symptoms], columns=feature_names)
    
    # Make prediction
    prediction = model.predict(symptoms_df)[0]
    
    # Convert prediction to a Python int type
    prediction = int(prediction)
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
