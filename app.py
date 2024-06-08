import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import joblib

# Load the trained models and other necessary data
def load_models():
    final_rf_model = joblib.load('final_rf_model.pkl')
    final_nb_model = joblib.load('final_nb_model.pkl')
    final_svm_model = joblib.load('final_svm_model.pkl')
    data_dict = joblib.load('data_dict.pkl')
    encoder = joblib.load('encoder.pkl')
    X_columns = joblib.load('X_columns.pkl')
    return final_rf_model, final_nb_model, final_svm_model, data_dict, encoder, X_columns

final_rf_model, final_nb_model, final_svm_model, data_dict, encoder, X_columns = load_models()

# Define the prediction function
def predict_disease(symptoms):
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        if symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
    input_df = pd.DataFrame([input_data], columns=X_columns)
    
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_df)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_df)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_df)[0]]
    
    final_prediction = Counter([rf_prediction, nb_prediction, svm_prediction]).most_common(1)[0][0]
    
    return final_prediction

# Streamlit UI
st.title('Disease Prediction System')
st.write('Select symptoms and click "Predict" to get the disease prediction.')

# Create a multi-select box for symptoms
symptoms = st.multiselect('Symptoms', options=list(data_dict["symptom_index"].keys()))

if st.button('Predict'):
    if symptoms:
        prediction = predict_disease(symptoms)
        st.write(f'Predicted Disease: {prediction}')
    else:
        st.write('Please select at least one symptom.')
