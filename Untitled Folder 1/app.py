import streamlit as st
import joblib
import numpy as np

# Load saved models
logistic_model = joblib.load('logistic.pkl')
knn_model1 = joblib.load('knn1.pkl')
knn_model2 = joblib.load('knn2.pkl')
svm_model = joblib.load('svm.pkl')
rf_model = joblib.load('randomforest.pkl')

# Function to make predictions
def predict(model, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app UI
st.title('Heart Disease Prediction')

# Collect user input for the prediction
st.header('Input Patient Data')

age = st.number_input('Age', min_value=1, max_value=120, value=50)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
cp = st.selectbox('Chest Pain Type (0-3)', options=[0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120)
chol = st.number_input('Serum Cholesterol (chol)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
restecg = st.selectbox('Resting ECG Results (0-2)', options=[0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', options=[0, 1])
oldpeak = st.number_input('ST Depression (oldpeak)', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment (0-2)', options=[0, 1, 2])
ca = st.selectbox('Number of Major Vessels (0-4)', options=[0, 1, 2, 3, 4])
thal = st.selectbox('Thalassemia (1=normal, 2=fixed defect, 3=reversable defect)', options=[1, 2, 3])

# Model selection
st.header('Select a Model')
model_choice = st.selectbox('Choose a model', ['Logistic Regression', 'KNN Model 1', 'KNN Model 2', 'SVM', 'Random Forest'])

# Button for making predictions
if st.button('Predict'):
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    
    if model_choice == 'Logistic Regression':
        result = predict(logistic_model, input_data)
    elif model_choice == 'KNN Model 1':
        result = predict(knn_model1, input_data)
    elif model_choice == 'KNN Model 2':
        result = predict(knn_model2, input_data)
    elif model_choice == 'SVM':
        result = predict(svm_model, input_data)
    elif model_choice == 'Random Forest':
        result = predict(rf_model, input_data)
    
    if result == 1:
        st.success('The patient is likely to have heart disease.')
    else:
        st.success('The patient is unlikely to have heart disease.')
