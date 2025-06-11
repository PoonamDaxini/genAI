import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import pickle
import numpy as np


# load the model
model = load_model('ann_regression_model.h5')

# load the encoder and scaler
with open('geo_encoder.pkl', 'rb') as f:
    geo_encoder = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

# load pickle file
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# streamlit app
st.title('Estimate Salary Prediction')
st.write('Enter customer details to predict churn.')

# user input
geography = st.selectbox('Geography', geo_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = st.selectbox('Exited', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

print("Input Data:", input_data)

geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))
# concatenate the input data with the encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
print("Input Data:", input_data)

# scale the input data
input_data_scaled = scaler.transform(input_data)
# make prediction
prediction = model.predict(input_data_scaled)
st.write("Predicted Estimated Salary:", prediction[0][0])

'''
# Convert prediction to binary class
predicted_class = (prediction > 0.5).astype(int)
st.write("Predicted Class:", predicted_class[0][0])
# Display the input data for debugging
st.write("Input Data:", input_data)
# Display the scaled input data for debugging
st.write("Scaled Input Data:", input_data_scaled)
# Display the prediction for debugging
st.write("Prediction Array:", prediction)

'''