import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import pickle
import numpy as np


# load pikle file
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# load the model
model = load_model('ann_model.h5')

# load the encoder and scaler
with open('geo_encoder.pkl', 'rb') as f:
    geo_encoder = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)


# Example input data
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}
geo_encoded = geo_encoder.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))

input_data_df = pd.DataFrame([input_data])
input_data = pd.concat([input_data_df, geo_encoded_df], axis=1)

input_data_df['Gender'] = label_encoder_gender.transform(input_data_df['Gender'])

input_data_df = input_data_df.drop(['Geography'], axis=1)

# concatenate the input data with the encoded geography
input_data_df = pd.concat([input_data_df, geo_encoded_df], axis=1)
print(input_data_df)

# scale the input data
input_data_scaled = scaler.transform(input_data_df)
print(input_data_scaled)
# make prediction
prediction = model.predict(input_data_scaled)
print("Prediction:", prediction)
# Convert prediction to binary class
predicted_class = (prediction > 0.5).astype(int)
print("Predicted class:", predicted_class)
