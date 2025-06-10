import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import pickle
import numpy as np


# load the model
model = load_model('ann_model.h5')

# load the encoder and scaler
with open('geo_encoder.pkl', 'rb') as f:
    geo_encoder = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)


# load pikle file
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# streamlit app
st.title('Customer Churn Prediction')
st.write('Enter customer details to predict churn.')

# user input
geography = st.selectbox('Geography', geo_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
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
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))
# concatenate the input data with the encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
# scale the input data

input_data_scaled = scaler.transform(input_data)
# make prediction
prediction = model.predict(input_data_scaled)
st.write("Prediction:", prediction)

# Convert prediction to binary class
predicted_class = (prediction > 0.5).astype(int)
st.write("Predicted class:", predicted_class)
if predicted_class[0][0] == 1:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
'''
# Display the input data
st.write("Input Data:")
st.dataframe(input_data)
# Display the scaled input data
st.write("Scaled Input Data:")
st.dataframe(input_data_scaled)
# Display the prediction
st.write("Prediction Result:")
st.write(f'Churn Probability: {prediction[0][0]:.2f}')
if prediction[0][0] > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
# Display the model summary
st.write("Model Summary:")
model.summary(print_fn=lambda x: st.text(x))
# Display the model architecture
st.write("Model Architecture:")
st.text(model.to_json())
# Display the model weights
st.write("Model Weights:")
weights = model.get_weights()
for i, weight in enumerate(weights):
    st.write(f"Layer {i} weights:")
    st.write(weight)
# Display the model configuration
st.write("Model Configuration:")
st.write(model.get_config())
# Display the model optimizer configuration
st.write("Model Optimizer Configuration:")
st.write(model.optimizer.get_config())
# Display the model loss and accuracy
st.write("Model Loss and Accuracy:")
loss, accuracy = model.evaluate(input_data_scaled, predicted_class, verbose=0)
st.write(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
# Display the model training history
st.write("Model Training History:")
if hasattr(model, 'history'):
    history = model.history.history
    st.write("Loss History:")
    st.line_chart(history['loss'])
    st.write("Accuracy History:")
    st.line_chart(history['accuracy'])
else:
    st.write("No training history available.")
# Display the model input shape
st.write("Model Input Shape:")
st.write(model.input_shape)
# Display the model output shape
st.write("Model Output Shape:")
st.write(model.output_shape)
# Display the model layers
st.write("Model Layers:")
for layer in model.layers:
    st.write(f"Layer {layer.name}: {layer.output_shape} - {layer.get_config()}")
# Display the model metrics
st.write("Model Metrics:")
st.write(model.metrics_names)
# Display the model callbacks
st.write("Model Callbacks:")
st.write(model.callbacks)
# Display the model training parameters
st.write("Model Training Parameters:")
st.write(model.trainable_variables)
# Display the model evaluation results
st.write("Model Evaluation Results:")
loss, accuracy = model.evaluate(input_data_scaled, predicted_class, verbose=0)
st.write(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
# Display the model prediction
st.write("Model Prediction:")
st.write(prediction)
# Display the model prediction class
st.write("Model Prediction Class:")
st.write(predicted_class)
# Display the model prediction probability
st.write("Model Prediction Probability:")
st.write(f'Churn Probability: {prediction[0][0]:.2f}')
# Display the model prediction class
st.write("Model Prediction Class:")
st.write(predicted_class[0][0])
# Display the model prediction result
st.write("Model Prediction Result:")
if predicted_class[0][0] == 1:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
# Display the model prediction result with probability
st.write("Model Prediction Result with Probability:")
if prediction[0][0] > 0.5:
    st.write(f'The customer is likely to churn with a probability of {prediction[0][0]:.2f}.')
else:
    st.write(f'The customer is not likely to churn with a probability of {1 - prediction[0][0]:.2f}.')
# Display the model prediction result with class
st.write("Model Prediction Result with Class:")
if predicted_class[0][0] == 1:
    st.write(f'The customer is likely to churn with a class of {predicted_class[0][0]}.')
else:
    st.write(f'The customer is not likely to churn with a class of {predicted_class[0][0]}.')
# Display the model prediction result with probability and class
'''