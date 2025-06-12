import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

# Load the trained RNN model
model = load_model('rnn_imdb_model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Load the IMDB dataset
max_features = 10000  # Number of unique words to consider

word_index = imdb.get_word_index()
# Reverse the word index to get words from indices
reverse_word_index = {v: k for k, v in word_index.items()}

# Function to decode review from indices
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to prepare input data
def prepare_input_data(review):
    encoded_review = [word_index.get(word, 0) for word in review.split()]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# prediction function
def predict_review(review):
    # Prepare the input data
    input_data = prepare_input_data(review)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Convert prediction to binary class
    # predicted_class = (prediction > 0.5).astype(int)
    
    return prediction, decode_review(input_data[0])

# # Example usage
data_review = "This movie was fantastic! The plot was engaging and the acting was superb."
predicted_class, decoded_review = predict_review(data_review)
# Print the results
print(f"Review: {decoded_review}")
print(f"predicted_class: {predicted_class}")
