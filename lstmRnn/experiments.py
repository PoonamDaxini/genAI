import nltk
import tensorflow as tf
nltk.download('gutenberg')
from nltk.corpus import gutenberg
import pandas as pd

# Load the Gutenberg corpus
corpus = gutenberg.raw('shakespeare-hamlet.txt')
# Save the file
with open('shakespeare-hamlet.txt', 'w') as file:
    file.write(corpus)


import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

with open('shakespeare-hamlet.txt', 'r') as file:
    text = file.read().lower()

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1 

# Create input sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)
# Pad sequences
max_sequence_length = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Create predictors and label
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = np.array(y)
# Convert labels to categorical
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=total_words)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# Build the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GRU
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=100, input_length=max_sequence_length - 1))
model.add(GRU(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
# Save the model
model.save('lstm_shakespeare_model.h5')
# Print the model summary
print(model.summary())

# predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_length):
    # Tokenize the input text
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
    
    # Predict the next word
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=-1)[0]
    
    # Convert index to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
        
input_text = "to be or not to be"
max_sequence_length = model.input_shape[1] + 1  # +1 for the label
predicted_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
print(f"The next word after '{input_text}' is '{predicted_word}'")

# Save the model and tokenizer
model.save('lstm_shakespeare_model.h5')
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

input_text = "Mar. Horatio saies, 'tis but our"
max_sequence_length = model.input_shape[1] + 1  # +1 for the label
predicted_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
print(f"The next word after '{input_text}' is '{predicted_word}'")
