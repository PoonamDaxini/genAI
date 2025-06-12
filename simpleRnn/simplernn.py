import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.models import Sequential

# Load the IMDB dataset
max_features = 10000  # Number of unique words to consider
maxlen = 500  # Maximum length of each review
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# Pad sequences to ensure uniform length
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


# print(x_train[0])

# Build the RNN model
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=128, input_length=maxlen))
model.add(SimpleRNN(128, return_sequences=False))  # Simple RNN layer
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# 
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
print(early_stopping)
tensorboard_callback = TensorBoard(log_dir='logs/rnn', histogram_freq=1)
# Train the model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, tensorboard_callback]
)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
# Save the model
model.save('rnn_imdb_model.h5')
# Print the model summary
print(model.summary())
