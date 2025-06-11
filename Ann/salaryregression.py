import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime


data = pd.read_csv('Churn_Modelling.csv')
# Select relevant features
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
# Encode categorical features
regression_geo_encoder = OneHotEncoder()
regression_geo_encoded = regression_geo_encoder.fit_transform(data[['Geography']]).toarray()
regression_geo_encoded_df = pd.DataFrame(regression_geo_encoded, columns=regression_geo_encoder.get_feature_names_out(['Geography']))


gender_encoder = LabelEncoder()
data['Gender'] = gender_encoder.fit_transform(data['Gender'])

# Concatenate the encoded geography with the original data
data = pd.concat([data.reset_index(drop=True), regression_geo_encoded_df], axis=1)
# Drop the original 'Geography' column
data = data.drop(['Geography'], axis=1)

# Split the data into features and target variable
X = data.drop(['EstimatedSalary'], axis=1)
y = data['EstimatedSalary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# load the encoder and scaler
with open('geo_encoder.pkl', 'rb') as f:
    geo_encoder = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

# load pickle file
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Build the ANN model Regression
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Output layer for regression
])
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# print(model.summary())

# set up tensorboard
log_dir = "regressionlogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

earlyStopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[tensorboard_callback, earlyStopping_callback]
)

# tensorboard --logdir regressionlogs/fit  ==> view tesnsor board graph

# Evaluate the model
loss = model.evaluate(X_test_scaled, y_test)

# save the model
model.save('ann_regression_model.h5')