import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier, KerasRegressor

data = pd.read_csv('Churn_Modelling.csv')
# Select relevant features
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
# Encode categorical features
geo_encoder = OneHotEncoder()
geo_encoded = geo_encoder.fit_transform(data[['Geography']]).toarray()

label_gender_encoder = LabelEncoder()
data['Gender'] = label_gender_encoder.fit_transform(data['Gender'])

geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))
# Concatenate the encoded geography with the original data
data = pd.concat([data.reset_index(drop=True), geo_encoded_df], axis=1)
# Drop the original 'Geography' column
data = data.drop(['Geography'], axis=1)
# Split the data into features and target variable
X = data.drop(['Exited'], axis=1)
y = data['Exited']  # Assuming 'Exited' is the target variable for classification
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale the features
scaler = StandardScaler()
# scaler = scaler.fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Save the encoders and scaler
with open('geo_encoder.pkl', 'wb') as f:
    pickle.dump(geo_encoder, f)

with open('label_gender_encoder.pkl', 'wb') as f:
    pickle.dump(label_gender_encoder, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Define the ANN model with different parameters
def create_model(optimizer='adam', activation='relu', neurons=64):
    model = Sequential([
        Dense(neurons, activation=activation, input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation=activation),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create a KerasClassifier for hyperparameter tuning
model = KerasClassifier(model=create_model, verbose=0, epochs=50, batch_size=32, activation='relu', neurons=64)
# Define the hyperparameter grid
param_grid = {
    'optimizer': ['adam', 'sgd'],
    'activation': ['relu', 'tanh'],
    'neurons': [32, 64, 128],
}

# Create a GridSearchCV object
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
# Fit the model
grid_result = grid.fit(X_train_scaled, y_train)
# Print the best parameters and score
print("Best Parameters:", grid_result.best_params_)
print("Best Score:", grid_result.best_score_)