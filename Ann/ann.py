import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping as EasyStopping
from tensorflow.keras.callbacks import TensorBoard
import datetime
import experiment


# build our ANN model
model=Sequential([
    Dense(64, activation='relu', input_shape=(experiment.x_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])


# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# set up the tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# set up early stopping
early_stopping = EasyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(experiment.x_train, experiment.y_train,
                    validation_data=(experiment.x_test, experiment.y_test),
                    callbacks=[tensorboard_callback, early_stopping], epochs=50, batch_size=32)
print(history)

model.save('ann_model.h5')