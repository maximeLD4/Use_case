import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
import warnings
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=DeprecationWarning)

# [[[<---past--->][<---future--->]]]
# [[[<--- Data (window size) --->]]]
n_past = 20
n_future = 5
n_features = 1

df = pd.read_csv('out/seq_data_0.csv')
seq = np.array(df["action_0"])
seq_first = seq[0:n_past]
df_shuffled = shuffle(df)

# X and y separation  # [[[<--- past (X) --->][<--- future (y) --->]]]
X = df_shuffled[[f'action_{i}' for i in range(n_past)]]
y = df_shuffled[[f'action_{i}' for i in range(n_past, n_past + n_future)]]
X = np.array(X)
y = np.array(y)

# split train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)  # random_state=42

# define models CNN
print("Model built")
model = Sequential()
model.add(tf.keras.Input(shape=(n_past, n_features)))
model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.6))
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.6))
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(tf.keras.layers.AveragePooling1D(pool_size=2))
model.add(LSTM(200, return_sequences=True, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.6))
model.add(LSTM(100, return_sequences=True, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.6))
model.add(LSTM(100, return_sequences=False, activation='relu'))
model.add(Dense(n_future))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# fit the model
epochs = 1000
history = model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_data=(X_val, y_val))  # batch_size=batch,
print(X_train[0])

# graph of the loss shows convergence
print("Model Loss compute")
plt.plot(history.history['loss'], label="train")
plt.plot(history.history['val_loss'], c='red', label="val")
plt.title('loss')
plt.xlabel('epochs')
plt.legend()
plt.show(block=False)

# Save the model
model.save('models/CNN_LSTM_Weights.keras')
"""
# test sequence
seq_pred = np.array(seq_first)
x_input = np.array([seq_pred])
print("seq_pre\n", seq_pred)
while len(seq_pred) < len(seq):
    yhat = model.predict(x_input, verbose=0)
    yhat_int = []
    for elemt in yhat[0]:
            elemt = int(elemt)
            if elemt < 0:
                yhat_int.append(0)
            elif elemt > 7:
                yhat_int.append(7)
            else:
                yhat_int.append(elemt)
    seq_pred = np.append(seq_pred, yhat_int)
    x_input = np.array([seq_pred[-n_past:]])

plt.figure(figsize=(20, 10))
plt.plot(seq, label="reel")
plt.plot(seq_pred, color="r", label="predict")
plt.legend()
plt.show(block=True)
"""
