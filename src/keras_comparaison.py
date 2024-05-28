from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd


inputs = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

targets = np.array([
    [2],
    [4],
    [6],
    [8],
    [10]
])


model = Sequential()
model.add(Dense(2, input_dim=1, activation='linear'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(inputs, targets, epochs=1000)

new_inputs = np.array([
    [6],
    [14],
    [16],
    [20],
    [10]
])
predictions = model.predict(new_inputs)
print(predictions)