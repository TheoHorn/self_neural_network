from Neural_network import Neural_network
from Layer import Layer
import numpy as np
import pandas as pd

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

nn = Neural_network()
nn.add_layer(Layer(2, 2))
nn.add_layer(Layer(1, 2, activation_function="sigmoid"))

nn.train(inputs, targets, learning_rate=0.1, error_function="mean_squared_error", max_epochs=10000)

new_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = nn.predict(new_inputs)
print(predictions)
