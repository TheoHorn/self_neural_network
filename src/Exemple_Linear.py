import numpy as np
from Neural_network import Neural_network
from Layer import Layer

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

nn = Neural_network()
nn.add_layer(Layer(2, 1,activation_function="linear"))
nn.add_layer(Layer(1, 2, activation_function="linear"))

nn.train(inputs, targets, learning_rate=0.0001 , error_function="mean_squared_error", max_epochs=10000)

new_inputs = np.array([
    [6],
    [14],
    [16],
    [20],
    [10]
])
predictions = nn.predict(new_inputs)
print(predictions)
