from Neural_network import Neural_network
from Layer import Layer
import numpy as np
import pandas as pd


# Generate input-target pairs with a step of 0.1 between 0 and 10
inputs = np.arange(0, 10.1, 0.1).reshape(-1, 1)
targets = np.array([[2 * x] for x in inputs.reshape(-1)])


# Shuffle the data to ensure that the batches are not always the same samples
shuffled_indices = np.random.permutation(len(inputs))
inputs = inputs[shuffled_indices]
targets = targets[shuffled_indices]

# Normalize the data to have values between 1 and 10

print(inputs[0:10])
print(targets[0:10])

nn = Neural_network()
nn.add_layer(Layer(2, 1,activation_function="linear"))
nn.add_layer(Layer(1, 2, activation_function="linear"))

nn.set_early_stopping(True, 0.1)

nn.train(inputs, targets, learning_rate=0.0001 , error_function="mean_squared_error", max_epochs=100000,batch_size=10)

new_inputs = np.array([[6],[14],[16],[20],[10]])
predictions = nn.predict(new_inputs)
print(predictions)
