from Neural_network import Neural_network
from Layer import Layer
import numpy as np


nw = Neural_network()
nw.add_layer(Layer(2, 8,"relu"))
nw.add_layer(Layer(1, 2,"softmax"))
print(nw)

inputs = []
targets = []
for i in range(100):
    input = np.random.rand(8)
    target = 0
    for j in range(8):
        target += input[j]**j
    inputs.append(input)
    targets.append([target])

inputs = np.array(inputs)
targets = np.array(targets)

nw.train(inputs, targets, learning_rate=0.1, error_function="mean_squared_error", max_epochs=10)

test_input = [2, 3, 4, 5, 6, 7, 8, 9]
true_result = 0
for i in range(8):
    true_result += test_input[i]**i

predicted_result = nw.predict(test_input)
print("True result: ", true_result, "Predicted result: ", predicted_result)