from Neural_network import Neural_network
from Layer import Layer
import numpy as np


nw = Neural_network()
nw.add_layer(Layer(2, 8,"relu"))
nw.add_layer(Layer(1, 2,"relu"))
print(nw)

inputs = []
targets = []
for i in range(10000):
    input = np.random.rand(8)
    target = 0
    target = 0
    for j in range(8):
        if j % 2 == 0:
            target += input[j]
        else:
            target += input[j]*0.5
    inputs.append(input)
    targets.append([target])

inputs = np.array(inputs)
targets = np.array(targets)

test_input = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
true_result = 0
for j in range(8):
    if j % 2 == 0:
        true_result += test_input[j]
    else:
        true_result += test_input[j] *0.5


nw.train(inputs, targets, learning_rate=0.0000001, error_function="mean_squared_error", max_epochs=1)
one_predicted_result = nw.predict(test_input)
print("True result with one epoch: ", true_result, "Predicted result: ", one_predicted_result)

nw.train(inputs, targets, learning_rate=0.0000001, error_function="mean_squared_error", max_epochs=1000)
predicted_result = nw.predict(test_input)
print("True result: ", true_result, "Predicted result: ", predicted_result)