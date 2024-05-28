import numpy as np

class Activation_functions:    
    def linear(x):
        return x

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def relu(x):
        return np.maximum(0, x)

    def tanh(x):
        return np.tanh(x)

    def softmax(x):
        # For numerical stability, subtract the maximum value from x
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0)

    def derivative_linear(x):
        return 1

    def derivative_sigmoid(x):
        # Use the sigmoid function to compute the derivative
        sigmoid_x = Activation_functions.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def derivative_relu(x):
        return np.where(x > 0, 1, 0)

    def derivative_tanh(x):
        # Use the tanh function to compute the derivative
        tanh_x = Activation_functions.tanh(x)
        return 1 - tanh_x ** 2

    def derivative_softmax(x):
        # The derivative of the softmax function is more complex and is
        # computed using the following formula:
        # softmax(x) * (1 - softmax(x)) for i = j
        # -softmax(x) * softmax(x) for i != j
        softmax_x = Activation_functions.softmax(x)
        n = softmax_x.shape[0]
        result = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    result[i, j] = softmax_x[i] * (1 - softmax_x[i])
                else:
                    result[i, j] = -softmax_x[i] * softmax_x[j]
        return result

    def get_value_from_name(name):
        if name == "linear":
            return Activation_functions.linear
        elif name == "sigmoid":
            return Activation_functions.sigmoid
        elif name == "relu":
            return Activation_functions.relu
        elif name == "tanh":
            return Activation_functions.tanh
        elif name == "softmax":
            return Activation_functions.softmax
        else:
            raise ValueError("Activation function not supported")
        
    def get_derivative_value_from_name(name):
        if name == "linear":
            return Activation_functions.derivative_linear
        elif name == "sigmoid":
            return Activation_functions.derivative_sigmoid
        elif name == "relu":
            return Activation_functions.derivative_relu
        elif name == "tanh":
            return Activation_functions.derivative_tanh
        elif name == "softmax":
            return Activation_functions.derivative_softmax
        else:
            raise ValueError("Activation function not supported")