import numpy as np

class Activation_functions:
    def identity(x):
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

    def derivate_identity(x):
        return 1

    def derivate_sigmoid(x):
        # Use the sigmoid function to compute the derivative
        sigmoid_x = Activation_functions.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def derivate_relu(x):
        return np.where(x > 0, 1, 0)

    def derivate_tanh(x):
        # Use the tanh function to compute the derivative
        tanh_x = Activation_functions.tanh(x)
        return 1 - tanh_x ** 2

    def derivate_softmax(x):
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
