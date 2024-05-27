import numpy as np

class Activation_functions:
    def identity(x):
        return x

    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    
    def relu(x):
        return max(0, x)
    
    def tanh(x):
        return np.tanh(x)
    
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def relu(x):
        if x > 0:
            return x
        else:
            return 0
        
    def tanh(x):
        return np.tanh(x)
    
    def derivate_identity(x):
        return 1
    
    def derivate_sigmoid(x):
        return x * (1 - x)
    
    def derivate_relu(x):
        if x > 0:
            return 1
        else:
            return 0
    
    def derivate_tanh(x):
        return 1 - x**2
    
    def derivate_softmax(x):
        return x * (1 - x)    