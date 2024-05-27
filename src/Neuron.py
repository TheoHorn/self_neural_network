import numpy as np
from Activation_functions import Activation_functions

class Neuron :
    def __init__(self) -> None:
        self.inputs = []
        self.weights = []
        self.bias = 0
        self.output = 0
        self.activation_function = None
    
    def set_activation_function(self, activation_name):
        if activation_name == "identity":
            self.activation_function = Activation_functions.identity
        elif activation_name == "sigmoid":
            self.activation_function = Activation_functions.sigmoid
        elif activation_name == "relu":
            self.activation_function = Activation_functions.relu
        elif activation_name == "tanh":
            self.activation_function = Activation_functions.tanh
        elif activation_name == "softmax":
            self.activation_function = Activation_functions.softmax
        else:
            raise ValueError("Activation function not supported")
        
    
    def set_inputs(self, inputs):
        self.inputs = inputs
    
    def set_weights(self, weights):
        self.weights = weights
    
    def set_bias(self, bias):
        self.bias = bias
    
    def get_output(self):
        return self.output
    
    def calculate_output(self):
        if len(self.inputs) == 0 or len(self.weights) == 0:
            raise ValueError("Inputs or weights not set")
        
        if len(self.inputs) != len(self.weights):
            raise ValueError("Number of inputs and weights must be the same")
        
        self.output = self.bias
        for i in range(len(self.inputs)):
            self.output += self.inputs[i] * self.weights[i]
        
        self.output = self.activation_function(self.output)
        return self.output
    
    