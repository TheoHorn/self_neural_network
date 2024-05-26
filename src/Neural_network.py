import Neuron
import numpy as np
import Layer

class Error_Functions:
    def mean_squared_error(outputs, targets):
        return np.mean(np.square(outputs - targets))
    
    def cross_entropy_error(outputs, targets):
        return -np.sum(targets * np.log(outputs))
    
    def mean_absolute_error(outputs, targets):
        return np.mean(np.abs(outputs - targets))

class Neural_network:
    def __init__(self) -> None:
        self.layers = []

    def add_layer(self, layer):
        if type(layer) != Layer.Layer:
            raise ValueError("Invalid layer")
        self.layers.append(layer)

    def get_outputs(self):
        return self.layers[-1].get_outputs()
    
    def set_inputs(self, inputs):
        self.layers[0].set_inputs(inputs)

    def calculate_outputs(self, inputs):
        self.set_inputs(inputs)
        for i in range(1, len(self.layers)):
            self.layers[i].calculate_outputs(self.layers[i-1].get_outputs())

    def calculate_errors(self, targets, error_function="mean_squared_error"):
        if error_function == "mean_squared_error":
            return Error_Functions.mean_squared_error(self.get_outputs(), targets)
        elif error_function == "cross_entropy_error":
            return Error_Functions.cross_entropy_error(self.get_outputs(), targets)
        elif error_function == "mean_absolute_error":
            return Error_Functions.mean_absolute_error(self.get_outputs(), targets)
        else:
            raise ValueError("Error function not supported")
        
    def backpropagation(self, targets):
        # TODO: Implement backpropagation