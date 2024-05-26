import Neuron
import numpy as np
import Layer

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