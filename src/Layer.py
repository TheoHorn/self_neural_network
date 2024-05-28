from Neuron import Neuron
import numpy as np

class Layer:
    def __init__(self, num_neurons, num_inputs, activation_function="identity", bias=0):
        self.neurons = [Neuron() for _ in range(num_neurons)]
        for neuron in self.neurons:
            neuron.set_num_inputs(num_inputs)
            neuron.set_weights(np.random.rand(num_inputs))
            neuron.set_activation_function(activation_function)
            neuron.set_bias(bias)

    def set_activation_function(self, activation_name):
        for neuron in self.neurons:
            neuron.set_activation_function(activation_name)

    def add_neuron(self, neuron):
        if type(neuron) != Neuron:
            raise ValueError("Invalid neuron")
        self.neurons.append(neuron)

    def get_outputs(self):
        return [neuron.get_output() for neuron in self.neurons]

    def set_inputs(self, inputs):
        for neuron in self.neurons:
            #print(self,inputs)
            neuron.set_inputs(inputs)

    def calculate_outputs(self, inputs):
        self.set_inputs(inputs)
        for neuron in self.neurons:
            neuron.calculate_output()
        return self.get_outputs() 

    def __str__(self):
        string = "Layer\n"
        for neuron in self.neurons:
            string += str(neuron)
        return string
