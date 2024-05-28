import Neuron
import numpy as np
import Layer
from Error_functions import Error_functions

class Neural_network:
    def __init__(self) -> None:
        self.layers = []
        self.inputs = []

    def add_layer(self, layer):
        if type(layer) != Layer.Layer:
            raise ValueError("Invalid layer")
        self.layers.append(layer)

    def get_outputs(self):
        return self.layers[-1].get_outputs()
    
    def set_first_inputs(self, inputs):
        self.inputs = inputs

    def calculate_outputs(self, inputs):
        #self.set_first_inputs(inputs)
        self.layers[0].calculate_outputs(self.inputs)
        for i in range(1, len(self.layers)):
            self.layers[i].calculate_outputs(self.layers[i-1].get_outputs())
        

    def calculate_errors(self, targets, error_function):
        if error_function == "mean_squared_error":
            return Error_functions.mean_squared_error(self.get_outputs(), targets)
        elif error_function == "cross_entropy_error":
            return Error_functions.cross_entropy_error(self.get_outputs(), targets)
        elif error_function == "mean_absolute_error":
            return Error_functions.mean_absolute_error(self.get_outputs(), targets)
        else:
            raise ValueError("Error function not supported")
        
    def gradient_descent(self, targets, learning_rate, error_function="mean_squared_error"):
        #calculate outputs
        self.calculate_outputs(targets)

        #calculate errors
        error = self.calculate_errors(targets,error_function)

        #retropropagation to obtain deltaW = -alpha * dE/dW
        for i in range(len(self.layers)-1, 0, -1):
            for j in range(len(self.layers[i].neurons)):
                neuron = self.layers[i].neurons[j]
                for k in range(len(neuron.weights)):
                    if i == len(self.layers)-1:
                        neuron.weights[k] -= learning_rate * (neuron.output - targets[j]) * neuron.inputs[k]
                    else:
                        neuron.weights[k] -= learning_rate * np.sum([self.layers[i+1].neurons[l].weights[j] * self.layers[i+1].neurons[l].output for l in range(len(self.layers[i+1].neurons))]) * neuron.inputs[k]

    def train(self, all_inputs, all_targets, learning_rate=0.1, error_function="mean_squared_error", max_epochs=1000, min_error=0.0001):
        for epoch in range(max_epochs):
            print("Epoch: ", epoch+1,"/",max_epochs)
            error = 0
            for i in range(len(all_inputs)):
                self.train_batch(all_inputs[i], all_targets[i], learning_rate, error_function)
                error += self.calculate_errors(all_targets[i], error_function)
            error /= len(all_inputs)
            print("Error: ", error)
            print(self)
            if error < min_error:
                break
    
    def train_batch(self, inputs, targets, learning_rate, error_function):
        self.set_first_inputs(inputs)
        self.gradient_descent(targets, learning_rate, error_function)
        
    def predict(self, inputs):
        self.calculate_outputs(inputs)
        return self.get_outputs()
    
    def save(self, filename):
        with open(filename, "w") as file:
            for layer in self.layers:
                for neuron in layer.neurons:
                    file.write(str(neuron.bias) + "\n")
                    for weight in neuron.weights:
                        file.write(str(weight) + "\n")
                    file.write(neuron.activation_function.__name__ + "\n")
                file.write("\n")

    def load(self, filename):
        with open(filename, "r") as file:
            lines = file.readlines()
            i = 0
            while i < len(lines):
                layer = Layer.Layer(len(lines[i].split()), len(lines[i+1].split()), lines[i+2].strip(), float(lines[i].strip()))
                for j in range(len(lines[i+1].split())):
                    layer.neurons[j].set_weights([float(x) for x in lines[i+1+j].split()])
                self.add_layer(layer)
                i += 3 + len(lines[i+1].split())

    def __str__(self) -> str:
        string = "Neural network\n"
        string += "Inputs: " + str(self.inputs) + "\n"
        for layer in self.layers:
            string += str(layer)
        return string