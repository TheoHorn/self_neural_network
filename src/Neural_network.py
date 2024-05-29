import numpy as np
import Layer
from Error_functions import Error_functions

class Neural_network:
    """
    This class represents a neural network that can be used for classification and regression tasks.
    """

    def __init__(self) -> None:
        self.layers = []
        self.inputs = []

    def add_layer(self, layer):
        if type(layer) != Layer.Layer:
            raise ValueError("Invalid layer")
        self.layers.append(layer)

    def get_outputs(self):
        return self.layers[-1].get_outputs()

    def set_inputs(self, inputs):
        self.inputs = inputs

    def calculate_outputs(self, inputs=None):
        if inputs is not None:
            self.set_inputs(inputs)
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
    
    def train(self, all_inputs, all_targets, learning_rate=0.00001, error_function="mean_squared_error", max_epochs=1000, min_error=0.0001):
        for epoch in range(max_epochs):
            print("Epoch: ", epoch+1,"/",max_epochs)
            error = 0
            for i in range(len(all_inputs)):
                self.gradient_descent(all_inputs[i], all_targets[i], learning_rate, error_function)
                error += self.calculate_errors(all_targets[i], error_function)
            error /= len(all_inputs)
            print("Error: ", error)
            if error < min_error:
                break
    
    def gradient_descent(self, inputs, targets, learning_rate, error_function="mean_squared_error"):
        # Forward pass
        self.calculate_outputs(inputs)

        # Backward pass
        output_errors = Error_functions.get_derivate_value_from_name(self.get_outputs(), targets, error_function)

        for i in range(len(self.layers) - 1, 0, -1):
            layer_errors = np.dot(self.layers[i].neurons[0].weights[1:], output_errors)
            for j in range(len(self.layers[i].neurons)):
                neuron = self.layers[i].neurons[j]
                delta_weights = np.zeros(len(neuron.weights))
                delta_bias = 0

                for k in range(len(neuron.weights)):
                    if k == 0:
                        delta_bias = learning_rate * layer_errors
                        delta_weights[k] = 0  # No need to update bias in weights array
                    else:
                        delta_weights[k] = learning_rate * layer_errors * neuron.inputs[k - 1] * neuron.activation_function_derivative(neuron.get_output())

                neuron.weights -= delta_weights
                neuron.bias -= delta_bias

            output_errors = layer_errors * neuron.activation_function_derivative(neuron.get_output())

        # Update output layer biases and weights
        for j in range(len(self.layers[-1].neurons)):
            neuron = self.layers[-1].neurons[j]
            delta_weights = np.zeros(len(neuron.weights))
            delta_bias = 0

            for k in range(len(neuron.weights)):
                if k == 0:
                    delta_bias = learning_rate * output_errors
                    delta_weights[k] = 0  # No need to update bias in weights array
                else:
                    delta_weights[k] = learning_rate * output_errors * neuron.inputs[k - 1] * neuron.activation_function_derivative(neuron.get_output())

            neuron.weights -= delta_weights
            neuron.bias -= delta_bias


    def predict(self, all_inputs):
        outputs = []
        for i in range(len(all_inputs)):
            self.calculate_outputs(all_inputs[i])
            outputs.append(self.get_outputs())
        return outputs

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
                layer = Layer.Layer(len(lines[i].split()), len(lines[i+1].split()), lines[i+2].strip().replace('\n', ''))
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