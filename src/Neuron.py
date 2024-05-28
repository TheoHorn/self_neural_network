from Activation_functions import Activation_functions

class Neuron:
    def __init__(self):
        self.num_inputs = 0
        self.inputs = []
        self.weights = []
        self.bias = 0
        self.output = 0
        self.activation_function = None
        self.derivative_activation_function = None

    def set_activation_function(self, activation_name):
        self.activation_function = Activation_functions.get_value_from_name(activation_name)
        self.activation_function_derivative = Activation_functions.get_derivative_value_from_name(activation_name)

    def set_num_inputs(self, num_inputs):
        self.num_inputs = num_inputs

    def set_inputs(self, inputs):
        if len(inputs) != self.num_inputs:
            raise ValueError("Invalid number of inputs")
        self.inputs = inputs[:]  # make a copy of the inputs list

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

    def __str__(self):
        string = "Neuron - " + str(self.num_inputs) + " inputs\n"
        for i in range(len(self.inputs)):
            string += "    in: " + str(self.inputs[i]) + ",w: " + str(self.weights[i]) + "\n"
        string += "bias: " + str(self.bias) + "\n"
        string += "output: " + str(self.output) + "\n"
        return string
