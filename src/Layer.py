import Neuron

class Layer:
    def __init__(self, num_neurons, inputs) -> None:
        self.neurons = [Neuron() for _ in range(num_neurons)]
        for neuron in self.neurons:
            neuron.set_inputs(inputs)

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
            neuron.set_inputs(inputs)

    def calculate_outputs(self, inputs):
        self.set_inputs(inputs)
        for neuron in self.neurons:
            neuron.calculate_output()

    