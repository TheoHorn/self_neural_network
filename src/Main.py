from Neural_network import Neural_network
from Layer import Layer


nw = Neural_network()
nw.add_layer(Layer(4, 8,"relu"))
nw.add_layer(Layer(2, 4,"relu"))
nw.add_layer(Layer(1, 2,"softmax"))

nw.calculate_outputs([1, 2, 3, 4, 5, 6, 7, 8])
print(nw.get_outputs())