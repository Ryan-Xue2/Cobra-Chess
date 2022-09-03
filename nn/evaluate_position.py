import numpy as np


class Layer:
    def __init__(self, num_neurons_in, num_neurons_out):
        self.weights = np.random.rand(num_neurons_out, num_neurons_in)
        self.bias = 0   


class NN:
    def __init__(self):
        """Initialize the layers of the NN"""
        self.num_input_neurons = 64
        self.num_output_neurons = 1
        self.num_hidden_neurons = 40

        self.layers = []
        self.hidden_layer = Layer(self.num_input_neurons, self.num_hidden_neurons)
        self.output_layer = Layer(self.num_hidden_neurons, self.num_output_neurons)
        self.layers.append(self.hidden_layer)
        self.layers.append(self.output_layer)

        self.outputs = []
        self.errors = []

    def forward_propagation(self, inputs):
        """Propagate forward in the network, and do stuff"""
        for layer in self.layers:
            new_inputs = []
            for weights in layer.weights:
                activation = self._activation(layer.bias, weights, inputs)
                output = self._transfer(activation)
                new_inputs.append(output)
            inputs = new_inputs
            self.outputs.append(inputs)
        return inputs

    def backward_propagation(self, expected):
        """Propagate backwards and do stuff"""
        # output layer find error
        error = (self.outputs[1][0] - expected) * self._transfer_derivative(self.outputs[1][0])
        print(self.outputs[1][0])
        print(error)

        

    def _activation(self, bias, weights, inputs):
        """Given the bias, weights and inputs to a neuron, calculate the activation"""
        activation = bias
        assert len(weights) == len(inputs)
        for i in range(len(weights)):
            activation += weights[i] * inputs[i]
        return activation

    def _transfer(self, activation):
        """ReLu function"""
        return max(0, activation)

    def _transfer_derivative(self, x):
        """Derivative for ReLu function"""
        if x > 0:
            return 1
        else:
            return 0

    def train(self):
        """Train the neural network on chess positions"""
        pass

    def test(self):
        """Test the neural network for loss and accuracy"""
        pass
    
mynn = NN()
mynn.forward_propagation([1] * 64)
mynn.backward_propagation(1000)
print(len(mynn.outputs[0]))