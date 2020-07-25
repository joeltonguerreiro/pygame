import numpy as np


def sigmoid(x, derivative=False):
    return x * (1.0 - x) if derivative else 1.0 / (1.0 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, bias=1):
        self.bias = bias

    # calcular a ação a ser executada seguindo a maior probabilidade
    def feed_forward(self, weights, input_values):
        input_array = np.array(input_values, ndmin=2)

        w1 = weights[0]
        w2 = weights[1]
        w3 = weights[2]

        layer1 = sigmoid(np.dot(input_array, w1) + self.bias)

        layer2 = sigmoid(np.dot(layer1, w2) + self.bias)

        output = sigmoid(np.dot(layer2, w3) + self.bias)

        return output
