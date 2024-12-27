from random import normalvariate
import numpy as np
from enum import Enum
from non_linearity_picker import *
import random

MOMENTUM_ALPHA = .9

class Optimizers(Enum):
    nag = 'nag'
    momentum = 'momentum'
    constant_eta = 'constant_eta'


class Network:
    def __init__(self, layers: list[int], variance = 1, non_linearity_function: Non_Linearity_Functions = Non_Linearity_Functions.sigmoid) -> None:
        self.layers = layers
        self.non_linearity_function = non_linearity_function
        self.weights = [
            np.array([
                [
                    normalvariate(0, variance) for j in range(inputs)
                ] for i in range(nodes)
            ]) for inputs, nodes in zip(layers[:-1], layers[1:])
        ]

        self.biases = [
            np.array([
                normalvariate(0, variance) for i in range(nodes)
            ]) for nodes in layers[1:]
        ]
    
    def backprop(self, inputs: np.ndarray, outputs: np.ndarray):
        nabla_b = [np.zeros(nodes) for nodes in self.layers[1:]]
        nabla_w = [np.zeros((nodes, nodes_left)) for nodes_left, nodes in zip(self.layers[1:], self.layers[:-1])]
        
        ## forward pass
        activations = [inputs]
        zs = []

        for b, w in zip(self.biases, self.weights):
            zs += [np.matmul(w, activations[-1]) + b]
            activations += [non_linearity(zs[-1], self.non_linearity_function)]
        
        ## backward pass
        delta = 2*(activations[-1] - outputs) * non_linearity_derivative(zs[-1], self.non_linearity_function)
        delta = delta.reshape((len(delta), 1))
        nabla_b[-1] = delta
        nabla_w[-1] = np.matmul(delta, activations[-2].reshape((1, len(activations[-2]))))
        
        for i in range(2, len(self.layers)):
            z = zs[-i]
            nlf_der = non_linearity_derivative(z, self.non_linearity_function)
            nlf_der = nlf_der.reshape((len(nlf_der), 1))
            delta = np.matmul(self.weights[-i + 1].transpose(), delta) * nlf_der
        
            nabla_b[-i] = delta
            nabla_w[-i] = np.matmul(delta, activations[-i-1].reshape((1, len(activations[-i-1]))))

        return (nabla_b, nabla_w)

    def gradient_mini_batch(self, batch: list[(np.ndarray, np.ndarray)]):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # print([nb.shape for nb in nabla_b])
        # print([nw.shape for nw in nabla_w])
        # print('-----------------')
        for input, output in batch:
            nabla_b_i, nabla_w_i = self.backprop(input, output)

            for i, pair in enumerate(zip(nabla_b_i, nabla_w_i)):
                bi, wi = pair

                # print(f"{nabla_b[i].shape} + {bi.shape}")
                # print(f"{nabla_w[i].shape} + {wi.shape}")

        
                nabla_b[i] = nabla_b[i] + bi.reshape(len(bi))
                nabla_w[i] = nabla_w[i] + wi

                # print()

        # print('-----------------')
        
        # print([nb.shape for nb in nabla_b])
        # print([nw.shape for nw in nabla_w])


        n = len(batch)
        for i in range(len(self.layers) - 1):
            nabla_b[i]/=n
            nabla_w[i]/=n

        return (nabla_b, nabla_w)

    def constant_eta_epoch_apply(self, mini_batch: list[(np.ndarray, np.ndarray)], eta: float):
        nabla_b, nabla_w = self.gradient_mini_batch(mini_batch)
        for i in range(len(self.layers) - 1):
            self.weights[i] = self.weights[i] - nabla_w[i]*eta
            self.biases[i] = self.biases[i] - nabla_b[i]*eta

    def momentum_epoch_apply(self, mini_batch: list[(np.ndarray, np.ndarray)], momentum_b: list[np.ndarray], momentum_w: list[np.ndarray], eta: float):
        nabla_b, nabla_w = self.gradient_mini_batch(mini_batch)
        for i in range(len(self.layers) - 1):
            nabla_w[i] = -nabla_w[i]*eta + momentum_w[i]*MOMENTUM_ALPHA
            nabla_b[i] = -nabla_b[i]*eta + momentum_b[i]*MOMENTUM_ALPHA

            self.weights[i] = self.weights[i] + nabla_w[i]
            self.biases[i] = self.biases[i] + nabla_b[i]

        return (nabla_b, nabla_w)

    def nag_epoch_apply(self, mini_batch: list[(np.ndarray, np.ndarray)], momentum_b: list[np.ndarray], momentum_w: list[np.ndarray], eta: float):
        for i in range(len(self.layers) - 1):
            self.weights[i] = self.weights[i] + momentum_w[i]*MOMENTUM_ALPHA
            self.biases[i] = self.biases[i] + momentum_b[i]*MOMENTUM_ALPHA

        nabla_b, nabla_w = self.gradient_mini_batch(mini_batch)

        for i in range(len(self.layers) - 1):
            self.weights[i] = self.weights[i] - nabla_w[i]*eta
            self.biases[i] = self.biases[i] - nabla_b[i]*eta

            nabla_w[i] = momentum_w[i]*MOMENTUM_ALPHA - nabla_w[i]*eta
            nabla_b[i] = momentum_b[i]*MOMENTUM_ALPHA - nabla_b[i]*eta

        return (nabla_b, nabla_w)

    def SGD_train(self, data: list[(np.ndarray, np.ndarray)], epochs: int, eta: float = 0.01, mini_batch_size: int = 32, sgd_sample_replacement: bool = False, optimizer: Optimizers = Optimizers.nag):
        def get_mini_batches():
            if sgd_sample_replacement:
                mini_batches = [
                    random.choices(data, k=mini_batch_size) for i in range(int(np.ceil(len(data)/mini_batch_size)))
                ]
            else:
                random.shuffle(data)
                mini_batches = [
                    data[k: k + mini_batch_size] for k in range(0, len(data), mini_batch_size)
                ]

            return mini_batches

        if optimizer == Optimizers.constant_eta:
            for i in range(epochs):
                mini_batches = get_mini_batches()
                for batch in mini_batches:
                    self.constant_eta_epoch_apply(batch, eta)

                print(f"Epoch {i + 1} Completed")

        elif optimizer == Optimizers.momentum:
            momentum_b = [np.zeros(b.shape) for b in self.biases]
            momentum_w = [np.zeros(w.shape) for w in self.weights]

            for i in range(epochs):
                mini_batches = get_mini_batches()
                for batch in mini_batches:
                    momentum_b, momentum_w = self.momentum_epoch_apply(batch, momentum_b, momentum_w, eta)

                print(f"Epoch {i + 1} Completed")

        elif optimizer == Optimizers.nag:
            momentum_b = [np.zeros(b.shape) for b in self.biases]
            momentum_w = [np.zeros(w.shape) for w in self.weights]

            for i in range(epochs):
        
                mini_batches = get_mini_batches()
                for batch in mini_batches:
                    momentum_b, momentum_w = self.nag_epoch_apply(batch, momentum_b, momentum_w, eta)

                print(f"Epoch {i + 1} Completed")

        else:
            raise ValueError("Invalid Optimizer")
        
    def eval(self, inputs: np.ndarray):
        activation = inputs

        for b, w in zip(self.biases, self.weights):
            z = np.matmul(w, activation) + b
            activation = non_linearity(z, self.non_linearity_function)

        return activation
                

if __name__ == "__main__":
    import json

    net = Network([28*28, 30, 30, 10])
    with open('./data/mnist_handwritten_test.json', 'r') as f:
        data = json.loads(f.read())

    print(len(data[0]['image']) == 28*28)
    print(data[0].keys())

    train = []

    for pair in data:
        train += [(np.array(pair['image']), np.array([(1 if i == pair['label'] else 0) for i in range(10)]))]

    net.SGD_train(train, 30, .01)
