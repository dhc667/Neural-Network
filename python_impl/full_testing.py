from Network import Network, Optimizers ,Non_Linearity_Functions
import json
import numpy as np


with open('./data/mnist_handwritten_train.json', 'r') as f:
    data = json.loads(f.read())

train = []

for pair in data:
    train += [(np.array(pair['image'])/255, np.array([(1 if i == pair['label'] else 0) for i in range(10)]))]
    
with open('./data/mnist_handwritten_test.json', 'r') as f:
    data = json.loads(f.read())

test = []

for pair in data:
    test += [(np.array(pair['image'])/255, np.array([(1 if i == pair['label'] else 0) for i in range(10)]))]


etas = [3, 1, 0.1, 0.01]
optimizers = [Optimizers.constant_eta, Optimizers.momentum, Optimizers.nag]
epochs = [10, 30, 50, 100]
mini_batch_size = [1, 10, 32, 50]
replacement = [True, False]
nlf = [Non_Linearity_Functions.sigmoid, Non_Linearity_Functions.relu]
layers = [[28*28, 30, 10], [28*28, 30, 30, 10], [28*28, 100, 10], [28*28, 100, 30, 10], [28*28, 30, 30, 30, 10], [28*28, 50, 50, 10]]

class Parameters:
    def __init__(self, layers, eta, optimizer, epoch, mini_batch_size, replacement, nlf):
        self.eta = eta
        self.optimizer = optimizer
        self.epochs = epoch
        self.mini_batch_size = mini_batch_size
        self.replacement = replacement
        self.layers = layers
        self.nlf = nlf


class Result:
    def __init__(self, parameters, accuracy, network):
        self.parameters = parameters
        self.accuracy = accuracy
        self.network = network 

    def serialize(self):
        return {
            'parameters': {
                'eta': self.parameters.eta,
                'optimizer': self.parameters.optimizer.name,
                'epochs': self.parameters.epochs,
                'mini_batch_size': self.parameters.mini_batch_size,
                'replacement': self.parameters.replacement,
                'layers': self.parameters.layers,
                'nlf': self.parameters.nlf.name
            },
            'accuracy': self.accuracy,
            'network': self.network.serialize()
        }

    def deserialize(json):
        parameters = Parameters(json['parameters']['layers'], json['parameters']['eta'], json['parameters']['optimizer'], json['parameters']['epochs'], json['parameters']['mini_batch_size'], json['parameters']['replacement'], json['parameters']['nlf'])
        network = Network.deserialize(json['network'])
        return Result(parameters, json['accuracy'], network)

def run(parameters: Parameters):
    network = Network(parameters.layers, 1, parameters.nlf)
    network.SGD_train(train, parameters.epochs, parameters.eta, parameters.mini_batch_size, parameters.replacement, parameters.optimizer)
    return Result(parameters, network.evaluate(test), network)

results = []
for eta in etas:
    for optimizer in optimizers:
        for epoch in epochs:
            for mini_batch in mini_batch_size:
                for replace in replacement:
                    for nlf in nlf:
                        for layer in layers:
                            parameters = Parameters(layer, eta, optimizer, epoch, mini_batch, replace, nlf)
                            results += [run(parameters)]

with open('./data/results.json', 'w') as f:
    f.write(json.dumps([result.serialize() for result in results]))

