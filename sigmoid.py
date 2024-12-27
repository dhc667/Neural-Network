import numpy as np

def sigmoid(x: np.ndarray):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x: np.ndarray):
    return sigmoid(x)*(1 - sigmoid(x))