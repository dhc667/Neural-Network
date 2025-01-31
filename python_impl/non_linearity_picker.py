from sigmoid import sigmoid, sigmoid_prime
from relu import relu, relu_prime
from enum import Enum
import numpy as np

class Non_Linearity_Functions(Enum):
    sigmoid = 'sigmoid'
    relu = 'relu'

def non_linearity(x: np.ndarray, non_linearity_function: Non_Linearity_Functions):
    if non_linearity_function == Non_Linearity_Functions.sigmoid:
        return sigmoid(x)
    elif non_linearity_function == Non_Linearity_Functions.relu:
        return relu(x)
    else:
        raise TypeError('Invalid non linearity function')

def non_linearity_derivative(x: np.ndarray, non_linearity_function: Non_Linearity_Functions):
    if non_linearity_function == Non_Linearity_Functions.sigmoid:
        return sigmoid_prime(x)
    elif non_linearity_function == Non_Linearity_Functions.relu:
        return relu_prime(x)
    else:
        raise TypeError('Invalid non linearity function')