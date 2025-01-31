import numpy as np

def relu(x: np.ndarray):
    return x * (x > 0)


def relu_prime(x: np.ndarray):
    return 1 * (x > 0)

if __name__ == "__main__":
    x = np.array([-2, 2])
    print(relu(x))
    print(relu_prime(x))