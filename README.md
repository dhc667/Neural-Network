# Readme

This is a simple python implementation of a neural network, in Network.py, located in `python_impl/`

The network is generic, one can specify:

```python
net = Network([10, 20, 30])
```

And that will create a network with 10 neurons in the first layer, 20 in the second, and 30 in the last layer.

We can then train it using stochastic gradient descent: For each epoch, the training data is split in mini-batches, a gradient of the network's weights and biases is calculated for the inputs and outputs of each mini-batch, the gradients are averaged out and the weights and biases are moved in the opposite direction, a distance equal to `eta`.

1. `data` is expected to be a list of tuples of ndarrays
2. `test` is expected to have the same type as `data`, it is optional: if supplied, accuracy will be measured after the application of each epoch

The first layer will be expected to have the same length as the first elements of the elements of the `data` and `test` parameters, and the last layer, the same as the second elements, i.e. the first layer is interpreted as the input layer and the last layer as the output layer.

It is recommended to normalize the data to avoid numerical errors.

```python
net.SGD_train(data=train, epochs=30, eta=2, mini_batch_size=50, test=test)
```

The default optimizer is `constant_eta`, which simply means that the average gradient calculated for each epoch will be normalized, multiplied by the original eta supplied in `SGD_Train` and added to the weights and biases on each epoch. Other optimizers were implemented, namely `nag` and `momentum`, however, after testing, the results with these were worse than using `constant_eta`

In `test.ipynb`, the results of testing some networks with the MNIST Handwritten numbers dataset are shown.
