import math
import numpy as np

class Neural:

    def activation(self, inputs, weights):
        """
        This is the activation fun for the bits neural network.
        It will perform the multiplication operation of inputs and weights and
        will return the floor value of those operations in an array format.
        :param inputs: list
        Input to the Neural Network
        :param weights: list
        Current Weight of the Neural Network
        :return: list
        Activation function values
        """
        return list(map(lambda num : math.floor(num), np.dot(inputs, weights)))


    def train(self, inputs, targets, eta, n_iterations):
        """
        This function will train the Neural Network on the given inputs
        :param inputs: list,
        Input list to the Neural Network
        :param targets: list,
        Dependent Values of the Input list
        :param eta: int,
        Learnable parameter
        :param n_iterations: int,
        Number of iteration needed to loop
        :return: array,
        Calculated weights of the Neural Network
        """

        # Add the bios node to the input if any of inputs are zero
        inputs = np.c_[inputs, -np.ones(len(inputs))]

        # Initalizing the predefined random weights
        weights = np.random.randn(len(inputs[0])) * 1e-4

        for n in range(n_iterations):
            print('-------------------Iteration {}---------------'.format(n))
            print('Weight\n{}'.format(weights))

            # Finding out the activation value
            activation_value = self.activation(inputs, weights)

            weights = weights - eta * np.dot(np.transpose(inputs), activation_value - targets)

            print('Activation Value\n{}'.format(activation_value))
            print('Weight\n{}'.format(weights))
            print('-------------------------------------------------')

        return weights

