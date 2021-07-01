""" simple_dnn.py - This will be a first stab at training a DNN on top
tagging dataset. DNN will be implemented using pytorch.

Author: Kevin Greif
Last updated 6/21/21
python3
"""

import torch


class simpleDNN(torch.nn.Module):
    """ simpleDNN - This class implements a simple dense neural network to be
    trained on the top tagging dataset.
    """

    def __init__(self, input_shape):
        """ __init__ - The initialization function for our neural network.
        Function will initialize weights and biases and create layers.

        Arguments:
            input_shape (tuple): a tuple which gives dimensions of input
            dataset (events, constituents)

        Returns:
            None
        """

        # Explicitly call init function of torch.nn for readability
        super(simpleDNN, self).__init__()

        # Set input shape to member instance
        self.input_shape = input_shape

        # Define network
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(self.input_shape, 100),
            torch.nn.BatchNorm1d(num_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.BatchNorm1d(num_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 2),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        """ forward - The forward pass for our network. Just consists of
        passing minibatch through the stack.

        Arguments:
            x (array): Minibatch data, of size (events, constituents)

        Returns:
            (array): The score for each event, of size (events)
        """
        return self.stack(x)


if __name__ == '__main__':

    # Initialize the simple model and print it
    input_shape = (320)
    my_model = simpleDNN(input_shape)
    print(my_model)
