""" classifier_trainer.py - This program will implement a class which handles
all of the work of training a classifier on the toptag dataset, assuming
for now that the model is written in pytorch.

Author: Kevin Greif
6/23/21
python3
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import data_dumper


class ClassifierTrainer():
    """ ClassifierTrainer - This class will handle the training and validation
    of a classifier written in pytorch, and also provide methods for saving
    and loading models. Meant for use on the top tagging dataset.
    """

    def __init__(self, model, optimizer, loss_function):
        """ __init__ - Init function for this class. Takes in the 3 things
        needed for model training: the model, optimizer, and loss function.
        Data will be supplied by the load_data method.

        Arguments:
            model (class) - The model to train
            optimizer (class) - The optimizer to use in training. Should be
            initialized with the model's parameters!
            loss_function (class) - The loss function to use

        Returns:
            None
        """

        # Set inputs to instance variables
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

    def load_data(self, torch_dl, flag=1):
        """ load_data - Load data into TrainClassifier class for use in
        training, validation, and testing. Expect data to be already in the
        form of a pytorch dataloader.

        Arguments:
            torch_dl (class): The pytorch dataloader containing data and with
            batch size defined.
            flag (int): Flag to control the type of data.
                1 = training
                2 = validation
                3 = testing

        Returns:
            None
        """

        # Set dataloader to appropriate value based on flag
        if flag == 1:
            self.train_data = torch_dl
            self.tr_batch_per_epoch = len(torch_dl)
        elif flag == 2:
            self.valid_data = torch_dl
            self.val_batch_per_epoch = len(torch_dl)
        elif flag == 3:
            self.test_data = torch_dl
            self.test_batch_per_epoch = len(torch_dl)
        else:
            raise ValueError("Flag must be either 1, 2, or 3!")

    def save_model(self, filePath):
        """ save_model - Save the model parameters at the given file path.

        Arguments:
            filePath (string): The path at which to save the model

        Returns:
            None
        """
        torch.save(self.model.state_dict(), filePath)

    def load_model(self, filePath):
        """ load_model - Load the state dict save at filePath into the model
        currently store in the trainer. Model must already be initialized and
        have the correct architecture to load the state dict.

        Arguments:
            filePath (string): Path to file to load state dict from

        Returns:
            None
        """
        self.model.load_state_dict(torch.load(filePath))

    def train(self, n_epochs, validate=True, checkpoints=None):
        """ train - Actually train the network for the given number of epochs.

        Arguments:
            n_epochs (int): The number of epochs to train for
            validate (bool): Set to false to disable validation
            checkpoints (string): If given, save model checkpoints as .pt files
            in the directory pointed to by this string. Model will be saved
            at the end of every epoch

        Returns:
            None
        """

        # Initialize arrays to keep track of loss
        self.tr_loss_array = np.zeros(n_epochs)
        self.val_loss_array = np.zeros(n_epochs)

        print("\n######## START TRAINING LOOP ########")
        # The training loop
        for epoch in range(n_epochs):
            print("\nNow starting epoch", str(epoch), "of", str(n_epochs))

            # Put model in training mode
            self.model.train()

            # Initialize loss counter
            tr_loss = 0

            for i, data in enumerate(self.train_data):

                # Pull sample and labels from data array, and flatten sample
                sample, label = data
                sample = torch.flatten(sample, start_dim=1, end_dim=2)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(sample)

                # Evaluate loss
                loss = self.loss_function(output, label)
                tr_loss += loss

                # Backprop and optimizer step
                loss.backward()
                self.optimizer.step()

            # Write training loss to array
            self.tr_loss_array[epoch] = tr_loss / self.tr_batch_per_epoch

            # Now validate if necessary!
            if validate:
                self.validate(epoch)

            # And save model if desired
            if checkpoints != None:
                filename = "/checkpt_e" + str(epoch) + ".pt"
                path  = checkpoints + filename
                self.save_model(path)

            # Lastly print losses
            print("--Training loss: ", str(self.tr_loss_array[epoch]))
            print("--Validation loss: ", str(self.val_loss_array[epoch]))

        # Print end of training loop
        print("\n######## END TRAINING LOOP ########")

    def validate(self, epoch=None):
        """ validate - Validate the model, usually in the course of training.
        This function simply calculates the validation loss and writes it
        to val_loss_array.

        Arguments:
            epoch (int) - The epoch in val_loss_array at which to write
            validation loss. If left at None, just print validation loss.

        Returns:
            None
        """

        # Put model in evaluation mode
        self.model.eval()

        # Initialize loss counter
        val_loss = 0

        for i, data in enumerate(self.valid_data):

            # Pull sample and labels from data array, and flatten sample
            sample, label = data
            sample = torch.flatten(sample, start_dim=1, end_dim=2)

            # Evaluation forward pass
            output = self.model(sample)

            # Evaluate loss
            loss = self.loss_function(output, label)
            val_loss += loss

        # Write validation loss to array, or print as desired
        avg_val_loss = val_loss / self.val_batch_per_epoch
        if epoch != None:
            self.val_loss_array[epoch] = avg_val_loss
        else:
            print("Validation loss: ", avg_val_loss)
