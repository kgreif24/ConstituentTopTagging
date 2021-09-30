""" data_loader.py - This class will prepare data for input into models
in much the same way as the data_handler class. The difference is that this
class will return a keras data generator constructed with h5py's indexing.
This will allow us to loop over an entire training dataset while only ever
having a fraction of it loaded in memory (important for training image based
tagger).

As of now the number of constituents is set in to_hdf.py step. Could make 
this into a keyword argument here if we ever want to vary number constituents.

Author: Kevin Greif
python3
Last updated 9/30/21
"""

import h5py
import awkward as ak
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence


class DataLoader(Sequence):
    """ DataLoader - The role of this class is to build a keras data generator
    object from data loaded in with h5py. It will subclass the keras Sequence
    class.
    """

    def __init__(self, file_path, batch_size=100, to_fit=True, shuffle=True,
                 net_type='dnn'):
        """ __init__ - Init function for this class. Will load in root file
        using uproot. As Sequence class has no specific init function, don't
        need to worry about calling super().

        Arguments:
            file_path (string): Path to h5py file containing TTree
            batch_size (int): The number of jets in each batch, used to
            calculate length of the loader
            to_fit (bool): If true, include labels in batch
            shuffle (bool): If true, shuffle jets before each epoch
            net_type (string): Specifies the model specific data preparation to use

        Returns:
            None
        """

        # First we set some instance variables for later use
        self.max_constits = 80 # Hardcoded for now, see file doc string
        self.batch_size = batch_size
        self.to_fit = to_fit
        self.shuffle = shuffle
        self.net_type = net_type

        # Load hdf5 file using h5py, braches will be loaded in get item function
        self.file = h5py.File(file_path, 'r')

        # Once we have lazy arrays loaded, we can pull the number of events
        # Use the weights array since it is just a vector with length equal to
        # the number of events
        self.num_events = len(self.file['weights'])

        # Now we need to find the sample shape, refactor this!
        self.sample_shape = (self.max_constits, 4)


    def __len__(self):
        """ __len__ - This function returns the number of batches in each epoch
        given the size of the data and the batch size. We will drop remainder
        of data for now.

        Arguments:
            None

        Returns:
            (int) - Number of batches in each epoch
        """
        return int(np.floor(self.num_events / self.batch_size))


    def __getitem__(self, index):
        """ __getitem__ - This function will return a single batch of
        data. It will need to get data from h5py file object, then perform
        the necessary model specific reshaping.

        Arguments:
            index (array) - The index of the batch to return

        Returns:
            (tuple) - A tuple with the entries (data, labels, weights) if to_fit
            is True, and only (data,) if to_fit is False.
        """

        # First we need to figure out how to slice data sitting in hdf5 file
        # Calculate start and stop indeces
        start = index * self.batch_size
        stop = (index + 1) * self.batch_size

        # Then pull just the data into memory
        batch_data = self.file['constits'][start:stop]

        # Now we have model dependent reshapes
        if 'dnn' in self.net_type:

            # Input shape for all dnn networks can be found from sample shape
            input_shape = np.prod(self.sample_shape)

            # Now we do reshaping
            shaped_data = batch_data.reshape((self.batch_size, input_shape))

        elif self.net_type == 'efn':

            # For EFNs, we need to split pT and angular information
            shaped_data = [batch_data[:,:,0], batch_data[:,:,1:3]]

        elif net_type == 'pfn':

            # For PFNs, we don't need to do anything!
            shaped_data = batch_data


        # Finally package everything into a tuple and return
        if self.to_fit:
            # If we are fitting, also load labels and weights
            batch_labels = self.file['labels'][start:stop,:]
            batch_weights = self.file['weights'][start:stop]

            # Finally return
            return shaped_data, batch_labels, batch_weights

        else:
            # If we are not fitting, just return the data
            return shaped_data


if __name__ == '__main__':

    # Let's set up some simple testing code.
    filepath = "/pub/kgreif/samples/h5dat/sample_4p2M_stan.hdf5"

    dloader = DataLoader(filepath)

    print(len(dloader))
    print(dloader.sample_shape)

    print(np.shape(dloader[0][0]))
    print(np.shape(dloader[0][1]))
    print(np.shape(dloader[0][2]))
