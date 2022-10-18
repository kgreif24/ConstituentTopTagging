""" dl_syst.py - Copy of the data loader class which is setup for training
a PFN on systematics data.

Author: Kevin Greif
python3
Last updated 10/18/22
"""

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

import time


class DataLoader(Sequence):
    """ DataLoader - The role of this class is to build a keras data generator
    object from data loaded in with h5py. It will subclass the keras Sequence
    class.
    """

    def __init__(self,
                 file_path,
                 batch_size=100,
                 mode='train',
                 max_constits=80,
                 max_jets=-1,
                 num_folds=5,
                 this_fold=1):
        """ __init__ - Init function for this class. Will load in root file
        using uproot. As Sequence class has no specific init function, don't
        need to worry about calling super().

        Arguments:
            file_path (string): Path to h5py file containing TTree
            batch_size (int): The number of jets in each batch, used to
            calculate length of the loader
            mode (string): Either train, valid, or test. Train gives everything but valid fold,
            valid gives only the valid fold, and test gives the entire dataset.
            net_type (string): Specifies the model specific data preparation to use
            max_constits (int): Number of constituents to include in constituent models
            max_jets (int): Number of jets to use in train / valid sets
            num_folds (int): The number of folds data is split into
            this_fold (int): The number of the fold to use, 1-num_folds

        Returns:
            None
        """

        # First we set some instance variables for later use
        self.max_constits = max_constits
        self.batch_size = batch_size
        self.mode = mode

        # Load hdf5 file using h5py, braches will be loaded in get item function
        self.file = h5py.File(file_path, 'r')

        # Once we have file loaded, we can pull the number of events
        tot_events = self.file.attrs.get('num_jets')

        # Set sample shape
        self.sample_shape = (self.max_constits, 4)

        # Decide train/valid split for the given fold. Generator will only
        # return data from one of these partitions, unless mode is set to train.
        indeces = np.arange(0, tot_events, 1)
        folds = np.array_split(indeces, num_folds)

        # Mode of loader will determine how we handle spliting up indeces. First handle
        # k-fold procedure for training and validation modes.
        if not self.mode == 'test':
            valid_indeces = folds.pop(this_fold - 1)  # -1 to make this_fold an index
            train_indeces = np.concatenate(folds)

            # Instead of using these indeces for indexing h5py file every time (slow)
            # lets only do this on the "seam" batch. Find location of seam in train_indeces
            self.seam = valid_indeces[0]

        # For testing we don't need to do anything, except define a dummy value of seam
        # to avoid error message. This is a bit messy, will clean up later if I remember.
        else:
            self.seam = 0

        # Now choose number of events and indeces based on whether we are doing
        # training or validation
        if self.mode == 'train':
            self.indeces = train_indeces
            self.num_events = len(train_indeces)
        elif self.mode == 'valid':
            self.indeces = valid_indeces
            self.num_events = len(valid_indeces)
        elif self.mode == 'test':
            self.indeces = indeces
            self.num_events = tot_events
        else:
            raise ValueError("Mode keyword argument must be train, test, or valid")


    def __len__(self):
        """ __len__ - This function returns the number of batches in each epoch
        given the size of the data and the batch size. Use remainder of data as
        partial batch.

        Arguments:
            None

        Returns:
            (int) - Number of batches in each epoch
        """
        return int(np.ceil(self.num_events / self.batch_size))


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

        # Correct indexing if this is the last batch
        this_bs = self.batch_size
        if stop > self.num_events:
            stop = self.num_events
            this_bs = stop - start

        # We want to use different indexing for the "seam" batch and all other batches
        # Condition on this here. Note this condition should only trigger when mode is
        # training. Validation and testing sequences should never trigger the condition,
        # so include an assertion to make sure this never happens.
        if start < self.seam and stop > self.seam:

            # Given this is true, we just index using a slice of the numpy array
            batch_indeces = self.indeces[start:stop]
            batch_data = self.file['constit'][batch_indeces,:self.max_constits,:]

            # We also need to load labels and weights, since this conditional will only
            # occur if we are training
            assert not self.mode == 'valid' or self.mode == 'test'
            batch_labels = self.file['labels'][batch_indeces]

        else:

            # If this batch doesn't sit on the seam, pull actual start/stop from numpy array
            batch_start = self.indeces[start]
            # Unfortunately finding stop index is rather complicated. We want to reference at stop-1
            # since stop is one more than an index. However we want to add 1 to the number at stop-1
            # as this number needs to be turned into the second number in a slice. What a mess.
            batch_stop = self.indeces[stop-1] + 1
            batch_data = self.file['constit'][batch_start:batch_stop,:self.max_constits,:]

            # Now load labels and weights the regular way
            batch_labels = self.file['labels'][batch_start:batch_stop]

        # Finally package everything into a tuple and return
        return batch_data, batch_labels, batch_weights


if __name__ == '__main__':

    # Let's set up some simple testing code.
    filepath = '/scratch/whiteson_group/kgreif/train_ln_m.h5'

    dloader = DataLoader(filepath, net_type='dnn', mode='train')

    print("Loader length:", len(dloader))
    print("Sample shape:", dloader.sample_shape)

    start = time.time()
    input, labels, weights = dloader[0]
    shape = (100,) + dloader.sample_shape
    input = input.reshape(shape)
    print("Time for loading a single batch:", time.time() - start)
