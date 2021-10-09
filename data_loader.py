""" data_loader.py - This class will prepare data for input into models
in much the same way as the data_handler class. The difference is that this
class will return a keras data generator constructed with h5py's indexing.
This will allow us to loop over an entire training dataset while only ever
having a fraction of it loaded in memory (important for training image based
tagger).

In the future, look into chunked memory to further optimize training speed.

As of now the number of constituents is set in to_hdf.py step. Could make 
this into a keyword argument here if we ever want to vary number constituents.

Further, shuffling is currently done by keras' fit function at batch level, 
so jets within a batch will always stay the same. Could look at doing more
fancy shuffling later if needed.

Author: Kevin Greif
python3
Last updated 10/4/21
"""

import h5py
import awkward as ak
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

import time


class DataLoader(Sequence):
    """ DataLoader - The role of this class is to build a keras data generator
    object from data loaded in with h5py. It will subclass the keras Sequence
    class.
    """

    def __init__(self, file_path, batch_size=100, valid=False, shuffle=True,
                 net_type='dnn', num_folds=5, this_fold=1):
        """ __init__ - Init function for this class. Will load in root file
        using uproot. As Sequence class has no specific init function, don't
        need to worry about calling super().

        Arguments:
            file_path (string): Path to h5py file containing TTree
            batch_size (int): The number of jets in each batch, used to
            calculate length of the loader
            valid (bool): If true, feed validation partition and not training
            shuffle (bool): If true, shuffle jets before each epoch
            net_type (string): Specifies the model specific data preparation to use
            num_folds (int): The number of folds data is split into
            this_fold (int): The number of the fold to use, 1-num_folds

        Returns:
            None
        """

        # First we set some instance variables for later use
        self.max_constits = 80 # Hardcoded for now, see file doc string
        self.batch_size = batch_size
        self.valid = valid
        self.shuffle = shuffle
        self.net_type = net_type

        # Load hdf5 file using h5py, braches will be loaded in get item function
        self.file = h5py.File(file_path, 'r')

        # Once we have file loaded, we can pull the number of events
        # Use the weights array since it is just a vector with length equal to
        # the number of events
        tot_events = len(self.file['weights'])

        # Now we need to find the sample shape, depends on net_type
        if 'hl' in self.net_type:
            self.sample_shape = (self.file['hl'].shape[1],)
        else:
            self.sample_shape = (self.max_constits, 4)

        # Decide train/valid split for the given fold. Generator will only
        # return data from one of these partitions, depending on the to_fit flag.
        indeces = np.arange(0, tot_events, 1)
        folds = np.array_split(indeces, num_folds)
        valid_indeces = folds.pop(this_fold - 1)  # -1 to make this_fold an index
        train_indeces = np.concatenate(folds)

        # Instead of using these indeces for indexing h5py file every time (slow)
        # lets only do this on the "seam" batch. Find location of seam in train_indeces
        self.seam = valid_indeces[0]

        # Now choose number of events and indeces based on whether we are doing
        # training or validation
        if not self.valid:
            self.indeces = train_indeces
            self.num_events = len(train_indeces)
        else:
            self.indeces = valid_indeces
            self.num_events = len(valid_indeces)
            

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

        # Now we decide what kind of data we want to pull (hl or constit)
        if 'hl' in self.net_type:
            data_key = 'hl'
        elif self.net_type == 'resnet':
            data_key = 'images'
        else:
            data_key = 'constits'

        # We watch to use different indexing for the "seam" batch and all other batches
        # Condition on this here. Note this will never happen for validation sequences
        # since self.seam is the index of the first element in the train_indeces that "jumps"
        # over validation data. This is either 0 or some number larger than the length of
        # the validation sequence.
        if start < self.seam and stop > self.seam:

            # Given this is true, we just index using a slice of the numpy array
            batch_indeces = self.indeces[start:stop]
            batch_data = self.file[data_key][batch_indeces]

            # Resnet needs pt data for pixel intensity
            if self.net_type == 'resnet':
                batch_pt = self.file['constits'][batch_indeces,:,0]

            # We also need to load labels and weights, since this conditional will only
            # occur if we are training
            assert not self.valid
            batch_labels = self.file['labels'][batch_indeces,:]
            batch_weights = self.file['weights'][batch_indeces]

        else:

            # If this batch doesn't sit on the seam, pull actual start/stop from numpy array
            batch_start = self.indeces[start]
            # Unfortunately finding stop index is rather complicated. We want to reference at stop-1
            # since stop is one more than an index. However we want to add 1 to the number at stop-1
            # as this number needs to be turned into the second number in a slice. What a mess.
            batch_stop = self.indeces[stop-1] + 1
            batch_data = self.file[data_key][batch_start:batch_stop]

            # Again, resnet needs pt data for pixel intensity
            if self.net_type == 'resnet':
                batch_pt = self.file['constits'][batch_start:batch_stop,:,0]

            # Now load labels and weights the regular way
            batch_labels = self.file['labels'][batch_start:batch_stop,:]
            batch_weights = self.file['weights'][batch_start:batch_stop]

        # Now we have model dependent reshapes
        if 'dnn' in self.net_type:

            time_start = time.time()

            # Input shape for all dnn networks can be found from sample shape
            input_shape = np.prod(self.sample_shape)

            # Now we do reshaping
            shaped_data = batch_data.reshape((this_bs, input_shape))

            time_stop = time.time()
            print("Reshape time:", time_stop - time_start)

        elif self.net_type == 'efn':

            # For EFNs, we need to split pT and angular information
            shaped_data = [batch_data[:,:,0], batch_data[:,:,1:3]]

        elif self.net_type == 'pfn':

            # For PFNs, we don't need to do anything!
            shaped_data = batch_data

        elif self.net_type == 'resnet':

            time_start = time.time()

            # For image based network, our data is a list of indeces. Will also
            # need jet i.d. information for indexing pixels
            jet_id = np.repeat(np.arange(0, this_bs, 1), self.max_constits, axis=0)

            # Now flatten all index arrays and pt array
            jet_index = np.ravel(jet_id)
            eta_index = np.ravel(batch_data[:,:,0])
            phi_index = np.ravel(batch_data[:,:,1])
            cons_pt = np.ravel(batch_pt)

            # We want to expand pt information into 3 channels before building images
            # (this makes for much less data for us to copy)
            expanded_pt = np.repeat(np.expand_dims(cons_pt, axis=-1), 3, axis=-1)

            # Build images, starting from a zero array and incrementing
            shaped_data = np.zeros((this_bs, 224, 224, 3), dtype=np.float32)
            shaped_data[jet_index, eta_index, phi_index,:] += expanded_pt

            time_stop = time.time()
            print("Reshape time:", time_stop - time_start)
            
        # Finally package everything into a tuple and return
        return shaped_data, batch_labels, batch_weights

class FakeLoader(Sequence):

    def __init__(self):
        self.sample_shape = (80, 4)

    def __len__(self):
        return 33600

    def __getitem__(self, index):
        labels_vec = np.ones(100, dtype=np.int8)
        labels_cat = np.eye(2, dtype=np.float32)[labels_vec]
        return (np.random.rand(100, 224, 224, 3), labels_cat)


if __name__ == '__main__':

    # Let's set up some simple testing code.
    filepath = "/pub/kgreif/samples/h5dat/sample_4p2M_stan.hdf5"

    dloader = DataLoader(filepath)

    print(len(dloader))
    print(dloader.sample_shape)

    print(np.shape(dloader[0][0]))
    print(np.shape(dloader[0][1]))
    print(np.shape(dloader[0][2]))
