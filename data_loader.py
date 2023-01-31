""" data_loader.py - This class will pipe data into the keras fit function.
This will allow us to loop over an entire training dataset while only ever
having a fraction of it loaded in memory.

Further, shuffling is currently done by keras' fit function at batch level,
so jets within a batch will always stay the same.

Author: Kevin Greif
python3
Last updated 10/16/22
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
                 net_type='dnn',
                 max_constits=80,
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
            num_folds (int): The number of folds data is split into
            this_fold (int): The number of the fold to use, 1-num_folds

        Returns:
            None
        """

        # First we set some instance variables for later use
        self.max_constits = max_constits
        self.batch_size = batch_size
        self.mode = mode
        self.net_type = net_type

        # Load hdf5 file using h5py, braches will be loaded in get item function
        self.file = h5py.File(file_path, 'r')

        # Once we have file loaded, we can pull the number of events
        tot_events = self.file.attrs.get('num_jets')

        # Now we need to find the sample shape, depends on net_type
        if 'hl' in self.net_type:
            self.sample_shape = self.file['hl'].shape[1:]
            self.load_taste = False
        elif self.net_type == 'resnet':
            self.sample_shape = (64, 64, 1)  # Simply hard code image dims
            self.load_taste = False
        elif self.net_type == 'dnn':
            self.sample_shape = (self.max_constits, 10)
            self.load_taste = True
        elif self.net_type == 'efn':
            self.sample_shape = (self.max_constits, 2)
            self.load_taste = True
        elif self.net_type == 'pfn':
            self.sample_shape = (self.max_constits, 7)
            self.load_taste = False
        elif self.net_type == 'pnet':
            self.sample_shape = (self.max_constits, 7)
            self.load_taste = False
        else:
            raise ValueError("Model type not recognized!")

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

        # Now we decide what kind of data we want to pull (hl or constit)
        if 'hl' in self.net_type:
            data_key = 'hl'
        elif self.net_type == 'resnet':
            data_key = 'images'
        else:
            data_key = 'constit'

        # We want to use different indexing for the "seam" batch and all other batches
        # Condition on this here. Note this condition should only trigger when mode is
        # training. Validation and testing sequences should never trigger the condition,
        # so include an assertion to make sure this never happens.
        if start < self.seam and stop > self.seam:

            # Given this is true, we just index using a slice of the numpy array
            batch_indeces = self.indeces[start:stop]
            batch_data = self.file[data_key][batch_indeces,...]
            if self.load_taste:
                taste_data = self.file['fjet_clus_taste'][batch_indeces,...]
                batch_data = np.concatenate([batch_data, taste_data], axis=2)

            # Resnet needs pt data for pixel intensity
            if self.net_type == 'resnet':
                batch_pt = self.file['constit'][batch_indeces,:,2]

            # We also need to load labels and weights, since this conditional will only
            # occur if we are training
            assert not self.mode == 'valid' or self.mode == 'test'
            batch_labels = self.file['labels'][batch_indeces]
            batch_weights = self.file['weights'][batch_indeces]

        else:

            # If this batch doesn't sit on the seam, pull actual start/stop from numpy array
            batch_start = self.indeces[start]
            # Unfortunately finding stop index is rather complicated. We want to reference at stop-1
            # since stop is one more than an index. However we want to add 1 to the number at stop-1
            # as this number needs to be turned into the second number in a slice. What a mess.
            batch_stop = self.indeces[stop-1] + 1
            batch_data = self.file[data_key][batch_start:batch_stop,...]
            if self.load_taste:
                taste_data = self.file['fjet_clus_taste'][batch_start:batch_stop,...]
                batch_data = np.concatenate([batch_data, taste_data], axis=2)

            # Again, resnet needs pt data for pixel intensity
            if self.net_type == 'resnet':
                batch_pt = self.file['constit'][batch_start:batch_stop,:,2]

            # Now load labels and weights the regular way
            batch_labels = self.file['labels'][batch_start:batch_stop]
            batch_weights = self.file['weights'][batch_start:batch_stop]

        # Now we have model dependent reshapes
        if self.net_type == 'hldnn':

            # For hldnn, we don't need to reshape
            shaped_data = batch_data

        elif self.net_type == 'dnn':

            # For DNNs, we need to flatten out vectors of constituents
            input_shape = (this_bs, np.prod(self.sample_shape))
            shaped_data = batch_data[:,:self.max_constits,:].reshape(input_shape)

        elif self.net_type == 'efn':

            # For EFNs, we need to split pT and angular information
            # Dump in taste information just to see what happens
            pT = batch_data[:,:self.max_constits,2]
            ang = batch_data[:,:self.max_constits,0:2]
            taste = batch_data[:,:self.max_constits,7:10]
            phi_in = np.concatenate((ang, taste), axis=2)
            shaped_data = [pT, phi_in]

        elif self.net_type == 'pfn':

            # For PFNs, we don't need to do anything (except index number of constits)
            shaped_data = batch_data[:,:self.max_constits,:]

        elif self.net_type == 'resnet':

            # Ignore self.max_constits in image based network data. That is use all of
            # the constituents to form jet images, not just however many we feed to constituent
            # networks.

            # For image based network, our data is a list of indeces. Will also
            # need jet i.d. information for indexing pixels.
            jet_id = np.repeat(np.arange(0, this_bs, 1), 200, axis=0)

            # Now flatten all index arrays and pt array
            jet_index = np.ravel(jet_id)
            eta_index = np.ravel(batch_data[:,:,0])
            phi_index = np.ravel(batch_data[:,:,1])
            cons_pt = np.ravel(batch_pt)

            # Build images, starting from a zero array and incrementing. Note
            # we use phi as our row coordinate and index from the bottom of the array.
            # This strange indexing makes jet images intuitive.
            shaped_data = np.zeros((this_bs, 64, 64, 1), dtype=np.float32)
            np.add.at(shaped_data, (jet_index, -1*phi_index, eta_index, 0), cons_pt)

        elif self.net_type == 'pnet':

            # pnet expects a dictionary containing points, features, and a mask.
            # Points will be eta,phi information stacked along constituent axis
            # Features will be (pT, eta, phi, E, taste) information as passed to PFN.
            # Mask will be pT information. Use this to allow code to process
            # variable length inputs.
            shaped_data = {}
            shaped_data['points'] = batch_data[:,:self.max_constits,0:2]
            shaped_data['features'] = batch_data[:,:self.max_constits,:]
            shaped_data['mask'] = batch_data[:,:self.max_constits,2]  # pT

        # Finally package everything into a tuple and return
        return shaped_data, batch_labels, batch_weights

class FakeLoader(Sequence):

    def __init__(self):
        self.sample_shape = (64, 64, 1)

    def __len__(self):
        return 129284

    def __getitem__(self, index):
        labels_vec = np.random.randint(0, 1, size=(100,))
        return (np.random.rand(100, 64, 64, 1), labels_vec)


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
