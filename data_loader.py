""" data_loader.py - This class will prepare data for input into models
in much the same way as the data_handler class. The difference is that this
class will return a keras data generator constructed with uproot's lazy arrays.
This will allow us to loop over an entire training dataset while only ever
having a fraction of it loaded in memory (important for training image based
tagger).

Author: Kevin Greif
python3
Last updated 9/16/21
"""

import uproot
import awkward as ak
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence


class DataLoader(Sequence):
    """ DataLoader - The role of this class is to build a keras data generator
    object from data loaded in with uproot. It will subclass the keras Sequence
    class.
    """

    def __init__(self, root_path, tree_name, input_branch,
                 batch_size=100, max_constits=80, to_fit=True, shuffle=True,
                 net_type='dnn'):
        """ __init__ - Init function for this class. Will load in root file
        using uproot. As Sequence class has no specific init function, don't
        need to worry about calling super().

        Arguments:
            root_path (string): Path to root file containing TTree
            tree_name (string): Name of TTree to pull in file
            input_branch (list): List of strings of the names of branches we wish
            to extract and make inputs for model
            extras (list): List of strings of the names of branches that will
            be extras, packaged into data generator but not fed directly to model.
            Examples are event weights, jetPt, etc.
            max_constits (int): The maximum number of constituents to keep
            in a single jet. Jets with less than this will be padded, jets
            with more will be padded. If set to 0, assume we are considering
            only high level jet information.

        Returns:
            None
        """

        # First we set some instance variables for later use
        self.input_branch = input_branch
        self.batch_size = batch_size
        self.max_constits = max_constits
        self.to_fit = to_fit
        self.shuffle = shuffle
        self.net_type = net_type

        # We want to open our uproot file using lazy arrays. This will
        # allow us to load in data as it becomes necessary rather than all at
        # once. Further we can load only the branches we will be using
        tree_path = root_path + ":" + tree_name
        # Here we hard code label and weight branch names
        branches = self.input_branch + ['fjet_signal', 'fjet_match_weight_pt']
        self.cache = uproot.LRUArrayCache("3 GB")
        self.arrays = uproot.lazy(tree_path,
                                  filter_name=branches,
                                  step_size='500 MB',
                                  array_cache=self.cache)

        # Once we have lazy arrays loaded, we can pull the number of events
        self.num_events = len(self.arrays[self.input_branch[0]])
        if self.max_constits:
            self.sample_shape = (self.max_constits, len(self.input_branch))
        else:
            self.sample_shape = (len(self.input_branch),)


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
        data. It will need to perform all of the necessary padding and reshaping
        on the fly. Concerned this will make everything very slow, but we'll see.

        Arguments:
            index (array) - The indeces of events to return

        Returns:
            (tuple) - A tuple with the entries (data, labels, weights) if to_fit
            is True, and only (data,) if to_fit is False.
        """

        # First we want to read data from lazy arrays. This will load data
        # into memory and fill cache. First get start/stop indeces of batch
        start = index * self.batch_size
        stop = (index + 1) * self.batch_size

        # Then get data from arrays
        batch_data = self.arrays[start:stop]

        ### Data processing

        # Initialize drop list to get rid of events which might have exit codes
        # instead of good variable values
        drop_index_list = []

        # Initialize list of branches that will be passed to numpy dstack
        padded_branches = []

        # Now we loop through each of the input branches
        for branch in self.input_branch:

            # Pull branch from batch_data
            this_branch = batch_data[branch]

            # We either have constituent data (awkward arrays with variable length per jet)
            # or high level data (arrays with equal length per jet). Treat these cases differently
            typestr = str(ak.type(this_branch))

            # Constituent case
            if 'var' in typestr:

                # Pad jets with less than the maximum number of constituents
                # with 0's, and trucate jets with more than the maximum.
                df_zero = ak.pad_none(this_branch, self.max_constits, axis=1, clip=True)
                df_zero = ak.fill_none(df_zero, 0)

                # Append df_zero to padded_branches
                padded_branches.append(ak.to_numpy(df_zero))

            # High level case
            else:

                raise NotImplementedError()

        # Now combine features in padded_branches into a rectangular array
        # using numpy.dstack
        stacked_batch = np.squeeze(np.dstack(tuple(padded_branches)))

        # Preprocessing can introduce NaNs and Infs, set these to 0 here
        np.nan_to_num(stacked_batch, copy=False)

        # Make cut on events, removing those whose indeces occur in drop_index if there are any
        if len(drop_index_list):
            drop_index = np.unique(np.concatenate(drop_index_list))
            final_batch = np.delete(stacked_batch, drop_index, axis=0)
            # cat_labels = np.delete(cat_labels, drop_index, axis=0)
            # extras = [np.delete(df, drop_index) for df in extras]

            # Update number of events
            self.num_events = np.shape(stacked_batch)[0]


        # Now we have model dependent reshapes
        if 'dnn' in self.net_type:

            # Input shape for all dnn networks can be found from sample shape
            input_shape = np.prod(self.sample_shape)

            # Now we do reshaping
            shaped_batch = stacked_batch.reshape((self.batch_size, input_shape))

        elif self.net_type == 'efn':

            # For EFNs, we need to split pT and angular information
            shaped_batch = [stacked_batch[:,:,0], stacked_batch[:,:,1:3]]

        elif net_type == 'pfn':

            # For PFNs, we don't need to do anything!
            shaped_batch = stacked_batch


        # Finally package everything into a tuple and return
        if self.to_fit:
            # If we are fitting, also process labels and weights

            ### Labels Processing

            # Send all of the labels to a 1 hot encoded
            # array. Hardcode 2 classes (signal, background)
            np_labels = ak.to_numpy(batch_data['fjet_signal'])
            cat_labels = np.eye(2, dtype='float32')[np_labels]

            ### Weights Processing

            # For weights, we just need to convert to numpy
            np_weights = ak.to_numpy(batch_data['fjet_match_weight_pt'])

            # Finally return
            return shaped_batch, cat_labels, np_weights

        else:
            # If we are not fitting, just return the data
            return shaped_batch


    def on_epoch_end(self):
        """ on_epoch_end - This function will get called at the end of each
        epoch. Here we will shuffle the data if desired.
        """

        indeces = np.arange(0, self.num_events, 1)
        np.random.shuffle(indeces)
        self.arrays = self.arrays[indeces]


    def sample_shape(self):
        """ sample_shape - Return the shape of each sample.

        Arguments:
            None

        Returns:
            (tuple): The shape of each sample as a tuple
        """

        if self.max_constits:
            return (self.max_constits, len(self.input_branch))
        else:
            return (len(self.input_branch),)


if __name__ == '__main__':

    # Let's set up some simple testing code.
    filepath = "../Data/sample_1p5M_nbpt_test.root"
    tree = 'FlatSubstructureJetTree'
    input_branches = ['fjet_sortClusNormByPt_pt', 'fjet_sortClusCenterRotFlip_eta',
                      'fjet_sortClusCenterRot_phi', 'fjet_sortClusNormByPt_e']
    extra_branches = ['fjet_match_weight_pt', 'fjet_pt']

    dloader = DataLoader(filepath, tree, input_branches, to_fit=True)

    print(np.shape(dloader[0][0]))
    print(np.shape(dloader[0][1]))
    print(np.shape(dloader[0][2]))
