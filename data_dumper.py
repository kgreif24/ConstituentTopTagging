""" data_dumper.py - This class will utilize uproot to take a TTree from a
root file, and dump specified branches into a ML friendly format. First
implementation will be torch dataloaders, but this may change!

Author: Kevin Greif
Last updated 6/21/21
python3
"""

import torch
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import numpy as np

# Use custom stylesheet for plotting
plt.style.use('/home/kgreif/Documents/General Programming/mattyplotsalot/allpurpose.mplstyle')


class DataDumper():
    """ DataDumper - Class takes in a path to a Root file, and then has
    several methods which dump data into desired format.
    """

    def __init__(self, root_path, tree_name, branches, signal_name):
        """ __init__ - Init function for this class. Will produce awkward
        arrays of selected branches, stored in a python list

        Arguments:
            root_path (string): Path to root file containing TTree
            tree_name (string): Name of TTree to pull in file
            branches (list): List of strings of the names of branches we wish
            to extract
            signal_name (string): Name of signal branch in tree

        Returns:
            None
        """

        # First we need to load in data using uproot
        input_file = uproot.open(root_path)

        # Separate out train/valid trees. REFACTOR!!!
        tree = input_file[tree_name]

        # Initialize list of ak arrays
        self.ak_list = []

        # Loop over branches list and push awkward array into ak_list
        for branch_name in branches:
            self.ak_list.append(tree[branch_name].array())

        # Arrays should have all equal first dimension, which is number of
        # events. Find this number
        self.num_events = len(self.ak_list[0])

        # Finally extract labels from tree. Natively store these as numpy
        self.labels = ak.to_numpy(tree[signal_name].array())

    def to_categorical(self, num_classes):
        """ to_categorical - Sends self.labels to a one-hot encoded numpy array.
        Array will have float datatype for usage with pytorch loss functions.

        Arguments:
            num_classes (int) - The number of classes to use in 1-hot encoding

        Returns:
            None
        """

        self.labels =  np.eye(num_classes, dtype='float32')[self.labels]

    def pad_zeros(self, max_constits):
        """ pad_zeros - This function will loop through awkward arrays in
        ak_list and pad with zeros so their shape is rectangular.

        Arguments:
            max_constits (int): The maximum number of constituents to have in
            a single event

        Returns:
            None
        """

        # Initialize new list and set max_constits to instance variable
        padded_list = []
        self.max_constits = max_constits

        # Loop through ak_list
        for data_feature in self.ak_list:

            # Pad with None values, and then replace None with 0's
            df_none = ak.pad_none(data_feature, max_constits, axis=1, clip=True)
            df_zero = ak.fill_none(df_none, 0)

            # Append to padded_list
            padded_list.append(df_zero)

        # Redefine ak_list
        self.ak_list = padded_list

    def numpy_stack(self):
        """ numpy_stack - Return data as a single numpy array, stacked in
        innermost dimension using the dstack function.

        Arguments:
            None

        Returns:
            (array): Stacked data
        """

        # First convert all elements of ak_list to numpy (requires they
        # all be rectangular!!). Also convert to float32 for pytorch usage
        self.ak_list = [ak.to_numpy(df).astype('float32') for df in self.ak_list]

        # Now call np.dstack
        return np.dstack(tuple(self.ak_list))

    def torch_dataloader(self, **kwargs):
        """ torch_dataloader - Packages data and labels into a torch data-
        loader which can be used for training a model.

        Arguments:
            None

        Returns:
            (torch.utils.data.DataLoader)
        """

        # First make a rectangular array of training data using numpy_stack
        data = self.numpy_stack()

        # Now make torch tensors
        data_torch = torch.from_numpy(data)
        label_torch = torch.from_numpy(self.labels)

        # And return DataLoader
        dataset = torch.utils.data.TensorDataset(data_torch, label_torch)
        return torch.utils.data.DataLoader(dataset, **kwargs)

    def sample_shape(self):
        """ sample_shape - Return the shape of each sample.

        Arguments:
            None

        Returns:
            (tuple): The shape of each sample as a tuple
        """
        return (self.max_constits, len(self.ak_list))

if __name__ == '__main__':

    # Some simple test code for this class
    my_branches = ['fjet_sortClusNormByPt_pt', 'fjet_sortClusCenterRotFlip_eta',
                'fjet_sortClusCenterRot_phi', 'fjet_sortClusNormByPt_e']
    my_dump = DataDumper("../Data/unshuf_test.root", "train", my_branches, 'fjet_signal')

    my_dump.to_categorical(2)

    my_dump.pad_zeros(80)

    torch_dl = my_dump.torch_dataloader(batch_size=100, shuffle=True)
    print(len(torch_dl))
