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
# plt.style.use('~/mattyplotsalot/allpurpose.mplstyle')


class DataDumper():
    """ DataDumper - Class takes in a path to a Root file, and then has
    several methods which dump data into desired format.
    """

    def __init__(self, root_path, tree_name, input_branch, signal_name, extras=None):
        """ __init__ - Init function for this class. Will produce awkward
        arrays of selected branches, stored in a python list

        Arguments:
            root_path (string): Path to root file containing TTree
            tree_name (string): Name of TTree to pull in file
            input_branch (list): List of strings of the names of branches we wish
            to extract and make inputs for model
            signal_name (string): Name of signal branch in tree
            extras (list): List of strings of the names of branches that will
            be extras, packaged into data loader but not fed directly to model.
            Examples are event weights, jetPt, etc.

        Returns:
            None
        """

        # First we need to load in data using uproot
        input_file = uproot.open(root_path)

        # Separate out train/valid trees.
        tree = input_file[tree_name]

        # Initialize dict of ak arrays for inputs and extras
        self.input_dict = {}
        self.extra_dict = {}

        # Loop over branches list and push awkward array into input_dict
        for branch_name in input_branch:
            self.input_dict[branch_name] = tree[branch_name].array()

        # Next deal with labels. Simply store these as an instance variable,
        # in the form of a numpy array.
        self.labels = ak.to_numpy(tree[signal_name].array())

        # The length of our labels array should be the number total number
        # of events in this set.
        self.num_events = len(self.labels)

        # Finally deal with the extras. Loop over extras list and push arrays
        # into extra_dict. Can natively store these as numpy, since we assume
        # they are event level features
        if extras != None:
            for branch_name in extras:
                np_extra = ak.to_numpy(tree[branch_name].array())
                self.extra_dict[branch_name] = np_extra

    def plot_branches(self, branches, log=True, directory=None):
        """ plot_branches - This function takes in a list of strings that are
        the names of branches to be plotted. It then either displays
        a histogram of these branches or saves a .png of the histogram to a
        directory.

        Arguments:
            branches (list): List of strings containing branch names that
            should be plotted. Can be in input or extras dict.
            log (bool): Plot on a log scale
            directory (string): Path of directory to save histograms in.
            If not set, display histograms

        Returns:
            None
        """

        # Loop through list of branches to plot
        for branch_name in branches:
            print("Now plotting", branch_name)

            # Find branch in either dictionary
            if branch_name in self.input_dict:
                thisBranch = self.input_dict[branch_name]
            elif branch_name in self.extra_dict:
                thisBranch = self.extra_dict[branch_name]
            else:
                print("Branch ", branch_name, " not found in dictionaries!")
                continue

            # Separate signal and background
            thisBranchSig = thisBranch[self.labels == 1, ...]
            thisBranchBkg = thisBranch[self.labels == 0, ...]

            # Call ak.flatten to remove all structure from array if needed
            typestr = str(ak.type(thisBranch))
            if 'var' in typestr:
                thisBranchSig = ak.flatten(thisBranchSig)
                thisBranchBkg = ak.flatten(thisBranchBkg)

            # Can now make a histogram
            n, bins, patches = plt.hist(thisBranchBkg, alpha=0.5, label='Background')
            plt.hist(thisBranchSig, bins=bins, alpha=0.5, label='Signal')
            if log:
                plt.yscale('log')
            plt.title(branch_name)
            plt.legend()
            if directory != None:
                filename = directory + branch_name + ".png"
                plt.savefig(filename, dpi=300)
                plt.clf()
            else:
                plt.show()

    def torch_dataloader(self, max_constits=80, **kwargs):
        """ torch_dataloader - Packages data and labels into a torch data-
        loader which can be used for training a model. Takes in maximum
        number of constituents to be passed in each jet.

        Arguments:
            max_constits (int): Number of constituents to be passed

        Returns:
            (torch.utils.data.DataLoader)
        """

        # First send all of the labels to a 1 hot encoded
        # array. Hardcode 2 classes (signal, background)
        cat_labels =  np.eye(2, dtype='float32')[self.labels]

        # Next pad jets with less than the maximum number of constituents
        # with 0's, and trucate jets with more than the maximum.
        # Assume only input features will be of variable length, need to be
        # padded.
        self.max_constits = max_constits
        # Loop through input_dict
        for key, data_feature in self.input_dict.items():
            # Pad with None values, and then replace None with 0's
            df_none = ak.pad_none(data_feature, max_constits, axis=1, clip=True)
            df_zero = ak.fill_none(df_none, 0)
            # Set ak_dict entry to df_zero
            self.input_dict[key] = df_zero

        # Now combine features in input_dict into a rectangular array using
        # numpy.stack. First convert to numpy
        np_dict = {key: ak.to_numpy(df).astype('float32') for key, df in self.input_dict.items()}
        # Call numpy.stack
        data = np.dstack(tuple(np_dict.values()))

        # Now make torch tensors from the data and labels
        data_torch = torch.from_numpy(data)
        label_torch = torch.from_numpy(cat_labels)

        # Next make torch tensors from the arrays in extra_dict
        # Want to pass these to tensor dataset constructer contained in list
        # that we can unpack.
        extra_tensors = []
        for key, data_feature in self.extra_dict.items():
            extra_tensors.append(torch.from_numpy(data_feature))

        # And return DataLoader
        dataset = torch.utils.data.TensorDataset(data_torch, label_torch, *extra_tensors)
        return torch.utils.data.DataLoader(dataset, **kwargs)

    def sample_shape(self):
        """ sample_shape - Return the shape of each sample.

        Arguments:
            None

        Returns:
            (tuple): The shape of each sample as a tuple
        """
        return (self.max_constits, len(self.input_dict))

if __name__ == '__main__':

    # Some simple test code for this class
    my_branches = ['fjet_sortClusNormByPt_pt', 'fjet_sortClusCenterRotFlip_eta',
                'fjet_sortClusCenterRot_phi', 'fjet_sortClusNormByPt_e']
    my_extras = ['fjet_testing_weight_pt', 'fjet_pt']
    my_dump = DataDumper("../Data/unshuf_test.root", "train", my_branches, 'fjet_signal', extras=my_extras)

    my_dump.plot_branches(my_extras)

    torch_dl = my_dump.torch_dataloader(max_constits=80, batch_size=100, shuffle=True)
    print(len(torch_dl))
    print(my_dump.sample_shape())
    print(my_dump.num_events)
    print(torch_dl.dataset[1])
