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

        # Initialize dict of ak arrays
        self.ak_dict = {}

        # Loop over branches list and push awkward array into ak_dict
        for branch_name in branches:
            self.ak_dict[branch_name] = tree[branch_name].array()

        # Arrays should have all equal first dimension, which is number of
        # events. Find this number
        self.num_events = len(tree[branches[0]])

        # Finally extract labels from tree. Natively store these as numpy
        self.labels = ak.to_numpy(tree[signal_name].array())

    def plot_branches(self, branches, log=True, directory=None):
        """ plot_branches - This function takes in a list of strings that are
        the names of branches to be plotted. It then either displays
        a histogram of these branches or saves a .png of the histogram to a
        directory.

        Arguments:
            branches (list): List of strings containing branch names
            signal (string): Key of signal branch in ak_dict
            log (bool): Plot on a log scale
            directory (string): Path of directory to save histograms in.
            If not set, display histograms

        Returns:
            None
        """

        # Loop through list of branches to plot
        for branch_name in branches:

            # Pull branch from ak_dict
            thisBranch = self.ak_dict[branch_name]

            # Separate signal and background
            thisBranchSig = thisBranch[self.labels == 1, :]
            thisBranchBkg = thisBranch[self.labels == 0, :]

            # Call ak.flatten to remove all structure from array
            thisBranchSig = ak.flatten(thisBranchSig)
            thisBranchBkg = ak.flatten(thisBranchBkg)

            # Can now make a histogram
            plt.hist(thisBranchSig, alpha=0.5, label='Signal')
            plt.hist(thisBranchBkg, alpha=0.5, label='Background')
            if log:
                plt.yscale('log')
            plt.title(branch_name)
            plt.legend()
            if directory != None:
                filename = directory + branch_name + ".png"
                plt.savefig(filename, dpi=300)
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
        self.max_constits = max_constits
        # Loop through ak_dict
        for key, data_feature in self.ak_dict.items():
            # Pad with None values, and then replace None with 0's
            df_none = ak.pad_none(data_feature, max_constits, axis=1, clip=True)
            df_zero = ak.fill_none(df_none, 0)
            # Set ak_dict entry to df_zero
            self.ak_dict[key] = df_zero

        # Now combine features in ak_dict into a rectangular array using
        # numpy.stack. First convert to numpy
        np_dict = {key: ak.to_numpy(df).astype('float32') for key, df in self.ak_dict.items()}
        # Call numpy.stack
        data = np.dstack(tuple(np_dict.values()))

        # Now make torch tensors
        data_torch = torch.from_numpy(data)
        label_torch = torch.from_numpy(cat_labels)

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
        return (self.max_constits, len(self.ak_dict))

if __name__ == '__main__':

    # Some simple test code for this class
    my_branches = ['fjet_sortClusStan_pt', 'fjet_sortClusCenterRotFlip_eta',
                'fjet_sortClusCenterRot_phi', 'fjet_sortClusStan_e']
    my_dump = DataDumper("../Data/sample_1M.root", "train", my_branches, 'fjet_signal')

    # my_dump.plot_branches(my_branches)

    torch_dl = my_dump.torch_dataloader(max_constits=80, batch_size=100, shuffle=True)
    print(len(torch_dl))
    print(my_dump.sample_shape())
