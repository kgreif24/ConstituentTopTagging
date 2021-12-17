""" data_handler.py - This class will utilize uproot to take a TTree from a
root file, and dump specified branches into a ML friendly format. This could be
a torch dataloader, or simply a numpy array that can be passed into a keras
fit routine.

Author: Kevin Greif
Last updated 8/8/21
python3
"""

import matplotlib.pyplot as plt
import uproot
import awkward as ak
import numpy as np

# Use custom stylesheet for plotting
# plt.style.use('~/mattyplotsalot/allpurpose.mplstyle')


class DataHandler():
    """ DataHandler - Class takes in a path to a Root file, and then has
    several methods which dump data into desired format.
    """

    def __init__(self, root_path, tree_name, input_branch,
                 extras=None, max_constits=80):
        """ __init__ - Init function for this class. Will produce awkward
        arrays of selected branches, stored in a python list

        Arguments:
            root_path (string): Path to root file containing TTree
            tree_name (string): Name of TTree to pull in file
            input_branch (list): List of strings of the names of branches we wish
            to extract and make inputs for model
            extras (list): List of strings of the names of branches that will
            be extras, packaged into data loader but not fed directly to model.
            Examples are event weights, jetPt, etc.
            max_constits (int): The maximum number of constituents to keep
            in a single jet. Jets with less than this will be padded, jets
            with more will be padded. If set to 0, assume we are considering
            only high level jet information.

        Returns:
            None
        """

        # First we need to load in data using uproot
        input_file = uproot.open(root_path)

        # Separate out train/valid trees.
        tree = input_file[tree_name]

        # Initialize dict of ak arrays for inputs and extras, as well as
        # max constits instance variable
        self.input_dict = {}
        self.extra_dict = {}
        self.max_constits = max_constits

        # Loop over branches list and push awkward array into input_dict
        for branch_name in input_branch:
            self.input_dict[branch_name] = tree[branch_name].array()

        # Next deal with labels. Simply store these as an instance variable,
        # in the form of a numpy array. Since we always assume signal branch
        # will have the same name, just hardcode this.
        signal_name = 'fjet_signal'
        self.labels = ak.to_numpy(tree[signal_name].array())

        # The length of our labels array should be the number total number
        # of events in this set.
        self.num_events = len(self.labels)

        # Finally deal with the extras. Loop over extras list and push arrays
        # into extra_dict.
        if extras != None:
            for branch_name in extras:
                self.extra_dict[branch_name] = tree[branch_name].array()


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


            # Some HL features have exit codes (-999), these events are ignored
            # in training. Remove these numbers from plotting here
            thisBranchSig = thisBranchSig[thisBranchSig != -999]
            thisBranchBkg = thisBranchBkg[thisBranchBkg != -999]

            # Can now make a histogram
            n, bins, patches = plt.hist(thisBranchBkg, alpha=0.5, label='Background')
            plt.hist(thisBranchSig, bins=bins, alpha=0.5, label='Signal')
            if log:
                plt.yscale('log')
            axes = plt.gca()
            bottom, top = axes.get_ylim()
            axes.set_ylim([1, top])
            plt.title(branch_name)
            plt.legend()
            if directory != None:
                filename = directory + branch_name + ".png"
                plt.savefig(filename, dpi=300)
                plt.clf()
            else:
                plt.show()


    def np_arrays(self, **kwargs):
        """ np_arrays - Packages data, labels, and extras in numpy arrays,
        and returns arrays as a list of that can then be sent into keras api.

        Arguments:
            None

        Returns:
            (list): List of np arrays
        """

        # First send all of the labels to a 1 hot encoded
        # array. Hardcode 2 classes (signal, background)
        cat_labels = np.eye(2, dtype='float32')[self.labels]

        # Initialize drop index list for cut event indeces
        drop_index_list = []

        # Loop through input dict features
        for key, data_feature in self.input_dict.items():

            # We either have constituent data (awkward arrays with variable length per jet)
            # or high level data (arrays with equal length per jet). Treat these cases differently
            typestr = str(ak.type(data_feature))

            # Constituent case
            if 'var' in typestr:

                # Pad jets with less than the maximum number of constituents
                # with 0's, and trucate jets with more than the maximum.
                df_zero = ak.pad_none(data_feature, self.max_constits, axis=1, clip=True)
                df_zero = ak.fill_none(df_zero, 0)

                # Set ak_dict entry to df_zero, alter in place to save memory
                self.input_dict[key] = df_zero

            # High level case
            else:

                # Some high level variables come with exit codes, set to -999
                # Drop these events by tracking indeces of events with these codes
                exit_indeces = np.asarray(data_feature == -999).nonzero()[0]
                drop_index_list.append(exit_indeces)

                # For events that are left, we want to standardize inputs
                # Subtract off mean and divide by standard deviation
                # (Note -999 events will get shifted but this is fine since we already found
                # their indeces)
                good_events = data_feature[data_feature != -999]
                mean = np.mean(good_events)
                stddev = np.std(good_events)
                self.input_dict[key] = (data_feature - mean) / stddev

        # Now combine features in input_dict into a rectangular array using
        # numpy.stack. First convert to numpy
        np_dict = {key: ak.to_numpy(df).astype('float32') for key, df in self.input_dict.items()}

        # Call numpy.stack
        data = np.squeeze(np.dstack(tuple(np_dict.values())))
        del np_dict

        # Preprocessing can introduce NaNs and Infs, set these to 0 here
        np.nan_to_num(data, copy=False)

        # Turn the extras into numpy arrays
        extras = [ak.to_numpy(df).astype('float32') for df in self.extra_dict.values()]

        # Make cut on events, removing those whose indeces occur in drop_index if there are any
        if len(drop_index_list):
            drop_index = np.unique(np.concatenate(drop_index_list))
            data = np.delete(data, drop_index, axis=0)
            cat_labels = np.delete(cat_labels, drop_index, axis=0)
            extras = [np.delete(df, drop_index) for df in extras]

        # Now package everything into a list and return
        data_list = [data, cat_labels] + extras

        return data_list


    def kfold_splits(self, num_folds=5, fold=1, **kwargs):
        """ kfold_splits - This function will call the np_arrays function
        to generate numpy arrays of the data, but then also break the data
        into k folds, and return the training and validation datasets. The
        number of folds and the fold this training run will use will be given
        as keyword arguments.

        Arguments:
            num_folds (int) - The number of folds to break data into
            fold (int) - The fold designated as validation data

        Returns:

        """

        # First we need to make numpy arrays
        data_arrs = self.np_arrays()

        # Prepare train and valid arrs to accept splits
        train_arrs = []
        valid_arrs = []

        # Now we loop through data arrs and use numpy's array_split function
        for arr in data_arrs:

            # First assert that we have no NaNs or -999s
            assert not np.any(np.isnan(arr))
            assert not np.any(arr == -999)

            # Make split
            split_arr = np.array_split(arr, num_folds, axis=0)
            valid_arrs.append(np.squeeze(split_arr.pop(fold)))
            train_arrs.append(np.squeeze(np.concatenate(split_arr, axis=0)))

        # Print some information about split
        train_events = train_arrs[0].shape[0]
        valid_events = valid_arrs[0].shape[0]
        print("Using fold", fold+1, "of", num_folds)
        print("Training events: ", train_events)
        print("Valid events: ", valid_events)
        print("Total events: ", train_events + valid_events)

        # Return the training and validation data
        return train_arrs, valid_arrs


    def model_shape(self, dataset, net_type='dnn', **kwargs):
        """ model_shape - This function will take in a dataset, supplied as the
        first element of the list of numpy arrays generated by the function
        np_arrays, and make model dependent changes so that the output of the
        data handler can be passed directly into the keras fit function.

        Arguments:
            dataset (array) - This is the array of data to reshape
            net_type (str) - The type of network we are building

        Returns:
            (array) - The reshaped dataset, sometimes reshaped into 2 different
                arrays in the case of the EFN
        """

        # First just find the shape of our data
        events = dataset.shape[0]
        sample_shape = self.sample_shape()

        # Now condition on the type of model we are building
        if 'dnn' in net_type:

            # Input shape for all dnn networks can be found from sample shape
            input_shape = np.prod(sample_shape)

            # Now we do reshaping
            shaped_data = dataset.reshape((events, input_shape))

        elif net_type == 'efn':

            # For EFNs, we need to split pT and angular information
            shaped_data = [dataset[:,:,0], dataset[:,:,1:3]]

        elif net_type == 'pfn':

            # For PFNs, we don't need to do anything!
            shaped_data = dataset

        elif net_type == 'resnet':

            # For image based network, we need to bin angular information into 224x224 images
            # And assign pixel intensity as the sum of pT that fall in that bin
            # Numpy's histogram2d function does this automatically, but we'll need to make
            # an event loop
            shaped_data = np.zeros((dataset.shape[0], 224, 224, 3))

            for i, jet in enumerate(dataset):
                hist, xbins, ybins = np.histogram2d(dataset[i,:,1], 
                                                    dataset[i,:,2], 
                                                    bins=224, 
                                                    range=[[-np.pi, np,pi], [-np.pi, np,pi]], 
                                                    weights=dataset[i,:,0])
                rgb_patch = np.repeat(hist[..., np.newaxis], 3, -1)
                shaped_data[i,:,:,:] = rgb_patch

        else:
            raise ValueError("Network type not recognized!")

        # Return results
        return shaped_data


    def sample_shape(self):
        """ sample_shape - Return the shape of each sample.

        Arguments:
            None

        Returns:
            (tuple): The shape of each sample as a tuple
        """

        if self.max_constits:
            return (self.max_constits, len(self.input_dict))
        else:
            return (len(self.input_dict),)

    def get_data(self, **kwargs):
        """ get_data - Umbrella function for getting data for training a model
        using Keras. Model type, number of folds, and particular training fold
        can be set as a keyword arguments.
        """

        # Make k-fold splitted numpy arrays
        train_arrs, valid_arrs = self.kfold_splits(**kwargs)

        # Make model dependent reshape of both train data and valid data
        train_arrs[0] = self.model_shape(train_arrs[0], **kwargs)
        valid_arrs[0] = self.model_shape(valid_arrs[0], **kwargs)

        # Return train arrays and valid arrays
        return train_arrs, valid_arrs


if __name__ == '__main__':

    # Test jet image generation
    filepath = "/pub/kgreif/samples/sample_1p5M_nbpt_test.root"
    input_branches = ['fjet_sortClusNormByPt_pt', 'fjet_sortClusCenterRotFlip_eta',
                      'fjet_sortClusCenterRot_phi', 'fjet_sortClusNormByPt_e']
    dhandler = DataHandler(filepath, "FlatSubstructureJetTree", input_branches,
                           extras=None, max_constits=80)

    train_arrs, valid_arrs = dhandler.get_data(
        net_type='resnet',
        num_folds=5,
        fold=0  # -1 to make number into array index
    )

    jet_img = train_arrs[0][0,:,:]
    plt.imshow(jet_img)
    plt.show()
