""" processing_utils.py - This program defines utility functions
that will be used to process data in the root2hdf.py script.

Author: Kevin Greif
python3
Last updated 11/5/21
"""


import numpy as np
import uproot
import h5py
import awkward as ak
import hep_ml.reweight as reweight


def find_raw_len(filename):
    """ find_len - Take in a path to a .root file and returns the number
    of events in that file.

    Arguments:
    filename (string) - The path to the file (including tree name)

    Returns
    (int) - The number of events
    """

    events = uproot.open(filename)
    return ak.num(events['fjet_pt'], axis=0)

def find_cut_len(filename, cuts):
    """ find_cut_len - Take in a path to a .root file and returns the number of 
    events in that file that will pass a set of cuts.

    Arguments:
    filename (string) - The path to the file (including tree name)
    cuts (string) - Cuts to apply to filter events

    Returns
    (int) - The number of events in file that will pass cuts
    """

    events = uproot.open(filename)
    arrays = events.arrays("fjet_pt", cut=cuts)
    return ak.num(arrays['fjet_pt'], axis=0)

def find_h5_len(filename):
    """ find_h5_len - Take in a path to a .h5 file and returns the length of 
    the labels dataset.

    Arguments:
    filename (string) - The path to the file
    
    Returns
    (int) - The length of the labels dataset
    """
    f = h5py.File(filename, 'r')
    return f['labels'].shape[0]


def flat_weights(pt, n_bins=200):
    """ flat_weights - This function will use the hepml reweight function
    to calculate weights that flatten the pT distribution passed in as a numpy
    array pt. This reweighting is done separately for signal/background, so we
    don't need to have both together to do this reweighting.

    Arguments:
    pt (array) - A numpy array containing jet pT
    n_bins (int) - The number of bins to use in the reweighting

    Returns:
    (array) - The array of weights that flattens the pT spectrum.
    """

    # Initialize array of ones
    weights = np.ones(len(pt))

    # Get range of pT spectrum and sample uniform distribution over this range
    ptmin, ptmax = pt.min(), pt.max()
    rng = np.random.default_rng()
    target = rng.uniform(low=ptmin, high=ptmax, size=len(pt))

    # Fit reweighter to uniform distribution
    reweighter = reweight.BinsReweighter(n_bins=n_bins, n_neighs=3)
    reweighter.fit(pt, target=target)

    # Predict new weights
    weights = reweighter.predict_weights(pt)
    weights /= weights.mean()

    return weights


def calc_weights(file, weight_func):
    """ calc_weights - This function calculates weights to adjust the pT spectrum of 
    the h5 file passed in as arguments. Applies the weight calculation function
    given by weight_func. This function takes pt as an argument and returns weights.

    Arguments:
    file (obj) - The file to calculate weights for, must be writable
    weight_func (function) - The function used to calculate weights. Must take in a 
    vector of jet pt and return jet weights.

    Returns:
    None
    """

    # Pull pt and other info from file
    pt_name = file.attrs.get("pt")[0]
    num_jets = file.attrs.get("num_jets")
    pt = file[pt_name][:]

    # Calculate weights
    weights = weight_func(pt)

    # Create new dataset in file
    weight_shape = (num_jets,)
    weight_data = file.create_dataset("weights", shape=weight_shape, dtype='f4')
    weight_data[:] = weights


def send_data(file_list, target):
    """ send_data - This function takes in a list of .h5 files, 
    and then takes each dataset in the files and writes them to target file.
    This is a method for "adding" h5 files while also shuffling their contents.

    Arguments:
    file_list (list) - A list containing strings giving the path of files to add
    target (obj) - h5 file object for the target file. Assumes we have write permissions
    and that the dataset structure of target is exactly the same as the source files.

    Returns:
    None
    """

    # Start counter to keep track of write index in target file
    start_index = 0

    # Loop through file list
    for file_name in file_list:
        print("Now processing file:", file_name)

        # Open file
        file = h5py.File(file_name, 'r')
        num_file_jets = file.attrs.get("num_jets")
        stop_index = start_index + num_file_jets

        # Extract keys for all datasets
        file_keys = file.keys()

        # Get random seed for our shuffles
        rng_seed = np.random.default_rng()
        seed = rng_seed.integers(1000)

        # Loop through each of the datasets
        for key in file_keys:
            print("Processing branch", key)
            
            # Extract all data from file
            dataset = file[key][...]

            # Initialize new rng using seed and shuffle dataset
            rng = np.random.default_rng(seed)
            rng.shuffle(dataset, axis=0)

            # Write shuffled data to target file
            target[key][start_index:stop_index,...] = dataset

        # Increment counters and close file
        start_index = stop_index
        file.close()

    # End by printing summary of how many jets were written to file
    print("We wrote", stop_index, "jets to target file")
    target.attrs.modify("num_jets", stop_index)



def standardize(file, calc_events=1000000):
    """ shuffle - Takes in an h5py file object and standardizes each of the datasets
    that are in the "hl" attribute. Assumes each of these datasets are one dimensional.
    This function works in place so file objec must have read/write permissions.

    Arguments:
    file (obj) - The h5 file object in which we will standardize hl variables
    calc_events (int) - The number of events we will consider in calculating mean/stddev

    Returns:
    None
    """

    # Correct calc_events if needed
    if calc_events > file['labels'].shape[0]:
        calc_events = file['labels'].shape[0]

    # Pull hl attribute
    hl_list = file.attrs.get("hl")

    # Loop through hl variables
    for hl_var in hl_list:
        print("Now standardizing", hl_var)

        # Pull data
        variable = file[hl_var][:]
        
        # For variables with large magnitudes (ECFs) divide by a large value to head off
        # overflows in calculating mean and stddev
        if hl_var == 'fjet_ECF3':
            variable /= 1e10
        elif hl_var == 'fjet_ECF2':
            variable /= 1e6

        # Pull calculation variables
        calc_variable = variable[:calc_events]
        
        # Calculate mean and std deviation
        mean = calc_variable.mean()
        stddev = calc_variable.std()

        # Standardize and write
        std_variable = (variable - mean) / stddev
        file[hl_var][:] = std_variable
