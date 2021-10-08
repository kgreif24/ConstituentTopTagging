""" to_hdf.py - This script will read in data from a .root file using uproot's
iterate feature, and save the data chunk by chunk into a hf file. The idea is
that it will be easy to write a python generator for loading data from h5py
while we can't really do this using uproot since uproot doesn't support slicing
by indeces (except for lazy arrays, but these are very slow).

Author: Kevin Greif
python3
Last updated 10/7/21
"""

import h5py
import uproot
import awkward as ak
import numpy as np
import argparse


# Parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', default=None, type=str,
                    help='Path of file to convert to hdf5')
parser.add_argument('-n', '--numConstits', default=80, type=int,
                    help='Number of constituents to save into hdf5 file')
parser.add_argument('--name', default='events.hdf5', type=str,
                    help='Name of hdf5 file to produce')
args = parser.parse_args()

#  Setup the branches we want to pull from .root file
constit_branches = ['fjet_sortClusNormByPt_pt', 'fjet_sortClusCenterRotFlip_eta',
                  'fjet_sortClusCenterRot_phi', 'fjet_sortClusNormByPt_e']
hl_branches = ['fjet_Tau1_wta', 'fjet_Tau2_wta', 'fjet_Tau3_wta', 'fjet_Split12',
               'fjet_Split23', 'fjet_ECF1', 'fjet_ECF2', 'fjet_ECF3', 'fjet_C2',
               'fjet_D2', 'fjet_Qw']
weight_branch = ['fjet_match_weight_pt']
signal_branch = ['fjet_signal']
all_branches = constit_branches + hl_branches + weight_branch + signal_branch

# Find the total number of events, will determine how we initialize hdf5 file
events = uproot.open(args.file + ":FlatSubstructureJetTree")
num_events = len(events[signal_branch[0]].array())

# Now we build hdf5 file
f = h5py.File(args.name, 'w')
constits = f.create_dataset("constits", (num_events, args.numConstits, len(constit_branches)), dtype='f4')
images = f.create_dataset("images", (num_events, args.numConstits, 2), dtype='i4')
hl = f.create_dataset("hl", (num_events, len(hl_branches)), dtype='f4')
weights = f.create_dataset("weights", (num_events,), dtype='f4')
labels = f.create_dataset("labels", (num_events, 2), dtype='i4')

# Initialize counter to keep track of where we are in file
start_index = 0

# Loop through .root file using uproot iterate
for batch in events.iterate(step_size='1 GB', filter_name=all_branches):

    # Find number of events in this batch
    num_batch = len(batch[signal_branch[0]])
    end_index = start_index + num_batch

    # Now need to perform processing for all of the data in this chunk
    # Start with constituents, looping through branches
    for i, branch_name in enumerate(constit_branches):

        # Zero pad data
        branch = batch[branch_name]
        b_zero = ak.pad_none(branch, args.numConstits, axis=1, clip=True)
        b_zero = ak.to_numpy(ak.fill_none(b_zero, 0))

        # Preprocessing can introdcue NaNs, set these to 0 here
        np.nan_to_num(b_zero, copy=False)
        assert not np.isnan(b_zero).any()

        # Write b_zero to hdf5 file
        constits[start_index:end_index,:,i] = b_zero

        # Now use np.digitize to find histogram bins for eta/phi
        if (i == 1 or i == 2):
            bins = np.arange(-np.pi, np.pi, 224)
            binned_branch = np.digitize(b_zero, bins)
            images[start_index:end_index,:,i-1] = binned_branch

    # Now deal with hl variables, assuming -999 exit codes have been dealt with
    # in data slim step, along with standardization
    for i, branch_name in enumerate(hl_branches):

        # We just need to write events to hdf5
        hl[start_index:end_index, i] = batch[branch_name]

    # Now deal with weights, just write in data
    weights[start_index:end_index] = batch[weight_branch[0]]

    # Now deal with labels, need to convert to 1-hot encoding
    vec_labels = ak.to_numpy(batch[signal_branch[0]])
    cat_labels =  np.eye(2, dtype='float32')[vec_labels]
    labels[start_index:end_index,:] = cat_labels

    # Finish by making end_index into new start_index
    start_index = end_index
    print("Through jet", start_index, "of", num_events)

# With hdf5 file written, we just close file and end the script
f.close()
