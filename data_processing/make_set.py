""" make_set.py - This program will generate an .h5 file ready for training,
starting from two sets of intermediates files, one of which will serve as
signal and the other background. It can also calculate training weights,
and perform simple standardizations.

Author: Kevin Greif
Last updated 7/5/22
python3
"""

import glob
import h5py
import numpy as np

import processing_utils as pu


# We start by finding a list of all intermediate files sitting in intermediate
# file folder.
n_files = 16
search_stem = "./dataloc/intermediates_raw/"
search_string = search_stem + "*.h5"
file_list = glob.glob(search_string)[:n_files]
print("Will process files", file_list)

# Find the lengths of each of these files
file_lengths = np.array([pu.find_h5_len(file) for file in file_list])
total_jets = np.sum(file_lengths)

# Make train / test split by partitioning file_list
split_index = 2
print(split_index)
test_list = file_list[:split_index]
train_list = file_list[split_index:]
n_test = np.sum(file_lengths[:split_index])
n_train = np.sum(file_lengths[split_index:])
print("N train jets:", n_train)
print("N test jets:", n_test)

# Make new h5 files for training/testin. Will have stacked data for easy use in
# network trainng
print("Building train/test h5 files...")
f_train = h5py.File('./dataloc/train_public.h5', 'w')
f_test = h5py.File('./dataloc/test_public.h5', 'w')
f_ref = h5py.File(file_list[0], 'r')
constit_branches = f_ref.attrs.get('constit')
hl_branches = f_ref.attrs.get('hl')
jet_branches = f_ref.attrs.get('pt')

# Loop to build datasets for train/test files
for file, num_jets in zip([f_train, f_test], [n_train, n_test]):

    # Constituents
    constit_shape = (num_jets, 200)
    for var in constit_branches:
        file.create_dataset(var, constit_shape, dtype='f4')

    # HL variables
    hl_shape = (num_jets,)
    for var in hl_branches:
        file.create_dataset(var, hl_shape, dtype='f4')

    # Jet 4 vector
    for var in jet_branches:
        file.create_dataset(var, hl_shape, dtype='f4')

    # Lables
    oth_shape = (num_jets,)
    file.create_dataset('labels', oth_shape, dtype='i4')

    # Attributes
    file.attrs.create("num_jets", num_jets)
    file.attrs.create("num_cons", len(constit_branches))
    file.attrs.create("num_hl", len(hl_branches))
    file.attrs.create("num_jet_features", len(jet_branches))
    file.attrs.create("jet", jet_branches)
    file.attrs.create("constit", constit_branches)
    file.attrs.create("hl", hl_branches)

# Send data to train/test
print("\nBegin processing data")

pu.unstacked_send(train_list, f_train)
pu.unstacked_send(test_list, f_test)

# Calculate weights
print("\nCalculating weights")
pu.calc_weights(f_train, pu.match_weights)
pu.calc_weights(f_test, pu.match_weights)

print("Finished buildling train/test files")
