""" make_set.py - This program will shuffle a set of h5 files (keeping
each dataset in relative order), standardize HL inputs, and calculate all training
weights. It will then combine h5 files into one single h5 file that can be
directly passed to a training routine.

Author: Kevin Greif
Last updated 12/14/21
python3
"""

import glob
import h5py
import numpy as np

import processing_utils as pu


# We start by finding a list of all intermediate files sitting in intermediate
# file folder.
search_string = "./dataloc/intermediates/*.h5"
file_list = glob.glob(search_string)
print("Will process files", file_list)

# Find the lengths of each of these files
file_lengths = np.array([pu.find_h5_len(file) for file in file_list])
total_jets = np.sum(file_lengths)

# Make train / test split by partitioning file_list
ratio = 0.8
split_index = int(np.around(len(file_list) * 0.8) - 1)
train_list = file_list[:split_index]
test_list = file_list[split_index:]
n_train = np.sum(file_lengths[:split_index])
n_test = np.sum(file_lengths[split_index:])
print("N train jets:", n_train)
print("N test jets:", n_test)

# Make new h5 files for training/testing, using guide of reference file
print("Building train/test h5 files...")
f_train = h5py.File('./dataloc/train.h5', 'a')
f_test = h5py.File('./dataloc/test.h5', 'a')
f_ref = h5py.File(file_list[0], 'r')
for file, num_jets in zip([f_train, f_test], [n_train, n_test]):
    for key in f_ref.keys():
        key_shape = (num_jets,) + f_ref[key].shape[1:]
        key_type = f_ref[key].dtype
        file.create_dataset(key, key_shape, key_type)
    for key, value in f_ref.attrs.items():
        file.attrs.create(key, value)

# Send data to train/test
print("\nBegin processing data")

pu.send_data(train_list, f_train)
pu.send_data(test_list, f_test)

# Standardize hl variables
print("\nStandardizing hl variables")
pu.standardize(f_train)
pu.standardize(f_test)

# Calculate weights
print("\nCalculating weights")
pu.calc_weights(f_train, pu.flat_weights)
pu.calc_weights(f_test, pu.flat_weights)

print("Finished buildling train/test files")


