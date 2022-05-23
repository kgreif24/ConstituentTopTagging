""" r2h_raw.py - This script does the same thing as root2hd5.py, but uses
the raw preprocessing, which does not apply consituent pre-processing.

Author: Kevin Greif
python3
Last updated 3/3/22
"""

import sys, os
import argparse
import random

import h5py
import uproot
import awkward as ak
import numpy as np

import processing_utils
import preprocessing


############### Parse command line ################

parser = argparse.ArgumentParser()
parser.add_argument('--signal', action='store_true')
parser.add_argument('--background', action='store_true')
args = parser.parse_args()
if (args.signal and not args.background):
    svb_flag = True
elif (not args.signal and args.background):
    svb_flag = False
else:
    raise ValueError("Command line args are non-sensical, try harder!")

############### Conversion Setup ##################

#  Setup the branches we want to pull from .root file and write to target files
s_constit_branches = ['fjet_clus_pt', 'fjet_clus_eta',
                      'fjet_clus_phi', 'fjet_clus_E']
hl_branches = ['fjet_Tau1_wta', 'fjet_Tau2_wta', 'fjet_Tau3_wta', 'fjet_Tau4_wta',
               'fjet_Split12', 'fjet_Split23', 'fjet_ECF1', 'fjet_ECF2', 'fjet_ECF3', 
               'fjet_C2', 'fjet_D2', 'fjet_Qw', 'fjet_L2', 'fjet_L3', 'fjet_ThrustMaj']
t_constit_branches = ['fjet_clus_eta', 'fjet_clus_phi', 'fjet_clus_pt', 'fjet_clus_E']
pt_branch = ['fjet_pt', 'fjet_eta', 'fjet_phi', 'fjet_m']
label_branch = ['labels']
source_branches = s_constit_branches + hl_branches + pt_branch
target_branches = t_constit_branches + hl_branches + pt_branch + label_branch

# Setup up options for pulling constituents
max_constits = 200

# Set number of files we are going to break jets into
n_files = 36

# Setup cuts we want to make on jets
common = "(abs(fjet_truthJet_eta)<2.0) & (fjet_truthJet_pt/1000.>350.) & (fjet_numConstituents > 3) & (fjet_m/1000.>40.)"
signal = (" & (abs(fjet_truth_dRmatched_particle_flavor)==6) &"
          " (abs(fjet_truth_dRmatched_particle_dR)<0.75) &"
          " (abs(fjet_truthJet_dRmatched_particle_dR_top_W_matched)<0.75) &"
          " (fjet_ungroomed_truthJet_m/1000.>140.) &"
          " (fjet_truthJet_ungroomedParent_GhostBHadronsFinalCount>=1) &"
          " (fjet_ungroomed_truthJet_Split23/1000.>exp(3.3-6.98e-4*fjet_ungroomed_truthJet_pt/1000.))")

# Set some more arguments depending on whether we are running signal or background
if svb_flag:
    total_cuts = common + signal
    listname = "./dat/ZprimeTTSamples.list"
    rw_type = 'w'
else:
    total_cuts = common
    listname = "./dat/DijetSamples.list"
    rw_type = 'a'

# Set directory for target files
tar_dir = './dataloc/intermediates_raw/'

# Finally, since we have a certain number of good jets in signal files, this will be the number
# of jets we ultimately want to include from both sig and background
total_events = 22375114
# total_events = 500000


###############  Pull information on jets before processing
print("Gathering info on source files...")

# Open .list file to get names of .root files
listfile = open(listname, "r")
files = listfile.readlines()
files = [f.rstrip() + ":FlatSubstructureJetTree" for f in files]

# Because files come from different pt slices, we need to pull a representative
# sample of each file for our train/test data sets. Find number of jets we expect
# to pull from each file
raw_file_events = np.array([processing_utils.find_cut_len(name, total_cuts) for name in files])
raw_events = np.sum(raw_file_events)
print("We have", raw_events, "jets in total")
print("We wish to keep", total_events, "of these jets")

# If statement to catch case where we request more jets than we have
if total_events > raw_events:
    total_events = raw_events
    print("Only have", raw_events, "jets, so keep this many")

# This line then calculates number of events we actually want to pull from each file
lim_file_events = np.around((raw_file_events / raw_events) * total_events).astype(int)

# Update total_events as rounding can cause us to be off by just a bit
if np.sum(lim_file_events) != total_events:
    total_events = np.sum(lim_file_events)
    print("Due to rounding we will instead keep", total_events, "of these jets")

print("Number of jets to take from each root file:")
print(lim_file_events)

# We can't know how many jets will end up in target files, so just set a max size
max_file_jets = 4000000


################ Build h5 files ####################
print("\nBuilding H5 files...")

# Initialize list for adding h5 file dictionaries
h5files = []

# Loop to create n_files h5 files with identical data set content
for file_num in range(n_files):

    # Initialize dictionary to accept file information
    filedict = {}

    # Open h5 file
    filename = tar_dir + "tt_dijet_samples_" + str(file_num) + ".h5"
    filedict["file"] = h5py.File(filename, rw_type)

    # How to deal with datasets varies by whether we are processing signal (first) or background (second)
    if svb_flag:

        # If signal we need to fresh create all datasets and put them in the file dictionary
        constits_size = (max_file_jets, max_constits)
        filedict["fjet_clus_eta"] = filedict["file"].create_dataset("fjet_clus_eta", constits_size, maxshape=constits_size, dtype='f4')
        filedict["fjet_clus_phi"] = filedict["file"].create_dataset("fjet_clus_phi", constits_size, maxshape=constits_size, dtype='f4')
        filedict["fjet_clus_pt"] = filedict["file"].create_dataset("fjet_clus_pt", constits_size, maxshape=constits_size, dtype='f4')
        filedict["fjet_clus_E"] = filedict["file"].create_dataset("fjet_clus_E", constits_size, maxshape=constits_size, dtype='f4')
        hl_size = (max_file_jets)
        for hl_var in hl_branches:
            filedict[hl_var] = filedict["file"].create_dataset(hl_var, hl_size, maxshape=hl_size, dtype='f4')
        filedict["fjet_pt"] = filedict["file"].create_dataset("fjet_pt", hl_size, maxshape=hl_size, dtype='f4')
        filedict["fjet_eta"] = filedict["file"].create_dataset("fjet_eta", hl_size, maxshape=hl_size, dtype='f4')
        filedict["fjet_phi"] = filedict["file"].create_dataset("fjet_phi", hl_size, maxshape=hl_size, dtype='f4')
        filedict["fjet_m"] = filedict["file"].create_dataset("fjet_m", hl_size, maxshape=hl_size, dtype='f4')
        filedict["labels"] = filedict["file"].create_dataset("labels", hl_size, maxshape=hl_size, dtype='i4')

        # Attribute for storing absolute number of jets written to file, and names of branches
        filedict["file"].attrs.create("num_jets", 0, dtype='i4')
        filedict["file"].attrs.create("constit", t_constit_branches)
        filedict["file"].attrs.create("hl", hl_branches)
        filedict["file"].attrs.create("pt", pt_branch)
        filedict["file"].attrs.create("label", label_branch)

    else:

        # If background, we need to retrieve all of the datasets and put them in the dictionary
        for branch in target_branches:
            filedict[branch] = filedict["file"][branch]

    # Append file dictionary to list
    h5files.append(filedict)

################ Processing Loop ###############

# Use values of h5 file attributes to find where to start writing in h5 file
start_index = np.array([targ_file['file'].attrs.get("num_jets") for targ_file in h5files])
# And initialize counter to keep track of how many new jets we write
write_events = np.zeros(n_files)

print("\nStarting processing loop...")
print("Initial write positions:", start_index)

# Now we loop through each of the files
for num_source, ifile in enumerate(files):

    print("\n\nNow processing file", ifile)

    # Open file using uproot
    events = uproot.open(ifile)

    # Start a counter to keep track of how many events we have written from file
    jets_from_file = 0

    # Iterate through the file using iterate, filtering out only branches we need
    for jet_batch, report in events.iterate(cut=total_cuts,
                                            step_size="200 MB", 
                                            filter_name=source_branches, 
                                            report=True):

        # Get jet batch and report
        print("\n", report)

        # Initialize batch data dictionary to accept all of our information
        batch_data = {}

        ##################### Constituents ##################

        # Pull and zero pad constituents
        pt = jet_batch['fjet_clus_pt']
        pt_zero = ak.pad_none(pt, max_constits, axis=1, clip=True)
        pt_zero = ak.to_numpy(ak.fill_none(pt_zero, 0, axis=None))

        eta = jet_batch['fjet_clus_eta']
        eta_zero = ak.pad_none(eta, max_constits, axis=1, clip=True)
        eta_zero = ak.to_numpy(ak.fill_none(eta_zero, 0, axis=None))

        phi = jet_batch['fjet_clus_phi']
        phi_zero = ak.pad_none(phi, max_constits, axis=1, clip=True)
        phi_zero = ak.to_numpy(ak.fill_none(phi_zero, 0, axis=None))

        en = jet_batch['fjet_clus_E']
        en_zero = ak.pad_none(en, max_constits, axis=1, clip=True)
        en_zero = ak.to_numpy(ak.fill_none(en_zero, 0, axis=None))

        # Get indeces to sort by increasing pt
        indeces = np.argsort(pt_zero, axis=1)

        # Sort constituents by DECREASING pT
        pt_sort = np.take_along_axis(pt_zero, indeces, axis=1)[:,::-1]
        eta_sort = np.take_along_axis(eta_zero, indeces, axis=1)[:,::-1]
        phi_sort = np.take_along_axis(phi_zero, indeces, axis=1)[:,::-1]
        en_sort = np.take_along_axis(en_zero, indeces, axis=1)[:,::-1]

        # Also make a cut on very small pT constituents (less than 100 MeV)
        small_pt = np.asarray(pt_sort < 100).nonzero()
        pt_sort[small_pt] = 0
        eta_sort[small_pt] = 0
        phi_sort[small_pt] = 0
        en_sort[small_pt] = 0

        # Send pre-processed constituents to batch_data
        batch_data["fjet_clus_eta"] = eta_sort
        batch_data["fjet_clus_phi"] = phi_sort
        batch_data["fjet_clus_pt"] = pt_sort
        batch_data["fjet_clus_E"] = en_sort

        #################### UFO 4 vector #####################

        # Loop through UFO 4 vector list
        for comp_name in pt_branch:

            # Pull data
            batch_comp = jet_batch[comp_name]

            # Send data to batch data
            batch_data[comp_name] = ak.to_numpy(batch_comp)

        ##################### High Level #################

        # Will do preprocessing once all of the data is assembled
        # However we need to look for -999 or other large negative exit codes

        # Array for storing indeces of jets to leave out of data set
        drop_list = np.array([], dtype=np.int32)

        # Loop through hl branches to find jets to skip
        for hl_var in hl_branches:

            # Find jets with -999 or large negative number for this variable
            exit_indeces = np.asarray(jet_batch[hl_var] == -999, dtype=np.int32).nonzero()[0]

            # Append to drop list
            drop_list = np.append(drop_list, exit_indeces)

            # Send hl_var to batch data
            batch_data[hl_var] = ak.to_numpy(jet_batch[hl_var])

        # Find unique elements of drop list, will use this at file write time
        print("Drop list:", drop_list)
        drop_list = np.unique(drop_list)

        # Find batch length given dropped jets
        raw_length = batch_data['fjet_clus_eta'].shape[0]
        batch_length = raw_length - len(drop_list)
        print("Raw number of jets in batch:", raw_length)
        print("Number of jets to drop:", len(drop_list))
        print("Number of jets to write:", batch_length)

        #################### Labels #####################

        # If jet has made it through cuts in loop, it is 
        # either signal or background depending on source file,
        # so just make vector of 1s or 0s.
        
        if svb_flag:
            batch_data['labels'] = np.ones(raw_length, dtype='i4')
        else:
            batch_data['labels'] = np.zeros(raw_length, dtype='i4')

        #################### Write ######################

        # If this batch will cause us to go over limit of jets to pull from this file
        # we need to truncate the batch
        if jets_from_file + batch_length >= lim_file_events[num_source]:
            overhang = jets_from_file + batch_length - lim_file_events[num_source]
            batch_length -= overhang
        else:
            overhang = 0

        # Loop over all target branches
        for branch in target_branches:

            # Drop bad jets from this data vector
            batch_data[branch] = np.delete(batch_data[branch], drop_list, axis=0)
            assert batch_data[branch].shape[0] == batch_length + overhang

            # Now we split data vector into n_files pieces using np.array_split
            # If we have an overhang, drop it here
            branch_splits = np.array_split(batch_data[branch][:batch_length,...], n_files)

            # Find length of each split and end indeces for writing
            split_lengths = [split.shape[0] for split in branch_splits]
            end_index = start_index + split_lengths

            # Loop through branch_splits and write to files
            for targ_num, (write_array, start, stop) in enumerate(zip(branch_splits, start_index, end_index)):

                # Write branch to correct h5 file with indeces given by start/stop
                h5files[targ_num][branch][start:stop,...] = write_array

        ################### Advance Indeces ##############

        # Increment write events counters
        write_events += split_lengths

        # Advance start index so we don't overwrite batch
        start_index = end_index

        # Increment jets_from_file and check if we should stop pulling from this file
        jets_from_file += batch_length
        if jets_from_file >= lim_file_events[num_source]:
            print("Have written", jets_from_file, "jets from this file")
            print("Reached limit of", lim_file_events[num_source], "jets. Skipping!")
            break

############# Set Attributes ################

# Loop through h5files and end_index, setting new value of attributes
for targ_file, stop in zip(h5files, end_index):
    targ_file["file"].attrs.modify("num_jets", stop)

################# Trim Zeros ################

# We now have all of the data in h5 targets. If we are processing background, this is
# the last step in preprocessing routine so trim datasets
if not svb_flag:
    print("\nTrimming zeros from datasets")

    # Loop through h5 target files
    for file_num, targ_file in enumerate(h5files):
        print("Now processing target file number", str(file_num))
    
        # Find appropriate size for datasets in this file
        constits_size = (end_index[file_num], max_constits)
        hl_size = (end_index[file_num],)
    
        # Loop through all target branches and resize
        for branch in t_constit_branches:
            targ_file[branch].resize(constits_size)

        hl_size_branches = hl_branches + pt_branch + label_branch
        for branch in hl_size_branches:
            targ_file[branch].resize(hl_size)

############## Finish and exit #############

# Print a summary
print("\nAt end of building files:")
print("Expected", total_events, "jets")
print("Wrote", np.sum(write_events), "jets")
print("H5 jets written breakdown:", write_events)
print("H5 jets total breakdown:", end_index)


# Finally, close the files
listfile.close()
for targ_file in h5files:
    targ_file["file"].close()

