""" eval_efn.py - This script will build and evaluate an energy flow
network, implemented with the energyflow package. It will make use of the
DataDumper class to quickly generate np arrays of the jet data.

Author: Kevin Greif
7/9/21
python3
"""

import sys, os
sys.path.append('/home/kgreif/Documents/HEPML Research/Top Tagging/ML-TopTagging')  # Need classes stored in home dir

import energyflow as ef
from energyflow.archs import EFN
import tensorflow as tf
import sklearn.metrics as metrics
import numpy as np

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('~/Documents/General Programming/mattyplotsalot/allpurpose.mplstyle')

from data_dumper import DataDumper


# If GPU is available, print message
print("\nStart EFN training script...")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


###################### Set parameters for training #######################

# Network parameters
modelpath = "/home/kgreif/Documents/HEPML Research/Top Tagging/ML-TopTagging/ef_models/training/trainEFN1Ms/run_2/checkpoints"
batch_size = 100

# Data parameters
filepath = "../../Data/sample_1M.root"
constit_branches = ['fjet_sortClusStan_pt', 'fjet_sortClusCenterRotFlip_eta',
                    'fjet_sortClusCenterRot_phi', 'fjet_sortClusStan_e']
extra_branches = ['fjet_match_weight_pt', 'fjet_pt']
my_max_constits = 80


############################# Process Data ################################

# Only need to look at validation data, trying to diagnose spiking issue

# Now build dds and make numpy arrays
print("Building data objects...")
dd_valid = DataDumper(filepath, "valid", constit_branches, "fjet_signal",
                      extras=extra_branches, max_constits=my_max_constits)
dd_valid.plot_branches(constit_branches + extra_branches, directory="./debug/")
valid_arrs = dd_valid.np_arrays()

# Delete dds to save memory
del dd_valid

# Now split up data array into correct inputs for efn
print("\nSplitting data arrays...")
z_valid = valid_arrs[0][:,:,0]
p_valid = valid_arrs[0][:,:,1:3]
labels_valid = valid_arrs[1]
raw_labels_valid = labels_valid[:,1]
weight_valid = valid_arrs[2]
pt_valid = valid_arrs[3]

# Delete arrs object to save memory
del valid_arrs

print("Shapes of input arrays and label array: ")
print(np.shape(z_valid))
print(np.shape(p_valid))
print(np.shape(labels_valid))


############################# Load EFN ##################################

print("\nLoading EFN model...")
efn = tf.keras.models.load_model(modelpath)

############################# Evaluate EFN ##############################

preds = efn.predict([z_valid, p_valid], batch_size=batch_size)
raw_preds = preds[:,1]

# Debuggin!!
botRange = 0.586
topRange = 0.596
indeces = np.where(np.logical_and(raw_preds > botRange, raw_preds < topRange))

z_spike = z_valid[indeces, ...]
p_spike = p_valid[indeces, ...]
raw_labels_spike = labels_valid[indeces]
weight_spike = weight_valid[indeces]
pt_spike = pt_valid[indeces]

n_constits_valid = np.count_nonzero(z_valid, axis=1)
n_constits_spike = np.count_nonzero(z_spike, axis=1)

plt.clf()
plt.hist(z_spike, bins=100)
plt.yscale('log')
plt.savefig("./debug/pt_spike.png")

plt.clf()
plt.hist(p_spike[:,0], bins=100)
plt.yscale('log')
plt.savefig("./debug/eta_spike.png")

plt.clf()
plt.hist(p_spike[:,1], bins=100)
plt.yscale('log')
plt.savefig("./debug/phi_spike.png")

plt.clf()
plt.hist(pt_spike, bins=100)
plt.yscale('log')
plt.savefig("./debug/pt_spike.png")

plt.clf()
plt.hist(n_constits_spike, bins=100)
plt.yscale('log')
plt.savefig("./debug/constits_spike.png")

plt.clf()
plt.hist(n_constits_valid, bins=100)
plt.yscale('log')
plt.savefig("./debug/constits_valid.png")

# Make a histogram of network output, separated into signal/background
preds_sig = preds[raw_labels_valid == 1,1]
preds_bkg = preds[raw_labels_valid == 0,1]
hist_bins = np.linspace(0, 1.0, 100)

plt.clf()
n, bins, patches = plt.hist(preds_sig, bins=hist_bins, alpha=0.5, label='Signal')
plt.hist(preds_bkg, bins=hist_bins, alpha=0.5, label='Background')
plt.legend()
plt.ylabel("Counts")
plt.xlabel("EFN output")
plt.title("EFN output over validation set")
plt.show()
