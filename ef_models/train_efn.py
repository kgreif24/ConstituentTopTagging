""" train_efn.py - This script will build, train, and evaluate an energy flow
network, implemented with the energyflow package. It will make use of the
DataDumper class to quickly generate np arrays of the jet data.

Author: Kevin Greif
7/5/21
python3
"""

import sys, os
sys.path.append('..')  # Need classes stored in directory above.

import energyflow as ef
from energyflow.archs import EFN
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('~/Documents/General Programming/mattyplotsalot/allpurpose.mplstyle')

from data_dumper import DataDumper


###################### Set parameters for training #######################

# Network parameters
Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)

# Training parameters
num_epoch = 5
batch_size = 500
checkpoint_filepath = "./training/test"

# Data parameters
# filepath = "/data/homezvol0/kgreif/toptag/samples/sample_1M.root"
filepath = "../../Data/sample_1M.root"
constit_branches = ['fjet_sortClusStan_pt', 'fjet_sortClusCenterRotFlip_eta',
                    'fjet_sortClusCenterRot_phi', 'fjet_sortClusStan_e']
extra_branches = ['fjet_match_weight_pt', 'fjet_pt']
my_max_constits = 80

############################# Process Data ################################

# Now build dds and make numpy arrays
print("Building data objects...")
dd_train = DataDumper(filepath, "train", constit_branches, "fjet_signal",
                      extras=extra_branches, max_constits=my_max_constits)
dd_valid = DataDumper(filepath, "valid", constit_branches, "fjet_signal",
                      extras=extra_branches, max_constits=my_max_constits)
train_arrs = dd_train.np_arrays()
valid_arrs = dd_valid.np_arrays()

# Delete dds to save memory
del dd_train
del dd_valid

# Now split up data array into correct inputs for efn
print("\nSplitting data arrays...")
z_train = train_arrs[0][:,:,0]
z_valid = valid_arrs[0][:,:,0]
p_train = train_arrs[0][:,:,1:3]
p_valid = valid_arrs[0][:,:,1:3]
labels_train = train_arrs[1]
labels_valid = valid_arrs[1]
weight_train = train_arrs[2]
weight_valid = valid_arrs[2]

print(np.shape(z_train))
print(np.shape(p_train))
print(np.shape(labels_train))
print(np.shape(labels_train[:,1]))

############################# Build EFN ##################################

# Build efn architecture
print("\nBuilding EFN model...")
efn = ef.archs.EFN(
    input_dim=2,
    Phi_sizes=Phi_sizes,
    F_sizes=F_sizes,
    compile_opts={'loss_weights': weight_train},
    filepath=checkpoint_filepath,
    patience=5)

# Immediately evaluate model on validation set to get a baseline
preds = efn.predict([z_valid, p_valid], batch_size=batch_size)

# Make a histogram of network output, separated into signal/background
preds_sig = preds[labels_valid[:,1] == 1]
preds_bkg = preds[labels_valid[:,1] == 0]
hist_bins = np.linspace(0, 1.0, 100)

plt.clf()
plt.hist(preds_sig[:,1], bins=hist_bins, alpha=0.5, label='Signal')
plt.hist(preds_bkg[:,1], bins=hist_bins, alpha=0.5, label='Background')
plt.legend()
plt.ylabel("Counts")
plt.xlabel("EFN output")
plt.title("EFN output over validation set")
plt.show()

############################### Train EFN #################################

# Train model by calling fit
train_hist = efn.fit([z_train, p_train], train_arrs[1],
    epochs=num_epoch,
    batch_size=batch_size,
    validation_data=([z_valid, p_valid], valid_arrs[1]),
    verbose=1)

############################# Evaluate EFN #################################

# Get predictions on validation data (don't have test yet)
preds = efn.predict([z_valid, p_valid], batch_size=batch_size)

# Get ROC curve and AUC
fpr, tpr, thresholds = roc_curve(valid_arrs[1][:,1], preds[:,1])
auc = roc_auc_score(valid_arrs[1][:,1], preds[:,1])

# Find background rejection at tpr = 0.5, 0.8 working points
wp_p5 = np.argmax(tpr > 0.5)
wp_p8 = np.argmax(tpr > 0.8)

# Finally print information on model performance
print("Background rejection at 0.5 signal efficiency: ", fprinv[wp_p5])
print("Background rejection at 0.8 signal efficiency: ", fprinv[wp_p8])
print("AUC score: ", auc)
