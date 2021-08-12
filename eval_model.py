""" eval_model.py - This script will evaluate a pretrained network of type
DNN, EFN, or PFN. It will make use of the DataHandler class to quickly
generate np arrays of the jet data.

Author: Kevin Greif
Last updated 8/11/21
python3
"""

import sys, os
import argparse

import energyflow as ef
from energyflow.archs import EFN
import tensorflow as tf
import sklearn.metrics as metrics
import numpy as np

import matplotlib
# Matplotlib import setup to use non-GUI backend, comment for interactive
# graphics
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('~/mattyplotsalot/allpurpose.mplstyle')

from data_handler import DataHandler


# If GPU is available, print message
print("\nStart model evaluation script...")
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))


########################## Parse Arguments ###########################

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--batchSize', default=100, type=int,
                    help='Batch size')
parser.add_argument('--maxConstits', default=80, type=int,
                    help='Number of constituents to include per event')
parser.add_argument('--file', default=None, type=str,
                    help='File name of network checkpoint')
args = parser.parse_args()

###################### Set parameters for evaluation #######################

# Net parameters
batch_size = args.batchSize

# Data parameters
filepath = "~/toptag/samples/csauer_initial.root"
constit_branches = ['fjet_sortClusNormByPt_pt', 'fjet_sortClusCenterRotFlip_eta',
                    'fjet_sortClusCenterRot_phi', 'fjet_sortClusNormByPt_e']
extra_branches = ['fjet_training_weight_pt', 'fjet_pt']
max_constits = args.maxConstits

############################# Process Data ################################

# Now build dhs and use them to plot all branches of interest
print("Building data object...")
dh_valid = DataHandler(filepath, "valid", constit_branches,
                       extras=extra_branches, max_constits=max_constits)

# Figure out sample shape
sample_shape = tuple([batch_size]) + dh_valid.sample_shape()

# Now make np arrays out of the data
print("\nBuilding data arrays...")
valid_arrs = dh_valid.np_arrays()

# Delete dhs to save memory
del dh_valid

# Split data arrays common to all model types
print("\nSplitting data arrays...")
labels_valid = valid_arrs[1]
weight_valid = valid_arrs[2]
valid_events = len(weight_valid)

# Verify that there are no NaNs in data, labels, or weights
assert not np.any(np.isnan(valid_arrs[0]))
assert not np.any(np.isnan(labels_valid))
assert not np.any(np.isnan(weight_valid))

############################# Load Model ##################################
print("\nLoading model...")

# Process data for EFN (REFACTOR!!)
valid_data = [valid_arrs[0][:,:,0], valid_arrs[0][:,:,1:3]]
del valid_arrs

# Load model
model = tf.keras.models.load_model(args.file)
model.summary()

########################## Evaluation ################################

print("\nEvaluation...")
preds = model.predict(valid_data, batch_size=batch_size)

# Make a histogram of network output, separated into signal/background
preds_sig = preds[labels_valid[:,1] == 1]
preds_bkg = preds[labels_valid[:,1] == 0]
hist_bins = np.linspace(0, 1.0, 100)

plt.clf()
plt.hist(preds_sig[:,1], bins=hist_bins, alpha=0.5, label='Signal')
plt.hist(preds_bkg[:,1], bins=hist_bins, alpha=0.5, label='Background')
plt.legend()
plt.ylabel("Counts")
plt.xlabel("Model output")
plt.title("Model output over validation set")
plt.savefig('./outfiles/initial_output.png', dpi=300)

# Get ROC curve and AUC
fpr, tpr, thresholds = metrics.roc_curve(labels_valid[:,1], preds[:,1])
auc = metrics.roc_auc_score(labels_valid[:,1], preds[:,1])
fprinv = 1 / fpr

# Find background rejection at tpr = 0.5, 0.8 working points
wp_p5 = np.argmax(tpr > 0.5)
wp_p8 = np.argmax(tpr > 0.8)

# Finally print information on model performance
print("Background rejection at 0.5 signal efficiency: ", fprinv[wp_p5])
print("Background rejection at 0.8 signal efficiency: ", fprinv[wp_p8])
print("AUC score: ", auc)

# Make an inverse roc plot. Take 1/fpr and plot this against tpr
plt.clf()
plt.plot(tpr, fprinv)
plt.yscale('log')
plt.ylabel('Background rejection')
plt.xlabel('Signal efficiency')
plt.savefig("./outfiles/roc.png", dpi=300)
