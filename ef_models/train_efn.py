""" train_efn.py - This script will build, train, and evaluate an energy flow
network, implemented with the energyflow package. It will make use of the
DataDumper class to quickly generate np arrays of the jet data.

Author: Kevin Greif
7/5/21
python3
"""

import sys, os
sys.path.append('/data/homezvol0/kgreif/toptag/ML-TopTagging')  # Need classes stored in home dir
import argparse

import energyflow as ef
from energyflow.archs import EFN
import tensorflow as tf
import sklearn.metrics as metrics
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('~/mattyplotsalot/allpurpose.mplstyle')

from data_dumper import DataDumper


# If GPU is available, print message
print("\nStart EFN training script...")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


########################## Parse Arguments ###########################

parser = argparse.ArgumentParser()

parser.add_argument('-N', '--numEpochs', default=100, type=int,
                    help='Number of epochs')
parser.add_argument('-b', '--batchSize', default=100, type=int,
                    help='Batch size')
parser.add_argument('--maxConstits', default=80, type=int,
                    help='Number of constituents to include per event')
parser.add_argument('-o', '--checkDir', default='./checkpoints', type=str,
                    help='Stem of file name at which to save checkpoints')
parser.add_argument('--nodes', default=100, type=int,
                    help='Number of nodes/layer to use in phi, f networks')
parser.add_argument('--latent', default=128, type=int,
                    help='Dimension of the latent space to use')
args = parser.parse_args()

###################### Set parameters for training #######################

# Network parameters
Phi_sizes = (args.nodes, args.nodes, args.latent)
F_sizes = (args.nodes, args.nodes, args.nodes)

# Training parameters
num_epoch = args.numEpochs
batch_size = args.batchSize
checkpoint_filepath = args.checkDir

# Data parameters
filepath = "/data/homezvol0/kgreif/toptag/samples/sample_nr1M.root"
constit_branches = ['fjet_sortClusNormByPt_pt', 'fjet_sortClusCenterRotFlip_eta',
                    'fjet_sortClusCenterRot_phi', 'fjet_sortClusNormByPt_e']
extra_branches = ['fjet_match_weight_pt', 'fjet_pt']
my_max_constits = args.maxConstits

############################# Process Data ################################

# Now build dds and use them to plot all branches of interest
print("Building data objects...")
dd_train = DataDumper(filepath, "train", constit_branches, "fjet_signal",
                      extras=extra_branches, max_constits=my_max_constits)
dd_valid = DataDumper(filepath, "valid", constit_branches, "fjet_signal",
                      extras=extra_branches, max_constits=my_max_constits)
dd_train.plot_branches(constit_branches + extra_branches, directory="./plots/")

# Now make np arrays out of the data
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
raw_labels_train = labels_train[:,1]
labels_valid = valid_arrs[1]
raw_labels_valid = labels_valid[:,1]
weight_train = train_arrs[2]
weight_valid = valid_arrs[2]

# Delete arrays to save memory
del train_arrs
del valid_arrs

print("Shapes of input arrays and label array: ")
print(np.shape(z_train))
print(np.shape(p_train))
print(np.shape(labels_train))

# Verify that there are no NaNs in data, labels, or weights
assert not np.any(np.isnan(z_train))
assert not np.any(np.isnan(p_train))
assert not np.any(np.isnan(labels_train))
assert not np.any(np.isnan(weight_train))

############################# Build EFN ##################################

# Build efn architecture
print("\nBuilding EFN model...")
efn = ef.archs.EFN(input_dim=2,
                   Phi_sizes=Phi_sizes,
                   F_sizes=F_sizes,
                   filepath=checkpoint_filepath,
                   patience=50,
                   earlystop_opts={'monitor': 'val_loss'})

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
plt.savefig('./plots/initial_output.png', dpi=300)

############################### Train EFN #################################

# Train model by calling fit
print("Training EFN...")
train_hist = efn.fit([z_train, p_train], labels_train,
                     epochs=num_epoch,
                     batch_size=batch_size,
                     sample_weight=weight_train,
                     validation_data=([z_valid, p_valid], labels_valid, weight_valid),
                     verbose=1)

# Plot losses
plt.clf()
plt.plot(train_hist.history['loss'], label='Training')
plt.plot(train_hist.history['val_loss'], label='Validation')
plt.title("Loss for EFN training")
plt.legend()
plt.ylabel("Crossentropy loss")
plt.xlabel("Epoch")
plt.savefig("./plots/loss.png", dpi=300)

############################# Evaluate EFN #################################

# Get predictions on validation data (don't have test yet)
print("\nEvaluate EFN...")
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
plt.savefig("./plots/final_output.png", dpi=300)

# Get ROC curve and AUC
fpr, tpr, thresholds = metrics.roc_curve(raw_labels_valid, preds[:,1])
auc = metrics.roc_auc_score(raw_labels_valid, preds[:,1])
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
plt.savefig("./plots/roc.png", dpi=300)
