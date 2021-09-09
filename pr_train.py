""" train_model.py - This script will build, train, and evaluate a network,
implemented with the keras/energyflow packages. It will make use of the
DataDumper class to quickly generate np arrays of the jet data.

Author: Kevin Greif
Last updated 8/10/21
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
from matplotlib.colors import LogNorm
import colorcet as cc

import h5py

import pp_utils as pp


# If GPU is available, print message
print("\nStart model training script...")
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))


########################## Parse Arguments ###########################

parser = argparse.ArgumentParser()

parser.add_argument('--type', default='dnn', type=str,
                    help='Type of model to build (dnn, efn, pfn)')
parser.add_argument('--nodes', default=None, type=int, nargs='*',
                    help='DNN number of nodes in layers')
parser.add_argument('--fsizes', default=None, type=int, nargs='*',
                    help='EFN/PFN number of nodes in f layers')
parser.add_argument('--phisizes', default=None, type=int, nargs='*',
                    help='EFN/PFN number of nodes in phi layers')
parser.add_argument('-N', '--numEpochs', default=100, type=int,
                    help='Number of epochs')
parser.add_argument('-b', '--batchSize', default=100, type=int,
                    help='Batch size')
parser.add_argument('--maxConstits', default=80, type=int,
                    help='Number of constituents to include per event')
parser.add_argument('-o', '--checkDir', default='./checkpoints', type=str,
                    help='Stem of file name at which to save checkpoints')
args = parser.parse_args()

###################### Set parameters for training #######################

# Network parameters
net_type = args.type

# Training parameters
num_epoch = args.numEpochs
batch_size = args.batchSize
checkpoint_filepath = args.checkDir

# Data parameters
filepath = "/pub/kgreif/samples/prong_data/dataset_1vs3prongs_raw.h5"
max_constits = 240
preprocess_ang = False
preprocess_pt = True

############################# Process Data ################################

# Load h5py file
f = h5py.File(filepath, 'r')

# Extract data as np arrays
z_data = f['consts_threeM'][:,:,0].astype('float32')
eta_data =  f['consts_threeM'][:,:,1].astype('float32')
phi_data = f['consts_threeM'][:,:,2].astype('float32')
labels_vec = f['y'][:].astype('int32')
labels = np.eye(2, dtype='float32')[labels_vec]
f.close()

# If desired, perform preprocessing on pT
if preprocess_pt:

    # Start by finding number of constituents, will then loop through this vector
    num_constits = np.count_nonzero(z_data, axis=1)

    # Generate zero arrays to store pt, eta, phi
    z_pp = np.zeros_like(z_data)
    
    # Event loop!
    for i, constits in enumerate(num_constits):

        # Separate out pt
        jet_pt = z_data[i,:constits]

        # Normalize pT by itself
        jet_pt = pp.normalize(jet_pt, jet_pt)

        # Dump results into prepared array
        z_pp[i,:constits] = jet_pt

    # Finish by setting new array to old variable name
    z_data = z_pp


# Let's do some visualization to make preprocessing works.
z_flat = z_data.flatten()
eta_flat = eta_data.flatten()
phi_flat = phi_data.flatten()

plt.hist2d(eta_flat, phi_flat, bins=100, range=[[-np.pi, np.pi], [-np.pi, np.pi]], norm=LogNorm())
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\phi$')
plt.savefig("plots/pp_test.png", dpi=300)

# With preprocessing and visualization done, we  need to call dstack to form out eta, phi pairs
ang_data = np.dstack((eta_data, phi_data))

# Make train/test split
(pt_train, pt_val,
 ang_train, ang_val,
 Y_train, Y_val) = ef.utils.data_split(z_data, ang_data, labels, test=0.25)

# Group data
valid_data = [pt_val, ang_val]
train_data = [pt_train, ang_train]
labels_valid = Y_val
labels_train = Y_train

############################# Build Model ##################################
print("\nBuilding model...")

# Build model
model = ef.archs.EFN(
    input_dim=2,
    Phi_sizes=tuple(args.phisizes),
    F_sizes=tuple(args.fsizes),
    Phi_acts="relu",
    F_acts="relu",
    Phi_k_inits="glorot_normal",
    F_k_inits="glorot_normal",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    output_act="softmax",
    summary=True
)

########################## Initial Evaluation ################################

print("\nPre-train evaluation...")
preds = model.predict(valid_data, batch_size=batch_size)

# Make a histogram of network output, separated into signal/background
preds_sig = preds[labels_valid[:,1] == 1,:]
preds_bkg = preds[labels_valid[:,1] == 0,:]
hist_bins = np.linspace(0, 1.0, 100)

plt.clf()
plt.hist(preds_sig[:,1], bins=hist_bins, alpha=0.5, label='Signal')
plt.hist(preds_bkg[:,1], bins=hist_bins, alpha=0.5, label='Background')
plt.legend()
plt.ylabel("Counts")
plt.xlabel("Model output")
plt.title("Model output over validation set")
plt.savefig('./plots/initial_output.png', dpi=300)

############################### Train EFN #################################

# Earlystopping callback
earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    mode='min'
)

# Checkpoint callback
check_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

# Tensorboard callback
tboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs',
    histogram_freq=1
)

# Train model by calling fit
print("\nTraining model...")
train_hist = model.fit(
    train_data,
    labels_train,
    epochs=num_epoch,
    batch_size=batch_size,
    validation_data=(valid_data, labels_valid),
    callbacks=[earlystop_callback, check_callback, tboard_callback],
    verbose=2
)

# Plot losses
plt.clf()
plt.plot(train_hist.history['loss'], label='Training')
plt.plot(train_hist.history['val_loss'], label='Validation')
plt.title("Loss for model training")
plt.legend()
plt.ylabel("Crossentropy loss")
plt.xlabel("Epoch")
plt.savefig("./plots/loss.png", dpi=300)

############################# Evaluate EFN #################################

# Get predictions on validation data
print("\nEvaluate model...")
preds = model.predict(valid_data, batch_size=batch_size)

# Make a histogram of network output, separated into signal/background
preds_sig = preds[labels_valid[:,1] == 1,:]
preds_bkg = preds[labels_valid[:,1] == 0,:]
hist_bins = np.linspace(0, 1.0, 100)

plt.clf()
plt.hist(preds_sig[:,1], bins=hist_bins, alpha=0.5, label='Signal')
plt.hist(preds_bkg[:,1], bins=hist_bins, alpha=0.5, label='Background')
plt.legend()
plt.ylabel("Counts")
plt.xlabel("Model output")
plt.title("Model output over validation set")
plt.savefig("./plots/final_output.png", dpi=300)

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
plt.savefig("./plots/roc.png", dpi=300)

# Finally just print some predictions to make sure spike outputs are identical
print(preds[:200,1])
