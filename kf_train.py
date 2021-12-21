""" kf_train.py - This script will build, train, and evaluate a network,
implemented with the keras/energyflow packages. It will make use of the
DataLoader class to pass to a keras fit function. This will allow data to
be loaded into memory batch by batch making use of h5py's array slicing.
Hopefully this is fast enough data transfer to make training resonably quick.

Author: Kevin Greif
Last updated 12/20/21
python3
"""

import sys, os
import argparse

import energyflow as ef
import tensorflow as tf
import sklearn.metrics as metrics
import numpy as np

import colorcet as cc
import matplotlib
# Matplotlib import setup to use non-GUI backend, comment for interactive
# graphics
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('~/mattyplotsalot/allpurpose.mplstyle')

from data_loader import DataLoader
from data_loader import FakeLoader
import models


# If GPU is available, print message
print("\nStart model training script...")
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

########################## Setup ###########################

parser = argparse.ArgumentParser()

parser.add_argument('--type', default='dnn', type=str,
                    help='Type of model to build (dnn, efn, pfn)')
parser.add_argument('--nodes', default=None, type=int, nargs='*',
                    help='DNN number of nodes in layers')
parser.add_argument('--fsizes', default=[], type=int, nargs='*',
                    help='EFN/PFN number of nodes in f layers')
parser.add_argument('--phisizes', default=[], type=int, nargs='*',
                    help='EFN/PFN number of nodes in phi layers')
parser.add_argument('--batchNorm', action='store_true', default=False,
                    help='If present, use batch norm in DNN based models')
parser.add_argument('--dropout', default=0., type=float,
                    help='The dropout rate to use in DNN layers and in EFN/PFN F network')
parser.add_argument('-N', '--numEpochs', default=100, type=int,
                    help='Number of epochs')
parser.add_argument('-b', '--batchSize', default=100, type=int,
                    help='Batch size')
parser.add_argument('--maxConstits', default=80, type=int,
                    help='Number of constituents to include per event')
parser.add_argument('-o', '--checkDir', default='./checkpoints', type=str,
                    help='Stem of file name at which to save checkpoints')
parser.add_argument('--numFolds', default=5, type=int,
                    help='Number of folds used in training run')
parser.add_argument('--fold', default=1, type=int,
                    help='The fold being used in this particular job')
args = parser.parse_args()

####################### Data Handling ######################

# Data parameters
filepath = "./dataloc/train.h5"

# Now build dhs and use them to plot all branches of interest
print("Building data objects...")
dtrain = DataLoader(filepath, batch_size=args.batchSize, net_type=args.type, num_folds=5, this_fold=2)
dvalid = DataLoader(filepath, batch_size=args.batchSize, net_type=args.type, num_folds=5, this_fold=2, valid=True)

########################## Get Model ########################

model = models.build_model(args.type, dtrain.sample_shape, args)

##################### Initial Evaluation ####################

print("\nPre-train evaluation...")
preds = model.predict(dvalid, batch_size=args.batchSize, verbose=1)

# Make a histogram of network output, separated into signal/background
labels_vec = dvalid.file['labels'][dvalid.indeces]
preds_sig = preds[labels_vec == 1]
preds_bkg = preds[labels_vec == 0]
hist_bins = np.linspace(0, 1.0, 100)

plt.clf()
plt.hist(preds_sig, bins=hist_bins, alpha=0.5, label='Signal')
plt.hist(preds_bkg, bins=hist_bins, alpha=0.5, label='Background')
plt.legend()
plt.ylabel("Counts")
plt.xlabel("Model output")
plt.title("Model output over validation set")
plt.savefig('./plots/initial_output.png', dpi=300
)

############################### Train model #################################

# Earlystopping callback
earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=50,
    mode='min'
)

# Checkpoint callback
check_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=args.checkDir,
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
    dtrain,
    epochs=args.numEpochs,
    batch_size=args.batchSize,
    validation_data=dvalid,
    callbacks=[earlystop_callback, check_callback, tboard_callback],
    verbose=1
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

############################# Evaluate model #################################

# Get predictions on validation data
print("\nEvaluate best model...")

# First load best checkpoint
model = tf.keras.models.load_model(args.checkDir)

# Predict
preds = model.predict(dvalid, batch_size=args.batchSize)

# Make a histogram of network output, separated into signal/background
preds_sig = preds[labels_vec == 1]
preds_bkg = preds[labels_vec == 0]
hist_bins = np.linspace(0, 1.0, 100)

plt.clf()
plt.hist(preds_sig, bins=hist_bins, alpha=0.5, label='Signal')
plt.hist(preds_bkg, bins=hist_bins, alpha=0.5, label='Background')
plt.legend()
plt.ylabel("Counts")
plt.xlabel("Model output")
plt.title("Model output over validation set")
plt.savefig("./plots/final_output.png", dpi=300)

# Get ROC curve and AUC
fpr, tpr, thresholds = metrics.roc_curve(labels_vec, preds)
auc = metrics.roc_auc_score(labels_vec, preds)
fprinv = 1 / fpr

# Find background rejection at tpr = 0.5, 0.8 working points
wp_p5 = np.argmax(tpr > 0.5)
wp_p8 = np.argmax(tpr > 0.8)

# Finally print information on model performance
print("AUC score: ", auc)
print("ACC score: ", np.max(train_hist.history['val_acc']))
print("Background rejection at 0.5 signal efficiency: ", fprinv[wp_p5])
print("Background rejection at 0.8 signal efficiency: ", fprinv[wp_p8])

# Make an inverse roc plot. Take 1/fpr and plot this against tpr
plt.clf()
plt.plot(tpr, fprinv)
plt.yscale('log')
plt.ylabel('Background rejection')
plt.xlabel('Signal efficiency')
plt.savefig("./plots/roc.png", dpi=300)
