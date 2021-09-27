""" kf_train.py - This script will build, train, and evaluate a network,
implemented with the keras/energyflow packages. It will make use of the
DataLoader class to pass to a keras fit function. This will allow data to be
loaded in 100 MB chunks using uproot's lazy arrays. Also uses the build_model
function in the models.py file.

Author: Kevin Greif
Last updated 9/22/21
python3
"""

import sys, os
import gc
import argparse

import energyflow as ef
from energyflow.archs import EFN
import tensorflow as tf
import sklearn.metrics as metrics
import numpy as np

import colorcet as cc
import matplotlib
# Matplotlib import setup to use non-GUI backend, comment for interactive
# graphics
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use('~/mattyplotsalot/allpurpose.mplstyle')
plt.style.use('~/Documents/General Programming/mattyplotsalot/allpurpose.mplstyle')

from data_loader import DataLoader
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
# filepath = "/pub/kgreif/samples/sample_4p2M_nbpt.root"
filepath = "../Data/sample_1p5M_nbpt_test.root"

if 'hl' in args.type:
    input_branches = ['fjet_Tau1_wta', 'fjet_Tau2_wta', 'fjet_Tau3_wta', 'fjet_Split12',
                      'fjet_Split23', 'fjet_ECF1', 'fjet_ECF2', 'fjet_ECF3', 'fjet_C2',
                      'fjet_D2', 'fjet_Qw']
else:
    input_branches = ['fjet_sortClusNormByPt_pt', 'fjet_sortClusCenterRotFlip_eta',
                      'fjet_sortClusCenterRot_phi', 'fjet_sortClusNormByPt_e']

# Now build dhs and use them to plot all branches of interest
print("Building data objects...")
dhandler = DataLoader(filepath, "FlatSubstructureJetTree", input_branches)

########################## Get Model ########################

model = models.build_model(args.type, dhandler.sample_shape, args)

# ##################### Initial Evaluation ####################
#
# print("\nPre-train evaluation...")
# preds = model.predict(valid_arrs[0], batch_size=args.batchSize)
#
# # Make a histogram of network output, separated into signal/background
# preds_sig = preds[valid_arrs[1][:,1] == 1]
# preds_bkg = preds[valid_arrs[1][:,1] == 0]
# hist_bins = np.linspace(0, 1.0, 100)
#
# plt.clf()
# plt.hist(preds_sig[:,1], bins=hist_bins, alpha=0.5, label='Signal')
# plt.hist(preds_bkg[:,1], bins=hist_bins, alpha=0.5, label='Background')
# plt.legend()
# plt.ylabel("Counts")
# plt.xlabel("Model output")
# plt.title("Model output over validation set")
# plt.savefig('./plots/initial_output.png', dpi=300)

############################### Train model #################################

# Earlystopping callback
earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
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
    dhandler,
    epochs=args.numEpochs,
    batch_size=args.batchSize,
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
plt.show()
# plt.savefig("./plots/loss.png", dpi=300)

# ############################# Evaluate model #################################
#
# # Get predictions on validation data
# print("\nEvaluate best model...")
#
# # First load best checkpoint
# model = tf.keras.models.load_model(args.checkDir)
#
# # Predict
# preds = model.predict(valid_arrs[0], batch_size=args.batchSize)
#
# # Make a histogram of network output, separated into signal/background
# preds_sig = preds[valid_arrs[1][:,1] == 1,:]
# preds_bkg = preds[valid_arrs[1][:,1] == 0,:]
# hist_bins = np.linspace(0, 1.0, 100)
#
# plt.clf()
# plt.hist(preds_sig[:,1], bins=hist_bins, alpha=0.5, label='Signal')
# plt.hist(preds_bkg[:,1], bins=hist_bins, alpha=0.5, label='Background')
# plt.legend()
# plt.ylabel("Counts")
# plt.xlabel("Model output")
# plt.title("Model output over validation set")
# plt.savefig("./plots/final_output.png", dpi=300)
#
# # Get ROC curve and AUC
# fpr, tpr, thresholds = metrics.roc_curve(valid_arrs[1][:,1], preds[:,1])
# auc = metrics.roc_auc_score(valid_arrs[1][:,1], preds[:,1])
# fprinv = 1 / fpr
#
# # Find background rejection at tpr = 0.5, 0.8 working points
# wp_p5 = np.argmax(tpr > 0.5)
# wp_p8 = np.argmax(tpr > 0.8)
#
# # Finally print information on model performance
# print("AUC score: ", auc)
# print("ACC score: ", np.max(train_hist.history['val_acc']))
# print("Background rejection at 0.5 signal efficiency: ", fprinv[wp_p5])
# print("Background rejection at 0.8 signal efficiency: ", fprinv[wp_p8])
#
# # Make an inverse roc plot. Take 1/fpr and plot this against tpr
# plt.clf()
# plt.plot(tpr, fprinv)
# plt.yscale('log')
# plt.ylabel('Background rejection')
# plt.xlabel('Signal efficiency')
# plt.savefig("./plots/roc.png", dpi=300)
