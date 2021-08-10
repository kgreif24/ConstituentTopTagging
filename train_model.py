""" train_model.py - This script will build, train, and evaluate a network,
implemented with the keras/energyflow packages. It will make use of the
DataDumper class to quickly generate np arrays of the jet data.

Author: Kevin Greif
Last updated 8/6/21
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
print("\nStart model training script...")
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))


########################## Parse Arguments ###########################

parser = argparse.ArgumentParser()

parser.add_argument('--type', default='dnn', type=str,
                    help='Type of model to build (dnn, efn, pfn)')
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
nodes = 50

# Training parameters
num_epoch = args.numEpochs
batch_size = args.batchSize
checkpoint_filepath = args.checkDir

# Data parameters
filepath = "~/toptag/samples/csauer_initial.root"
constit_branches = ['fjet_sortClusNormByPt_pt', 'fjet_sortClusCenterRotFlip_eta',
                    'fjet_sortClusCenterRot_phi', 'fjet_sortClusNormByPt_e']
extra_branches = ['fjet_training_weight_pt', 'fjet_pt']
max_constits = args.maxConstits

############################# Process Data ################################

# Now build dhs and use them to plot all branches of interest
print("Building data objects...")
dh_train = DataHandler(filepath, "train", constit_branches,
                      extras=extra_branches, max_constits=max_constits)
dh_valid = DataHandler(filepath, "valid", constit_branches,
                      extras=extra_branches, max_constits=max_constits)
dh_train.plot_branches(constit_branches + extra_branches, directory="./plots/")

# Figure out sample shape
sample_shape = tuple([batch_size]) + dh_train.sample_shape()

# Now make np arrays out of the data
print("\nBuilding data arrays...")
train_arrs = dh_train.np_arrays()
valid_arrs = dh_valid.np_arrays()

# Delete dhs to save memory
del dh_train
del dh_valid

# Split data arrays common to all model types
print("\nSplitting data arrays...")
labels_train = train_arrs[1]
labels_valid = valid_arrs[1]
weight_train = train_arrs[2]
weight_valid = valid_arrs[2]
train_events = len(weight_train)
valid_events = len(weight_valid)

# Verify that there are no NaNs in data, labels, or weights
assert not np.any(np.isnan(train_arrs[0]))
assert not np.any(np.isnan(labels_train))
assert not np.any(np.isnan(weight_train))

############################# Build Model ##################################
print("\nBuilding model...")

if net_type == 'dnn':
    print("\n*** DNN model ***")

    # Find input shape
    input_shape = sample_shape[1] * sample_shape[2]

    # Get data into proper setting (and delete old arrays)
    train_data = train_arrs[0].reshape((train_events, input_shape))
    valid_data = valid_arrs[0].reshape((valid_events, input_shape))
    print("Shape of training data: ", np.shape(train_data))
    del train_arrs
    del valid_arrs

    # Find input shape
    input_shape = sample_shape[1] * sample_shape[2]

    # Build model
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(tf.keras.layers.Dense(nodes))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(nodes))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    # Earlystopping callback
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
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

elif net_type == 'efn':

    # Group data into pT and angular information
    train_data = [train_arrs[0][:,:,0], train_arrs[0][:,:,1:3]]
    valid_data = [valid_arrs[0][:,:,0], valid_arrs[0][:,:,1:3]]

    # Build model (NEED TO IMPLEMENT PHI AND F SIZES)
    efn = ef.archs.EFN(input_dim=2,
                       Phi_sizes=Phi_sizes,
                       F_sizes=F_sizes,
                       filepath=checkpoint_filepath,
                       patience=50,
                       earlystop_opts={'monitor': 'val_loss'})

elif net_type == 'pfn':

    # Define PFN here!
    pass

else:
    raise ValueError("Model type is not known!")

# Once model is built, print summary
model.summary()


########################## Initial Evaluation ################################

print("\nPre-train evaluation...")
preds = model.predict(valid_data, batch_size=batch_size)

# Make a histogram of network output, separated into signal/background
preds_sig = preds[labels_valid[:,1] == 1]
preds_bkg = preds[labels_valid[:,1] == 0]
hist_bins = np.linspace(0, 1.0, 100)

plt.clf()
plt.hist(preds_sig[:,1], bins=hist_bins, alpha=0.5, label='Signal')
plt.hist(preds_bkg[:,1], bins=hist_bins, alpha=0.5, label='Background')
plt.yscale('log')
plt.legend()
plt.ylabel("Counts")
plt.xlabel("Model output")
plt.title("Model output over validation set")
plt.savefig('./plots/initial_output.png', dpi=300)

############################### Train EFN #################################

# Train model by calling fit
print("\nTraining model...")
train_hist = model.fit(
    train_data,
    labels_train,
    epochs=num_epoch,
    batch_size=batch_size,
    sample_weight=weight_train,
    validation_data=(valid_data, labels_valid, weight_valid),
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
preds_sig = preds[labels_valid[:,1] == 1]
preds_bkg = preds[labels_valid[:,1] == 0]
hist_bins = np.linspace(0, 1.0, 100)

plt.clf()
plt.hist(preds_sig[:,1], bins=hist_bins, alpha=0.5, label='Signal')
plt.hist(preds_bkg[:,1], bins=hist_bins, alpha=0.5, label='Background')
plt.yscale('log')
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
