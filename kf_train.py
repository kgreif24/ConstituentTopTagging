""" kf_train.py - This script will build, train, and evaluate a network,
implemented with the keras/energyflow packages. It will make use of the
DataHandler class to quickly generate np arrays of the jet data. Script also
splits data in train/validation, for use in k-fold cross validation.

Author: Kevin Greif
Last updated 8/16/21
python3
"""

import sys, os
import gc
import argparse
import tracemalloc

import energyflow as ef
from energyflow.archs import EFN
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

from data_handler import DataHandler

# If GPU is available, print message
print("\nStart model training script...")
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))


########################## Define Functions ##########################

# Here we'll define some utility function to be used in training.
# (Refactor this into a separate file please)

def log_model_output(epoch, logs):
    """ log_model_output - This function is meant to be passed to a 
    keras Lambda callback class. It is not a pure function, and will
    inherit its variables from the instance. It will plot
    the model output over the validation set.
    """

    # Run model prediction over validation set
    preds = model.predict(valid_data, batch_size=batch_size)

    # Split model output into signal and background
    preds_sig = preds[valid_labels[:,1] == 1, 1]
    preds_bkg = preds[valid_labels[:,1] == 0, 1]

    # Now log histograms to tensorboard
    with file_writer.as_default():
        tf.summary.histogram('signal', preds_sig, step=epoch)
        tf.summary.histogram('background', preds_bkg, step=epoch)

    gc.collect()


########################## Parse Arguments ###########################

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

###################### Set parameters for training #######################

# Network parameters
net_type = args.type

# Training parameters
num_epoch = args.numEpochs
batch_size = args.batchSize
checkpoint_filepath = args.checkDir

# Data parameters
filepath = "/pub/kgreif/samples/sample_4p2M_nbpt.root"

if 'hl' in net_type:
    input_branches = ['fjet_Tau1_wta', 'fjet_Tau2_wta', 'fjet_Tau3_wta', 'fjet_Split12',
                      'fjet_Split23', 'fjet_ECF1', 'fjet_ECF2', 'fjet_ECF3', 'fjet_C2',
                      'fjet_D2', 'fjet_Qw']
else:
    input_branches = ['fjet_sortClusNormByPt_pt', 'fjet_sortClusCenterRotFlip_eta',
                      'fjet_sortClusCenterRot_phi', 'fjet_sortClusNormByPt_e']

extra_branches = ['fjet_match_weight_pt', 'fjet_pt']
max_constits = args.maxConstits
num_folds = args.numFolds
fold = args.fold - 1    # -1 to make fold an index in array

############################# Process Data ################################

# Now build dhs and use them to plot all branches of interest
print("Building data objects...")
dhandler = DataHandler(filepath, "FlatSubstructureJetTree", input_branches,
                       extras=extra_branches, max_constits=max_constits)
dhandler.plot_branches(input_branches + extra_branches, directory="./plots/")

# Figure out sample shape
sample_shape = tuple([batch_size]) + dhandler.sample_shape()

# Now make np arrays out of the data
print("\nBuilding data arrays...")
data_arrs = dhandler.np_arrays()

# Make train/valid split
print("\nSplitting data arrays...")
data = np.array_split(data_arrs[0], num_folds, axis=0)
labels = np.array_split(data_arrs[1], num_folds, axis=0)
weight = np.array_split(data_arrs[2], num_folds, axis=0)
valid_data = np.squeeze(data.pop(fold))
valid_labels = np.squeeze(labels.pop(fold))
valid_weight = np.squeeze(weight.pop(fold))
valid_events = len(valid_weight)
train_data = np.squeeze(np.concatenate(data, axis=0))
train_labels = np.squeeze(np.concatenate(labels, axis=0))
train_weight = np.squeeze(np.concatenate(weight, axis=0))
train_events = len(train_weight)
del data_arrs, data, labels, weight

print("Using fold", fold+1, "of", num_folds)
print("Training events: ", train_events)
print("Valid events: ", valid_events)
print("Total events: ", train_events + valid_events)
print(np.shape(train_data))

# Verify that there are no NaNs in data, labels, or weights
assert not np.any(np.isnan(train_data))
assert not np.any(np.isnan(valid_data))
assert not np.any(train_data == -999)
assert not np.any(valid_data == -999)
assert np.shape(train_data)[0] == train_events

############################# Build Model ##################################
print("\nBuilding model...")

if 'dnn' in net_type:
    print("\n*** DNN model ***")

    # Find input shape
    input_shape = np.prod(sample_shape[1:])

    # Get data into proper setting
    train_data = train_data.reshape((train_events, input_shape))
    valid_data = valid_data.reshape((valid_events, input_shape))
    print("Shape of training data: ", np.shape(train_data))

    # Build model
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_shape,)))
    for layer in args.nodes:
        model.add(tf.keras.layers.Dense(layer, kernel_initializer='he_uniform'))
        # model.add(tf.keras.layers.BatchNormalization(axis=1))
        model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(2, kernel_initializer='he_uniform', activation='softmax'))

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc')]
    )

    # Once model is built, print summary
    model.summary()

elif net_type == 'efn':

    # Group data into pT and angular information (and delete old arrays)
    train_data = [train_data[:,:,0], train_data[:,:,1:3]]
    valid_data = [valid_data[:,:,0], valid_data[:,:,1:3]]

    # Build model
    model = ef.archs.EFN(
        input_dim=2,
        Phi_sizes=tuple(args.phisizes),
        F_sizes=tuple(args.fsizes),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
    )

elif net_type == 'pfn':

    # Data is already grouped into proper format for PFN, so just build model
    model = ef.archs.PFN(
        input_dim=4,
        Phi_sizes=tuple(args.phisizes),
        F_sizes=tuple(args.fsizes),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
    )

else:
    raise ValueError("Model type is not known!")


########################## Initial Evaluation ################################

print("\nPre-train evaluation...")
preds = model.predict(valid_data, batch_size=batch_size)

# Make a histogram of network output, separated into signal/background
preds_sig = preds[valid_labels[:,1] == 1]
preds_bkg = preds[valid_labels[:,1] == 0]
hist_bins = np.linspace(0, 1.0, 100)

plt.clf()
plt.hist(preds_sig[:,1], bins=hist_bins, alpha=0.5, label='Signal')
plt.hist(preds_bkg[:,1], bins=hist_bins, alpha=0.5, label='Background')
plt.legend()
plt.ylabel("Counts")
plt.xlabel("Model output")
plt.title("Model output over validation set")
plt.savefig('./plots/initial_output.png', dpi=300)

############################### Train model #################################

# Earlystopping callback
earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc',
    patience=20,
    mode='max'
)

# Checkpoint callback
check_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_acc',
    mode='max',
    save_best_only=True
)

# Tensorboard callback
tboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs',
    histogram_freq=1
)

# Log model output callback
output_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_model_output)

# Make file writer for use in writing custom histograms
file_writer = tf.summary.create_file_writer('logs/validation')

# Train model by calling fit
print("\nTraining model...")
train_hist = model.fit(
    train_data,
    train_labels,
    epochs=num_epoch,
    batch_size=batch_size,
    sample_weight=train_weight,
    validation_data=(valid_data, valid_labels, valid_weight),
    callbacks=[earlystop_callback, check_callback, tboard_callback, output_callback],
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

############################# Evaluate model #################################

# Get predictions on validation data
print("\nEvaluate best model...")

# First load best checkpoint
model = tf.keras.models.load_model(checkpoint_filepath)

# Predict
preds = model.predict(valid_data, batch_size=batch_size)

# Make a histogram of network output, separated into signal/background
preds_sig = preds[valid_labels[:,1] == 1,:]
preds_bkg = preds[valid_labels[:,1] == 0,:]
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
fpr, tpr, thresholds = metrics.roc_curve(valid_labels[:,1], preds[:,1])
auc = metrics.roc_auc_score(valid_labels[:,1], preds[:,1])
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
