""" eval_save.py - This script will load a model using the standard
tf.keras API, evaluate the model over the test set, and then save
the model's predictions to a .npy file. This prevents us from having
to run inference on the model over and over again.

Author: Kevin Greif
Last updated 2/14/22
python3
"""

import argparse

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from data_loader import DataLoader

# If GPU is available, print message
print("\nStart model evaluation script...")
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))


########################## Parse Arguments ###########################

parser = argparse.ArgumentParser()

parser.add_argument('--type', default=None, type=str,
                    help='Type of model to evaluate (dnn, efn, pfn, etc...)')
parser.add_argument('-b', '--batchSize', default=100, type=int,
                    help='Batch size')
parser.add_argument('--maxConstits', default=80, type=int,
                    help='Number of constituents to include per event')
parser.add_argument('--file', default=None, type=str,
                    help='File name of network checkpoint')
args = parser.parse_args()

###################### Set parameters for evaluation #######################

# Net parameters
net_type = args.type
net_file = args.file
batch_size = args.batchSize

# Data parameters
filepath = "/data0/kgreif/test_mc_m.h5"

############################# Process Data ################################

# Build data loader for this model
print("Building data object...")
dl = DataLoader(filepath, net_type=net_type, mode='test')
labels = dl.file['labels'][:]
jet_pt = dl.file['fjet_pt'][:]

####################### Load and Evaluate Model ########################

print("\nLoading model...")
model = tf.keras.models.load_model(net_file)
model.summary()

preds = model.predict(dl, batch_size=batch_size, verbose=1)
disc_preds = (preds > 0.5).astype(int)

####################### Save Predictions to File #######################

preds_path = net_file + "/preds.npz"
np.savez(preds_path, preds=preds, disc_preds=disc_preds, labels=labels, jet_pt=jet_pt)

