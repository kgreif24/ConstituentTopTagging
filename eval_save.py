""" eval_save.py - This script will load a model using the standard
tf.keras API, evaluate the model over the test set, and then save
the model's predictions to a .npy file. This prevents us from having
to run inference on the model over and over again.

Author: Kevin Greif
Last updated 2/14/22
python3
"""

import argparse
import glob
import os, sys

import tensorflow as tf
import numpy as np
from energyflow.archs import EFN, PFN

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
parser.add_argument('--out', default='preds.npz', type=str,
                    help='Name of .npz file to store predictions in')
args = parser.parse_args()

###################### Set parameters for evaluation #######################

# Net parameters
net_type = args.type
batch_size = args.batchSize

# Find most recent checkpoint
checks = glob.glob(args.file + '/*')
checks = [i for i in checks if os.path.isdir(i)]
checks.sort(key=os.path.getmtime)
net_file = checks[-1]
print("Found checkpoint file {}".format(net_file))

# Data parameters
filepath = "/pub/kgreif/samples/h5dat/test_ln_nominal.h5"

############################# Process Data ################################

# Build data loader for this model
print("Building data object...")
dl = DataLoader(filepath, net_type=net_type, mode='test', max_constits=args.maxConstits, batch_size=batch_size, use_weights=False)
labels = dl.file['labels'][:]
jet_pt = dl.file['fjet_pt'][:]

####################### Load and Evaluate Model ########################

print("\nLoading model...")
model = tf.keras.models.load_model(net_file)
model.summary()

preds = model.predict(dl, batch_size=batch_size, verbose=2)
disc_preds = (preds > 0.5).astype(int)

####################### Save Predictions to File #######################

preds_path = args.file + "/" + args.out
np.savez(preds_path, preds=preds, disc_preds=disc_preds, labels=labels, jet_pt=jet_pt)

