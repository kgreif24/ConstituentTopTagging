""" kf_train.py - A script to parse command line arguments and run the
training routine as set up in the ModelTrainer class.

Author: Kevin Greif
Last updated 1/4/22
python3
"""

import sys, os
import argparse

import tensorflow as tf

from model_trainer import ModelTrainer


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
parser.add_argument('-lr', '--learningRate', default=1e-4, type=float,
                    help='Initial learning rate to be used in training')
parser.add_argument('--maxConstits', default=80, type=int,
                    help='Number of constituents to include per event')
parser.add_argument('--schedule', action='store_true', default=False,
                    help='If present, use the learning rate scheduler defined in model_trainer.py')
parser.add_argument('--numFolds', default=5, type=int,
                    help='Number of folds used in training run')
parser.add_argument('--fold', default=1, type=int,
                    help='The fold being used in this particular job')
args = parser.parse_args()


# Convert args namespace to dict to use in building model
setup = vars(args)

# Define training data location
file = '/data0/kgreif/train.h5'

# Create ModelTrainer instance
mt = ModelTrainer(setup, file)

# Run routine
plots = './plots'
checks = './checkpoints'
mt.routine(setup['numEpochs'], plots, checks, use_schedule=args.schedule)
