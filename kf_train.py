""" kf_train.py - A script to parse command line arguments and run the
training routine as set up in the ModelTrainer class.

Author: Kevin Greif
Last updated 1/14/22
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
                    help='DNN number of nodes in layers, or resnet number of blocks in stages')
parser.add_argument('--fsizes', default=[], type=int, nargs='*',
                    help='EFN/PFN number of nodes in f layers')
parser.add_argument('--phisizes', default=[], type=int, nargs='*',
                    help='EFN/PFN number of nodes in phi layers')
parser.add_argument('--batchNorm', action='store_true', default=False,
                    help='If present, use batch norm in DNN based models')
parser.add_argument('--latent_dropout', default=0., type=float,
                    help='The dropout rate to apply to the latent layer in EFN/PFN networks')
parser.add_argument('--dropout', default=0., type=float,
                    help='The dropout rate to use in DNN layers and in EFN/PFN F network')
parser.add_argument('--l1reg', default=0., type=float,
                    help='The amount of L1 regularization to apply to DNN based networks')
parser.add_argument('--bnMom', default=0.1, type=float,
                    help='The momentum to use in batch normalization layers (resnet specific)')
parser.add_argument('--knn', default=18, type=int,
                    help='k to use in particle nets k-nearest neighbors operation')
parser.add_argument('--n_blocks', default=3, type=int,
                    help='Number of GraphConv blocks ot use in particle net')
parser.add_argument('--convs', default=[], type=int, nargs='*',
                    help='Number of hidden graph features in each block in particle net')
parser.add_argument('--block_depth', default=[], type=int, nargs='*',
                    help='Number of graph convolutions to include in each block')
parser.add_argument('--pooling', default='max', choices=['average', 'max'], type=str,
                    help='Whether to use max of average pooling in graph conv in particle net')
parser.add_argument('-N', '--numEpochs', default=100, type=int,
                    help='Number of epochs')
parser.add_argument('-b', '--batchSize', default=100, type=int,
                    help='Batch size')
parser.add_argument('-lr', '--learningRate', default=1e-4, type=float,
                    help='Initial learning rate to be used in training')
parser.add_argument('--maxConstits', default=80, type=int,
                    help='Number of constituents to include per event')
parser.add_argument('--schedule', default=0, type=int,
                    help='Set which lr scheduler to use, 1 for step, 2 for cycle')
parser.add_argument('--numFolds', default=5, type=int,
                    help='Number of folds used in training run')
parser.add_argument('--fold', default=None, type=int,
                    help='The fold being used in this particular job')
parser.add_argument('--file', default=None, required=True, type=str,
                    help='The file to be used for training')
parser.add_argument('--dir', default=None, required=True, type=str,
                    help='The directory to be used for placing training output')
parser.add_argument('--logdir', default='.', type=str,
                    help='The directory to be used for placing tensorboard logs')
args = parser.parse_args()

# Convert args namespace to dict to use in building model
setup = vars(args)

# Create ModelTrainer instance
mt = ModelTrainer(setup, args.file)

# Run routine
plots = args.dir + '/plots'
checks = args.dir + '/checkpoints'
mt.routine(plots, checks, args.logdir, patience=24, use_schedule=args.schedule)
