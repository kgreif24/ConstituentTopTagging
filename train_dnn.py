""" train_model.py - This script will train a simple dense neural network
on the top tagging dataset. Makes use of DataDumper nad ClassifierTrainer
classes to do most of the legwork.

Author: Kevin Greif
6/22/21
python3
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import data_dumper
import classifier_trainer
from dnn.simple_dnn import simpleDNN


# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--enableCuda', action='store_true',
                    help='Enable CUDA')
parser.add_argument('-N', '--numEpochs', default=100, type=int,
                    help='Number of epochs')
parser.add_argument('-b', '--batchSize', default=100, type=int,
                    help='Batch size')
parser.add_argument('--maxConstits', default=80, type=int,
                    help='Number of constituents to include per event')
parser.add_argument('-o', '--checkDir', default='./checkpoints', type=str,
                    help='Stem of file name at which to save checkpoints')
args = parser.parse_args()

# Find device
args.device = None
useCuda = args.enableCuda and torch.cuda.is_available()
if useCuda:
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

# Start by setting some training parameters
n_epochs = args.numEpochs
my_batch_size = args.batchSize
my_max_constits = args.maxConstits
constit_branches = ['fjet_sortClusStan_pt', 'fjet_sortClusCenterRotFlip_eta',
                    'fjet_sortClusCenterRot_phi', 'fjet_sortClusStan_e']

# Build datadumper and return pytorch dataloader object
print("\nBuilding data objects...")
dd_train = data_dumper.DataDumper("/data/homezvol0/kgreif/toptag/samples/sample_1M.root", "train",
                                   constit_branches, 'fjet_signal', 'fjet_match_weight_pt')
dd_valid = data_dumper.DataDumper("/data/homezvol0/kgreif/toptag/samples/sample_1M.root", "valid",
                                   constit_branches, 'fjet_signal', 'fjet_match_weight_pt')
train_dl = dd_train.torch_dataloader(max_constits=my_max_constits, batch_size=my_batch_size, shuffle=True)
valid_dl = dd_valid.torch_dataloader(max_constits=my_max_constits, batch_size=my_batch_size, shuffle=True)
print("Training events: ", dd_train.num_events)
print("Validation events: ", dd_valid.num_events)

# Find shape of each mini batch
sample_shape = tuple([my_batch_size]) + dd_train.sample_shape()
input_shape = sample_shape[1] * sample_shape[2]

# Delete data dumpers to save on memory!
del dd_train
del dd_valid

# Now build the model!
print("\nBuilding model...")
model = simpleDNN(input_shape).to(device=args.device)
print(model)

# Build optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.BCELoss(reduction='none')  # to apply event weights set reduction to none

# Now build ClassifierTrainer for this model
model_trainer = classifier_trainer.ClassifierTrainer(model, optimizer, loss_func)

# Next load data into the model_trainer
model_trainer.load_data(train_dl, flag=1)
model_trainer.load_data(valid_dl, flag=2)

# Analyze the model before training
model_trainer.analyze(filename="./plots/initial_output.png", my_device=args.device)

# Train the model
min_loss_index = model_trainer.train(n_epochs,
                                     my_device=args.device,
                                     validate=True,
                                     checkpoints=args.checkDir)

# Show some simple results
print("\nFinal train loss: ", model_trainer.tr_loss_array[min_loss_index])
print("Final valid loss: ", model_trainer.val_loss_array[min_loss_index])

# Plot loss and save figure
plt.plot(model_trainer.tr_loss_array, label="Training")
plt.plot(model_trainer.val_loss_array, label="Validation")
plt.title("Loss for DNN training")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.savefig("./plots/loss.png", dpi=300)
plt.clf()

# Load in model with the best training loss
filestr = args.checkDir + "/checkpt_e" + str(min_loss_index) + ".pt"
model_trainer.load_model(filestr)

# Finally analyze the model after training
analyze_dict = model_trainer.analyze(filename="./plots/final_output.png",
                                     my_device=args.device)

# Calculate background rejection at FPR = 0.5, 0.8 working points


# Print out metrics
print("AUC score: ", analyze_dict['auc'])
