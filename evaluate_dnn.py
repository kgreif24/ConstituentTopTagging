""" evaluate_dnn.py - This script will take a .pt file which stores a
state_dict, and a .root file which stores a dataset as command line arguments.
It will then evaluate the model over the validation set, plot an inverse
roc curve, and print out some metrics.

Author: Kevin Greif
7/2/21
python3
"""

import sklearn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import data_dumper
import classifier_trainer


# Start by parsing command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str,
                    help='Path to .root file containing data')
parser.add_argument('--stdict', type=str,
                    help='Path to .pt file containing state dict')
args = parser.parse_args()

# Set some parameters (include these in .pt file eventually!)
my_max_constits = 80
my_batch_size = 100
constit_branches = ['fjet_sortClusStan_pt', 'fjet_sortClusCenterRotFlip_eta',
                    'fjet_sortClusCenterRot_phi', 'fjet_sortClusStan_e']

# Next load in data using datadumper class
print("\nBuilding data object...")
dd_eval = data_dumper.DataDumper(args.data, "valid",
                                   constit_branches, 'fjet_signal', 'fjet_match_weight_pt')
eval_dl = dd_eval.torch_dataloader(max_constits=my_max_constits, batch_size=my_batch_size, shuffle=True)

# Find shape of each minibatch
sample_shape = tuple([my_batch_size]) + dd_eval.sample_shape()
input_shape = sample_shape[1] * sample_shape[2]

# And delete data dumper to save on memory
del dd_eval

# Now build the model
print("\nBuilding model...")
model = simpleDNN(input_shape)
print(model)

# Build optimizer and loss function, optimizer not really necessary but
# required as argument to classifier trainer.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.BCELoss(reduction='none')  # to apply event weights set reduction to none

# Now build ClassifierTrainer for this model
model_trainer = classifier_trainer.ClassifierTrainer(model, optimizer, loss_func)

# Next load data into the model_trainer
model_trainer.load_data(eval_dl, flag=2)

# Now comes the new stuff. We want to load the state dict into the model
model_trainer.load_model(args.stdict)

# And run the analyze method to get fpr, tpr
analyze_dict = model_trainer.analyze()
fpr = analyze_dict['fpr']
tpr = analyze_dict['tpr']
fprinv = 1 / fpr

# Find background rejection at tpr = 0.5, 0.8 working points
wp_p5 = np.argmax(tpr > 0.5)
wp_p8 = np.argmax(tpr > 0.8)

print("Background rejection at 0.5 signal efficiency: ", fprinv[wp_p5])
print("Background rejection at 0.8 signal efficiency: ", fprinv[wp_p8])
print("AUC score: ", analyze_dict['auc'])

# Make an inverse roc plot. Take 1/fpr and plot this against tpr
plt.plot(tpr, fprinv)
