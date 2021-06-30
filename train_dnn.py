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
import data_dumper
import classifier_trainer
from dnn.simple_dnn import simpleDNN


# Start by setting some training parameters
n_epochs = 100
my_batch_size = 100
max_constits = 80
constit_branches = ['fjet_sortClusStan_pt', 'fjet_sortClusCenterRotFlip_eta',
                    'fjet_sortClusCenterRot_phi', 'fjet_sortClusStan_e']

# Build datadumper and return pytorch dataloader object
print("\nBuilding data objects...")
dd_train = data_dumper.DataDumper("/data/homezvol0/kgreif/toptag/samples/sample_1M.root", "train",
                                   constit_branches, 'fjet_signal')
dd_valid = data_dumper.DataDumper("/data/homezvol0/kgreif/toptag/samples/sample_1M.root", "valid",
                                   constit_branches, 'fjet_signal')
train_dl = dd_train.torch_dataloader(max_constits=80, batch_size=my_batch_size, shuffle=True)
valid_dl = dd_train.torch_dataloader(max_constits=80, batch_size=my_batch_size, shuffle=True)

# Find shape of each mini batch
sample_shape = tuple([my_batch_size]) + dd_train.sample_shape()
input_shape = sample_shape[1] * sample_shape[2]

# Now build the model!
print("\nBuilding model...")
model = simpleDNN(input_shape)

# Build optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.BCELoss()

# Now build ClassifierTrainer for this model
model_trainer = classifier_trainer.ClassifierTrainer(model, optimizer, loss_func)

# Next load data into the model_trainer
model_trainer.load_data(train_dl, flag=1)
model_trainer.load_data(valid_dl, flag=2)

# Analyze the model before training
model_trainer.analyze(filename="./plots/initial_output.png")

# Train the model
min_loss_index = model_trainer.train(n_epochs, validate=True, checkpoints="./checkpoints")

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
model_trainer.load_model("./checkpoints/checkpt_e" + str(min_loss_index) + ".pt")

# Finally analyze the model after training
model_trainer.analyze(filename="./plots/final_output.png")
