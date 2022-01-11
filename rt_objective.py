""" rt_objective.py - This script will define a function which can serve as
the objective trainable in doing hyperparameter optimization with ray tune.

Author: Kevin Greif
Last updated 1/10/22
python 3
"""


from ray import tune
from ray.tune.integration.keras import TuneReportCallback
import numpy as np

from base_trainer import BaseTrainer

def node_list(num_layers, num_nodes):
    return num_nodes * np.ones(num_layers, dtype=np.int32)

def objective(config):

    # Extract number of epochs for each iteration
    num_epochs = config['numEpochs']

    # Find node list and make it an element of the config dictionary
    config['nodes'] = node_list(config['hidden_layers'], config['nodes_per_layer'])

    # Initialize base trainer class. This is a stupid signature!
    trainer = BaseTrainer(config, config['filepath'])

    # Build tune callback
    callback = TuneReportCallback({'score': 'val_loss'}, on='epoch_end')

    # Run training
    hist = trainer.train(config['numEpochs'], [callback])
