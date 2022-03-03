""" rt_objective.py - This script will define a function which can serve as
the objective trainable in doing hyperparameter optimization with ray tune.

Author: Kevin Greif
Last updated 1/10/22
python 3
"""


from ray import tune
from ray.tune.integration.keras import TuneReportCheckpointCallback
import numpy as np

from base_trainer import BaseTrainer

def node_list(num_layers, num_nodes):
    return int(num_nodes) * np.ones(int(num_layers), dtype=np.int32)

def objective(config, checkpoint_dir=None):

    # For resnet, node list is already in config

    # # Find node lists and make them an element of the config dictionary
    # config['phisizes'] = node_list(config['phi_layers'], config['phi_nodes'])
    # config['fsizes'] = node_list(config['f_layers'], config['f_nodes'])

    # Initialize base trainer class
    trainer = BaseTrainer(config, config['filepath'])

    # Build tune callback
    callback = TuneReportCheckpointCallback({'score': 'val_loss', 'acc': 'val_acc'}, filename='model', on='epoch_end')

    # Run training
    hist = trainer.train([callback])
