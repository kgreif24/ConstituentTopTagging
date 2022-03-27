""" rt_objective.py - This script will define a function which can serve as
the objective trainable in doing hyperparameter optimization with ray tune.

Author: Kevin Greif
Last updated 1/10/22
python 3
"""

import os

from ray import tune
from ray.tune.integration.keras import TuneReportCheckpointCallback
import numpy as np

from base_trainer import BaseTrainer

def node_list(num_layers, num_nodes):
    return int(num_nodes) * np.ones(int(num_layers), dtype=np.int32)

def objective(config, checkpoint_dir=None):

    # First find whether we should load from file or build model from scratch
    trial_dir = tune.get_trial_dir()
    print("Looking at directory:", trial_dir)

    # Initialize model dir variable to use as a "found model to load" flag
    model_dir = None

    # Loop through number of epochs to look for newest checkpoint
    for i in range(config['numEpochs']):
        
        # Build checkpoint dir to look for
        check_dir = "checkpoint_" + str(i).zfill(6)
        check_dir = os.path.join(trial_dir, check_dir)
        
        # If checkpoint dir exists, set a model directory to load from
        if os.path.isdir(check_dir):
            model_dir = os.path.join(check_dir, "model")

    # At end of loop should have newest checkpoint directory if one exists
    # Now alter config dict so we load from checkpoint
    if model_dir is not None:
        config['checkpoint'] = model_dir

    # Create base trainer class
    trainer = BaseTrainer(config, config['filepath'])

    # Build tune callback
    callback = TuneReportCheckpointCallback({'score': 'val_loss', 'acc': 'val_acc'}, filename='model', on='epoch_end')

    # Run training
    hist = trainer.train([callback])
