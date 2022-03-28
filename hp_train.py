""" hp_train.py - This script will run hyper-parameter training using
ray tune. It will set up a search space, and run the tuning job using the
trainable defined in rt_objective.py.

Authors: Kevin Greif
Last updated 2/19/22
python3
"""

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

from rt_objective import objective

# Set epochs
max_epochs = 80

# Start by setting up search space
config = {
    "filepath": '/scratch/whiteson_group/kgreif/train_ln_m.h5',
    "type": 'dnn',
    "maxConstits": 80,
    "numFolds": 5,
    "fold": None,
    "numEpochs": max_epochs,
    "hidden_layers": tune.quniform(2, 5, 1),
    "nodes_per_layer": tune.quniform(50, 500, 50),
    "dropout": tune.uniform(0, 0.2),
    "batchNorm": tune.choice([True, False]),
    "l1reg": tune.loguniform(1e-5, 1e-2),
    "learningRate": tune.loguniform(1e-5, 1e-2),
    "batchSize": tune.quniform(100, 500, 50)
}

# Attach ray cluster
# ray.init(address='auto')
ray.init(num_cpus=3)

# Make search algorithm
algo = HyperOptSearch(
    metric='score',
    mode='min'
)
algo = ConcurrencyLimiter(algo, max_concurrent=3)

# ASHA scheduler
ash_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='score',
    mode='min',
    max_t=max_epochs,
    grace_period=15,
    reduction_factor=4
)

# Then run the trial
analysis = tune.run(
    objective,
    search_alg=algo,
    scheduler=ash_scheduler,
    config=config,
    name='dnn',
    resume="AUTO",
    num_samples=100,
    keep_checkpoints_num=1,
    checkpoint_score_attr='min-score',
    stop={'training_iteration': max_epochs},
    resources_per_trial={'cpu': 1, 'gpu': 1},
    local_dir='/DFS-L/DATA/whiteson/kgreif/tt_model_repo',
    verbose=1
)

# Print results
print("Best hyperparameters found were:", analysis.best_config)

# Exit ray
ray.shutdown()
