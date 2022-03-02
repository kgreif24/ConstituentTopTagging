""" hp_train.py - This script will run hyper-parameter training using
ray tune. It will set up a search space, and run the tuning job using the
trainable defined in rt_objective.py.

Authors: Kevin Greif
Last updated 2/19/22
python3
"""

import ray
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch

from rt_objective import objective

# Set epochs
max_epochs = 30

# Start by setting up search space
config = {
    "filepath": '/tmp/tt_data/train_mc_m.h5',
    "type": 'pfn',
    "maxConstits": 80,
    "numFolds": 5,
    "fold": None,
    "phi_layers": tune.quniform(1, 5, 1),
    "phi_nodes": tune.quniform(50, 400, 50), 
    "f_layers": tune.quniform(2, 5, 1),
    "f_nodes": tune.quniform(100, 500, 50),
    "learningRate": tune.loguniform(1e-5, 1e-2),
    "latent_dropout": tune.uniform(0, 0.2),
    "f_dropout": tune.uniform(0, 0.2),
    "phi_reg": tune.loguniform(1e-6, 1e-2),
    "f_reg": tune.loguniform(1e-6, 1e-2),
    "numEpochs": max_epochs,
    "batchSize": tune.quniform(100, 500, 50)
}

# Attach ray cluster
ray.init(address='auto')

# Make search algorithm
algo = HyperOptSearch(
    metric='score',
    mode='min'
)
algo = ConcurrencyLimiter(algo, max_concurrent=9)

# Then run the trial
analysis = tune.run(
    objective,
    search_alg=algo,
    config=config,
    name='pfn',
    resume="AUTO",
    metric='score',
    mode='min',
    num_samples=160,
    keep_checkpoints_num=1,
    checkpoint_score_attr='min-score',
    stop={'training_iteration': max_epochs},
    resources_per_trial={'cpu': 1, 'gpu': 1},
    local_dir='/pub/kgreif/tt_model_repo',
    verbose=1
)

# Print results
print("Best hyperparameters found were:", analysis.best_config)

# Exit ray
ray.shutdown()
