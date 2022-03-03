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
from hyperopt import hp

from rt_objective import objective

# Set epochs
max_epochs = 10

# Start by setting up search space
space = {
    "filepath": '/tmp/tag_data/train_mc_s.h5',
    "type": 'resnet',
    "maxConstits": 80,
    "numFolds": 5,
    "fold": None,
    "numEpochs": max_epochs,
    "stages": hp.choice('stages', [
        (hp.quniform('stage1.1', 1, 6, 1),),
        (hp.quniform('stage2.1', 1, 6, 1), hp.quniform('stage2.2', 1, 6, 1)),
        (hp.quniform('stage3.1', 1, 6, 1), hp.quniform('stage3.2', 1, 6, 1), hp.quniform('stage3.3', 1, 4, 1)),
        (hp.quniform('stage4.1', 1, 6, 1), hp.quniform('stage4.2', 1, 6, 1), hp.quniform('stage4.3', 1, 4, 1), hp.quniform('stage4.4', 1, 3, 1))
    ]),
    "dropout": hp.uniform('dropout', 0, 0.6),
    "bnMom": hp.uniform('bnMom', 0.1, 0.99),
    "learningRate": hp.loguniform('learningRate', 1e-5, 1e-2),
    "batchSize": hp.quniform('batchSize', 100, 500, 50)
}

# Attach ray cluster
ray.init(address='auto')

# Make search algorithm
algo = HyperOptSearch(
    space,
    metric='score',
    mode='min'
)
algo = ConcurrencyLimiter(algo, max_concurrent=1)

# Then run the trial
analysis = tune.run(
    objective,
    search_alg=algo,
    name='resnet',
    resume="AUTO",
    metric='score',
    mode='min',
    num_samples=2,
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
