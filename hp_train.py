""" hp_train.py - This script will run hyper-parameter training using
ray tune. It will set up a search space, and run the tuning job using the
trainable defined in rt_objective.py.

Authors: Kevin Greif
Last updated 2/19/22
python3
"""

import ray
from ray import tune
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

from rt_objective import objective

# Set epochs
max_epochs = 30

# Start by setting up search space
space = {
    "filepath": '/tmp/tag_data/train_ln_m.h5',
    "type": 'pnet',
    "maxConstits": 80,
    "numFolds": 5,
    "fold": None,
    "numEpochs": max_epochs,
    "block_depth": hp.quniform('block_depth', 2, 4, 1),
    "blocks": hp.choice('blocks', [
        {'n_blocks': 1, 'convs': (hp.quniform('stage1.1', 128, 512, 64),)},
        {'n_blocks': 2, 'convs': (hp.quniform('stage2.1', 32, 256, 32), hp.quniform('stage2.2', 128, 512, 64))},
        {'n_blocks': 3, 'convs': (hp.quniform('stage3.1', 32, 256, 32), hp.quniform('stage3.2', 32, 256, 32), hp.quniform('stage3.3', 128, 512, 64))},
        {'n_blocks': 4, 'convs': (hp.quniform('stage4.1', 32, 256, 32), hp.quniform('stage4.2', 32, 256, 32), hp.quniform('stage4.3', 32, 256, 32), hp.quniform('stage4.4', 128, 512, 64))}
    ]),
    "nodes": hp.quniform('nodes', 50, 500, 25),
    "pooling": hp.choice('pooling', ['average', 'max']),
    "knn": hp.quniform('knn', 10, 30, 2),
    "dropout": hp.uniform('dropout', 0, 0.6),
    "bnMom": hp.uniform('bnMom', 0.1, 0.99),
    "learningRate": hp.loguniform('learningRate', -11.5, -4.6),
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
algo = ConcurrencyLimiter(algo, max_concurrent=12)

# ASHA scheduler
ash_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='score',
    mode='min',
    max_t=max_epochs,
    grace_period=6,
    reduction_factor=4
)

# Then run the trial
analysis = tune.run(
    objective,
    search_alg=algo,
    scheduler=ash_scheduler,
    name='pnet',
    resume="ERRORED_ONLY",
    num_samples=24,
    keep_checkpoints_num=1,
    checkpoint_score_attr='min-score',
    stop={'training_iteration': max_epochs},
    resources_per_trial={'cpu': 1, 'gpu': 1},
    local_dir='/pub/kgreif/tt_model_repo',
    verbose=1
)

# Exit ray
ray.shutdown()
