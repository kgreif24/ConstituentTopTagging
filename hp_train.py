""" hp_train.py - This script will run hyper-parameter training using
ray tune. It will set up a search space, and run the tuning job using the
trainable defined in rt_objective.py.

Authors: Kevin Greif
Last updated 1/10/22
python3
"""

from ray import tune

from rt_objective import objective


# Start by setting up search space
config = {
    "filepath": '/pub/kgreif/samples/h5dat/train_mc_s.h5',
    "type": 'hldnn',
    "maxConstits": 80,
    "numFolds": 5,
    "fold": None,
    "hidden_layers": tune.choice([2, 3, 4, 5]),
    "nodes_per_layer": tune.choice([10, 20, 40, 80]),
    "learningRate": tune.loguniform(1e-5, 1e-2),
    "batchNorm": tune.choice([True, False]),
    "dropout": tune.uniform(0, 0.2),
    "l1reg": tune.loguniform(1e-5, 1e-2),
    "numEpochs": 10,
    "batchSize": tune.choice([100, 200])
}

# Then run the trial
analysis = tune.run(
    objective,
    config=config,
    name='hldnn',
    metric='score',
    mode='min',
    num_samples=1,
    stop={'training_iteration': 5},
    resources_per_trial={'cpu': 1, 'gpu': 1},
    local_dir='./tuning/hlDNN_tune',
    verbose=1
)
