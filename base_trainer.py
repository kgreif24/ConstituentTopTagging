""" base_trainer.py - This program will define a class that sets up and runs
network training. This bare bones routine is meant to be either used in
hyperparameter tuning, or subclassed into a more detailed routine.

Author: Kevin Greif
Last updated 1/4/22
python3
"""

import glob
import os

import numpy as np
import tensorflow as tf

from data_loader import DataLoader
import models

class BaseTrainer:

    def __init__(self, setup, datafile):
        """ __init__ - Init function for base trainer class. This function
        will essentially set up a model using the models.py framework. In
        classes that inherit from BaseTrainer, additional setup can be done in
        init function that calls this function using super.

        Arguments:
        setup (dict) - A python dict whose elements tell us how to set up
            the model. Required items are as follows:

            1. type (string) - A string that gives the type of model to train, e.g. 'pfn'
            2. batchSize (int) - The batch size to use
            3. numFolds (int) - The number of folds to partition training data into.
                This can be used to tune the train/valid split ratio
            4. fold (int) - The fold to be reserved as validation data. If None, a random
                fold is selected.
            5+. The rest of the hyperparameters needed to train the model

        or...

        setup (dict) - A python dict whose elements tell us how to load the model. Required items 
        are as above, with path to directory containing saved model in place of model type.

        datafile (string) - The location of .h5 file we will use to train

        Returns:
        None
        """

        # Set some arguments to instance variables
        self.batchSize = int(setup['batchSize'])
        self.initial_lr = setup['learningRate']
        self.numEpochs = setup['numEpochs']

        # Find fold to be used
        if isinstance(setup['fold'], int):
            fold = setup['fold']
        else:
            fold = np.random.randint(1, setup['numFolds'] + 1)

        # Build data loaders for training and validation
        self.dtrain = DataLoader(
            datafile,
            batch_size=self.batchSize,
            net_type=setup['type'],
            max_constits=setup['maxConstits'],
            num_folds=setup['numFolds'],
            this_fold=fold,
            mode='train',
            use_weights=setup['no_weights']
        )

        self.dvalid = DataLoader(
            datafile,
            batch_size=self.batchSize,
            net_type=setup['type'],
            max_constits=setup['maxConstits'],
            num_folds=setup['numFolds'],
            this_fold=fold,
            mode='valid',
            use_weights=setup['no_weights']
        )

        # Build model using models.py interface, or by simply loading tf model
        if setup['checkpoint'] != None:

            # Take the newest checkpoint
            checkpoints = glob.glob(setup['checkpoint'] + '/*')
            checkpoints.sort(key=os.path.getmtime)
            self.model = tf.keras.models.load_model(checkpoints[-1])

        else:
            self.model = models.build_model(setup, self.dtrain.sample_shape)


    def train(self, callbacks):
        """ train - This function simply runs the training routine as set up
        in the init function. It returns the entire training history object.

        Arguments:
        callbacks (list) - A list of callback objects to pass to the fit
            routine.

        Returns:
        (float) - The validation loss from this training run.
        """

        # Simply call the keras fit function on the model.
        train_hist = self.model.fit(
            self.dtrain,
            epochs=self.numEpochs,
            batch_size=self.batchSize,
            validation_data=self.dvalid,
            callbacks=callbacks,
            verbose=2
        )

        # Return train history
        return train_hist

    def prediction(self):
        """ prediction - This function will simply run the predict method of
        the model run over the valid dataset

        Arguments:
        None

        Returns:
        (array) - The array of predictions over the validation set
        """

        return self.model.predict(self.dvalid, self.batchSize, verbose=2)
