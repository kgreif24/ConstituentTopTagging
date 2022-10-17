""" models.py - This script defines a function that builds various types
of keras or energyflow models based on a passed in argument list. Meant to be
used with kf_train.py script and ModelTrainer class.

Author: Kevin Greif
python3
Last updated 1/4/22
"""

import collections

from energyflow.archs import EFN, PFN
from classification_models.tfkeras import Classifiers
import tensorflow as tf
import numpy as np

from classification_models.tfkeras import Classifiers
from pnet.tf_keras_model import get_particle_net


def build_model(setup, sample_shape, summary=True):
    """ build_model - This function will build and return a keras model.
    All model hyperparameters are controlled by the arguments provided in
    arglist. When using, make sure that the arglist contains all of the proper
    information to build the model.

    Arguments:
        setup (dict) - The dictionary that contains all of the variables
            necessary to build and compile the tensorflow model.
        sample_shape (tuple) - The shape of each data point that will be fed
            to the model.
        summary (bool) - If true, print a summary of the model

    Returns:
        (obj) - A keras model ready to be trained
    """

    if 'dnn' in setup['type']:

        # First infer input shape from sample shape
        input_shape = np.prod(sample_shape)

        # Build model
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(input_shape,)))
        if setup['batchNorm']:
            model.add(tf.keras.layers.BatchNormalization(axis=1))
        for layer in setup['nodes']:
            model.add(tf.keras.layers.Dense(
                layer,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=tf.keras.regularizers.l1(l1=setup['l1reg']))
            )
            if setup['batchNorm']:
                model.add(tf.keras.layers.BatchNormalization(axis=1))
            model.add(tf.keras.layers.ReLU())
            model.add(tf.keras.layers.Dropout(setup['dropout']))
        model.add(tf.keras.layers.Dense(
            1,
            kernel_initializer='glorot_uniform',
            # kernel_regularizer=tf.keras.regularizers.l1(l1=setup['l1reg']),
            activation='sigmoid')
        )

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=setup['learningRate']),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.BinaryAccuracy(name='acc')]
        )

        # Once model is built, print summary
        if summary:
            model.summary()

    elif setup['type'] == 'efn':

        model = EFN(
            input_dim=5,
            Phi_sizes=tuple(setup['phisizes']),
            F_sizes=tuple(setup['fsizes']),
            Phi_acts="relu",
            F_acts="relu",
            Phi_k_inits="glorot_normal",
            F_k_inits="glorot_normal",
            latent_dropout=0.0,
            F_dropouts=setup['dropout'],
            mask_val=0,
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=setup['learningRate']),
            output_dim=1,
            output_act='sigmoid',
            summary=summary
        )

    elif setup['type'] == 'pfn':

        model = PFN(
            input_dim=10,
            Phi_sizes=tuple(setup['phisizes']),
            F_sizes=tuple(setup['fsizes']),
            Phi_acts="relu",
            F_acts="relu",
            Phi_k_inits="glorot_normal",
            F_k_inits="glorot_normal",
            latent_dropout=setup['latent_dropout'],
            F_dropouts=setup['dropout'],
            Phi_l2_regs=0,
            F_l2_regs=0,
            mask_val=0,
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=setup['learningRate']),
            output_dim=1,
            output_act="sigmoid",
            summary=summary
        )

    elif setup['type'] == 'resnet':

        # Make param named tuple for resnet
        ModelParams = collections.namedtuple(
            'ModelParams',
            ['model_name', 'repetitions', 'attention']
        )
        params = ModelParams('resnet50', setup['nodes'], None)

        # Load ResNet 50 from image classifiers
        input_tens = tf.keras.layers.Input(shape=sample_shape)
        ResNet50, preprocess_input = Classifiers.get('resnet50')
        model = ResNet50(params,
                         input_shape=sample_shape,
                         input_tensor=input_tens,
                         include_top=False,
                         classes=1,
                         weights=None,
                         dropout=setup['dropout'],
                         bn_mom=setup['bnMom'])

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=setup['learningRate']),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.BinaryAccuracy(name='acc')]
        )

        # Summary for this model is stupidly long, just print number of parameters
        if summary:
            print("Model parameters:", model.count_params())
            print("Initial learning rate:", setup['learningRate'])

    elif setup['type'] == 'pnet':

        # Get particle net model
        shapes = {'points': (sample_shape[0],2), 'features': sample_shape, 'mask': (sample_shape[0],1)}
        model = get_particle_net(1, shapes, setup)

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=setup['learningRate']),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.BinaryAccuracy(name='acc')]
        )
        
        if summary:
            model.summary()

    else:
        raise ValueError("Model type is not known!")

    return model
