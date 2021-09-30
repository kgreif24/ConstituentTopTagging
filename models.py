""" models.py - This script defines a function that builds various types
of keras or energyflow models based on a passed in argument list. Meant to be
used with kf_train.py script.

Author: Kevin Greif
python3
Last updated 9/14/21
"""

import energyflow as ef
from energyflow.archs import EFN
import tensorflow as tf
import sklearn.metrics as metrics
import numpy as np


def build_model(net_type, sample_shape, arglist):
    """ build_model - This function will build and return a keras model.
    All model hyperparameters are controlled by the net_type string and the
    arguments provided in arglist. When using, make sure that the arglist
    contains all of the proper information to build the model.

    Arguments:
        net_type (str) - A string that gives the type of model to build
        sample_shape (tuple) - The shape of each jet, used to infer the shape
            of inputs to the DNN models.
        arglist (obj) - The argument parser that contains all information for
            this training run

    Returns:
        (obj) - A keras model ready to be trained
    """

    if 'dnn' in net_type:

        # First infer input shape from sample shape
        input_shape = np.prod(sample_shape)

        # Build model
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(input_shape,)))
        for layer in arglist.nodes:
            model.add(tf.keras.layers.Dense(layer, 
                                            kernel_initializer='glorot_uniform', 
                                            kernel_regularizer=tf.keras.regularizers.l1(l1=1e-3)))
            if arglist.batchNorm:
                model.add(tf.keras.layers.BatchNormalization(axis=1))
            model.add(tf.keras.layers.ReLU())
            model.add(tf.keras.layers.Dropout(arglist.dropout))
        model.add(tf.keras.layers.Dense(2, 
                                        kernel_initializer='glorot_uniform', 
                                        kernel_regularizer=tf.keras.regularizers.l1(l1=1e-3),
                                        activation='softmax'))

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc')]
        )

        # Once model is built, print summary
        model.summary()

    elif net_type == 'efn':

        model = ef.archs.EFN(
            input_dim=2,
            Phi_sizes=tuple(arglist.phisizes),
            F_sizes=tuple(arglist.fsizes),
            Phi_acts="relu",
            F_acts="relu",
            Phi_k_inits="glorot_normal",
            F_k_inits="glorot_normal",
            latent_dropout=0.0,
            F_dropouts=0.0,
            mask_val=0,
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            output_act='softmax',
            summary=True
        )

    elif net_type == 'pfn':

        model = ef.archs.PFN(
            input_dim=4,
            Phi_sizes=tuple(arglist.phisizes),
            F_sizes=tuple(arglist.fsizes),
            Phi_acts="relu",
            F_acts="relu",
            Phi_k_inits="glorot_normal",
            F_k_inits="glorot_normal",
            latent_dropout=0.0,
            F_dropouts=0.0,
            mask_val=0,
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            output_act="softmax",
            summary=True
        )

    elif net_type == 'resnet':
        
        ResNeXt50, preprocess_input = Classifiers.get('resnext50')
        res_model = ResNeXt50(include_top=False, input_shape=(224, 224, 3), weights=None)
        model = tf.keras.models.Sequential()
        model.add(res_model)
        model.add(tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last'))
        model.add(tf.keras.layers.Dense(2, activation='softmax'))
        model.summary()

    else:
        raise ValueError("Model type is not known!")

    return model


########################## Train Functions ##########################

# Here we'll define some utility function to be used in model training

def log_model_output(epoch, logs):
    """ log_model_output - This function is meant to be passed to a
    keras Lambda callback class. It is not a pure function, and will
    inherit its variables from the instance. It will plot
    the model output over the validation set.
    """

    # Run model prediction over validation set
    preds = model.predict(valid_data, batch_size=batch_size)

    # Split model output into signal and background
    preds_sig = preds[valid_labels[:,1] == 1, 1]
    preds_bkg = preds[valid_labels[:,1] == 0, 1]

    # Now log histograms to tensorboard
    with file_writer.as_default():
        tf.summary.histogram('signal', preds_sig, step=epoch)
        tf.summary.histogram('background', preds_bkg, step=epoch)

    gc.collect()
