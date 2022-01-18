""" model_trainer.py - This program will define a class which trains a
stand alone model. The idea is that this class will provide advanced features
like plotting and checkpointing on top of the bare bones routine defined in
BaseTrainer.

Author: Kevin Greif
Last updated 1/4/22
python3
"""

import tensorflow as tf
import sklearn.metrics as metrics
import numpy as np

import colorcet as cc
import matplotlib
# Matplotlib import setup to use non-GUI backend, comment for interactive
# graphics
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('~/mattyplotsalot/allpurpose.mplstyle')

from base_trainer import BaseTrainer


class ModelTrainer(BaseTrainer):

    # Add init function to make plotdir and checkdir instance variables in
    # future.

    def routine(self, epochs, plotdir, checkdir, patience=20, use_schedule=False):
        """ routine - This function will execute a training routine that
        plots things like model output and loss, evaluates complex metrics,
        and takes care of early stopping and checkpointing.

        Arguments:
        epochs (int) - The number of epochs to run training
        plotdir (string) - The directory in which to store plots
        checkdir (string) - The directory in which to store checkpoints
        patience (int) - The patience to use in the early stopping callback.
        schedule (bool) - If true, use lr scheduler using keras callback

        Returns:
        None
        """

        # Evaluate model on validation set before training
        # self.predict_plot(plotdir + "/initial_output.png")

        # Build callbacks to be used during training.
        callbacks = []

        # Learning rate scheduler is specific to resnet training
        if use_schedule:
            
            def schedule(epoch, lr):
                if epoch == 10 or epoch == 20:
                    lr *= 0.1
                    print("New learning rate:", lr)
                return lr
            
            schedule_callback = tf.keras.callbacks.LearningRateScheduler(schedule)
            callbacks.append(schedule_callback)
        
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min'
        )
        callbacks.append(earlystop_callback)

        check_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkdir,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )
        callbacks.append(check_callback)

        # Run training routine
        train_hist = self.train(epochs, callbacks)

        # Plot losses
        plt.clf()
        plt.plot(train_hist.history['loss'], label='Training')
        plt.plot(train_hist.history['val_loss'], label='Validation')
        plt.title("Loss for model training")
        plt.legend()
        plt.ylabel("Crossentropy loss")
        plt.xlabel("Epoch")
        plt.savefig(plotdir + "/loss.png", dpi=300)

        # Run evaluation routine
        self.evaluation(checkdir, plotdir)

    def predict_plot(self, filename):
        """ predict_plot - This function calls the prediction method and also
        generates plots of the model output that are stored at the given filename.

        Arguments:
        directory (string) - Name with which to save file

        Returns:
        (array), (array) - The array of predictions, labels over the validation
            set
        """

        # Run prediction
        preds = self.prediction()

        # Make plot using dvalid's labels
        labels_vec = self.dvalid.file['labels'][self.dvalid.indeces]
        preds_sig = preds[labels_vec == 1]
        preds_bkg = preds[labels_vec == 0]
        hist_bins = np.linspace(0, 1.0, 100)

        plt.clf()
        plt.hist(preds_sig, bins=hist_bins, alpha=0.5, label='Signal')
        plt.hist(preds_bkg, bins=hist_bins, alpha=0.5, label='Background')
        plt.legend()
        plt.ylabel("Counts")
        plt.xlabel("Model output")
        plt.title("Model output over validation set")
        plt.savefig(filename, dpi=300)

        return preds, labels_vec

    def evaluation(self, checkdir, plotdir):
        """ evaluation - This function will perform evaluation on a model
        stored in the directory given by checkdir. It will make plots and
        store them in the plotdir directory, as well as print out information.

        Arguments:
        checkdir (string) - The location of the saved model to evaluate
        plotdir (string) - The directory in which to save plots

        Returns:
        None
        """

        # Load model
        self.model = tf.keras.models.load_model(checkdir)

        # Run predict_plot function, calculate discrete predictions
        preds, labels = self.predict_plot(plotdir + "/final_output.png")
        disc_preds = (preds > 0.5).astype(int)

        # Get ROC curve and AUC
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
        auc = metrics.roc_auc_score(labels, preds)
        acc = metrics.accuracy_score(labels, disc_preds)
        fprinv = 1 / fpr

        # Find background rejection at tpr = 0.5, 0.8 working points
        wp_p5 = np.argmax(tpr > 0.5)
        wp_p8 = np.argmax(tpr > 0.8)

        # Finally print information on model performance
        print("AUC score: ", auc)
        print("ACC score: ", acc)
        print("Background rejection at 0.5 signal efficiency: ", fprinv[wp_p5])
        print("Background rejection at 0.8 signal efficiency: ", fprinv[wp_p8])
