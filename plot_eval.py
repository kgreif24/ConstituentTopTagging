""" plot_eval.py - This script will evaluate the predictions of a network
saved as a .npz file. The file must contain the model predictions, the
predictions discretized with a cut off of 0.5, the labels, and the jet pT

Author: Kevin Greif
Last updated 2/15/22
python3
"""

# Standard imports
import sys, os
import argparse

# Package imports
import sklearn.metrics as metrics
import numpy as np

# MPL imports
import matplotlib
# Matplotlib import setup to use non-GUI backend, comment for interactive
# graphics
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('~/mattyplotsalot/allpurpose.mplstyle')
import colorcet as cc


########################## Parse Arguments ###########################

parser = argparse.ArgumentParser()

parser.add_argument('--file', default=None, type=str, nargs='+',
                    help='File name of network checkpoint')
parser.add_argument('--names', default=None, type=str, nargs='+',
                    help='Names of model for plotting/saving')
args = parser.parse_args()

###################### Set parameters #######################

# Net parameters
net_file = args.file
net_name = args.names

############################# Evaluation Loop #############################

for model_num, (file, name) in enumerate(zip(net_file, net_name)):

    # Load data from file
    evaluation = np.load(file)
    preds = evaluation['preds']
    disc_preds = evaluation['disc_preds']
    labels = evaluation['labels']
    pt = evaluation['jet_pt']

    # Make a histogram of network output, separated into signal/background
    preds_sig = preds[labels == 1]
    preds_bkg = preds[labels == 0]
    hist_bins = np.linspace(0, 1.0, 100)

    plt.figure(1)
    plt.hist(preds_sig, bins=hist_bins, alpha=0.5, label='Signal')
    plt.hist(preds_bkg, bins=hist_bins, alpha=0.5, label='Background')
    plt.legend()
    plt.ylabel("Counts")
    plt.xlabel("Model output")
    plt.savefig('./outfiles/' + name + '.png', dpi=300)
    plt.clf()

    # Get total ROC curve, AUC and ACC
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
    auc = metrics.roc_auc_score(labels, preds)
    acc = metrics.accuracy_score(labels, disc_preds)
    fprinv = 1 / fpr

    # Plot inverse roc curve
    # Also want to fix color for this model
    if model_num == 0:
        # color = '#b8b8b8'
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][model_num]
    else:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][model_num]
    plt.figure(2)
    plt.plot(tpr, fprinv, label=name, color=color)

    # Find background rejection at tpr = 0.5, 0.8 working points
    wp_p5 = np.argmax(tpr > 0.5)
    wp_p8 = np.argmax(tpr > 0.8)

    # Print information on total model performance
    print("AUC score: ", auc)
    print("ACC score: ", acc)
    print("Background rejection at 0.5 signal efficiency: ", fprinv[wp_p5])
    print("Background rejection at 0.8 signal efficiency: ", fprinv[wp_p8])

    # Now we want to bin performance information into pt bins. Let's loop through
    # an array of bins. Note array defines bin edges so we want to go up to
    # len - 1
    pt_bins = np.linspace(350000, 3150000, 15)
    jet_indeces = np.arange(0, len(labels), 1)
    wp_50_array = np.zeros(len(pt_bins)-1)
    wp_80_array = np.zeros(len(pt_bins)-1)

    for i in range(len(pt_bins)-1):

        # Find indeces of predictions for jets in pt range
        condition = np.logical_and(pt > pt_bins[i], pt < pt_bins[i+1])
        bin_indeces = np.asarray(condition).nonzero()[0]

        # Now take a sub-sample of predictions within the pt bin
        bin_preds = preds[bin_indeces]
        bin_labels = labels[bin_indeces]

        # Now we want to calculate background rejection at working points
        fpr, tpr, thresholds = metrics.roc_curve(bin_labels, bin_preds)
        fprinv = 1 / fpr
        wp_p5 = np.argmax(tpr > 0.5)
        wp_p8 = np.argmax(tpr > 0.8)
        wp_50_array[i] = fprinv[wp_p5]
        wp_80_array[i] = fprinv[wp_p8]

    # Now we have binned performance information. Make a step plot!
    # Find midpoints of bins
    plt.figure(3)
    # Some funny buisness to get plotting to come out right
    wp_50_array = np.concatenate((wp_50_array, wp_50_array[-1:]))
    wp_80_array = np.concatenate((wp_80_array, wp_80_array[-1:]))
    plot_bins = pt_bins / 1000
    label5 = name + r' $\epsilon_{sig} = 0.5$'
    label8 = name + r' $\epsilon_{sig} = 0.8$'
    plt.step(plot_bins, wp_50_array, '-', color=color, where='post', label=label5)
    plt.step(plot_bins, wp_80_array, '--', color=color, where='post', label=label8)

    # Finally we want to bin roc curves and model output by pt
    # Define some new pt bins
    hand_pt_bins = np.array([350000, 1000000, 1500000, 2000000, 3000000, 14000000])

    # Loop through hand defined bins
    for i in range(len(hand_pt_bins)-1):

        # Find indeces of predictions for jets in pt range
        condition = np.logical_and(pt > hand_pt_bins[i], pt < hand_pt_bins[i+1])
        bin_indeces = np.asarray(condition).nonzero()[0]

        # Take sub-sample of predictions within pt bin
        bin_preds = preds[bin_indeces]
        bin_labels = labels[bin_indeces]

        # Split into sig/bkg
        bin_preds_sig = bin_preds[bin_labels==1]
        bin_preds_bkg = bin_preds[bin_labels==0]

        # Make model output plot for this pt bin
        plt.figure(4)
        plt.hist(bin_preds_sig, bins=hist_bins, alpha=0.5, label='Signal')
        plt.hist(bin_preds_bkg, bins=hist_bins, alpha=0.5, label='Background')
        plt.legend()
        plt.ylabel("Counts")
        plt.xlabel("Model output")
        bin_label = "[" + str(hand_pt_bins[i]/1e6) + "," + str(hand_pt_bins[i+1]/1e6) + "]"
        plt.title(bin_label + " (TeV)")
        plt.savefig('./outfiles/' + name + '_bin' + str(i) + '.png', dpi=300)
        plt.clf()

        # Now find roc curve in this pt bin
        fpr, tpr, thresholds = metrics.roc_curve(bin_labels, bin_preds)
        fprinv = 1 / fpr

        # Plot roc curve
        plt.figure(5)
        plt.plot(tpr, fprinv, label=bin_label)

    # After looping through bins, we put finishing touches to bin roc plot
    plt.figure(5)
    plt.yscale('log')
    plt.legend()
    plt.ylabel('Background rejection')
    plt.xlabel('Signal efficiency')
    plt.title(name)
    plt.savefig("./outfiles/" + name + "_binned_roc.png", dpi=300)
    plt.clf() # Make sure to clear figure for next model!


# Add finishing touches to inverse roc plot
plt.figure(2)
plt.yscale('log')
plt.legend()
plt.ylabel('Background rejection')
plt.xlabel('Signal efficiency')
plt.savefig("./outfiles/roc.png", dpi=300)

# And on binned performance plot
plt.figure(3)
plt.legend()
plt.ylabel('Background rejection')
plt.xlabel('Jet pT (GeV)')
plt.savefig("./outfiles/binned_perf.png", dpi=300)
