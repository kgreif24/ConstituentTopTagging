""" eval_model.py - This script will evaluate a pretrained network of type
hlDNN, DNN, EFN, or PFN. It will make use of the DataHandler class to quickly
generate np arrays of the jet data.

Author: Kevin Greif
Last updated 10/11/21
python3
"""

# Standard imports
import sys, os
import argparse

# Package imports
import energyflow as ef
from energyflow.archs import EFN
import tensorflow as tf
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

# Custom imports
from data_handler import DataHandler


# If GPU is available, print message
print("\nStart model evaluation script...")
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))


########################## Parse Arguments ###########################

parser = argparse.ArgumentParser()

parser.add_argument('--type', default=None, type=str, nargs='+',
                    help='Type of model to build (dnn, efn, pfn)')
parser.add_argument('-b', '--batchSize', default=100, type=int,
                    help='Batch size')
parser.add_argument('--maxConstits', default=80, type=int,
                    help='Number of constituents to include per event')
parser.add_argument('--file', default=None, type=str, nargs='+',
                    help='File name of network checkpoint')
parser.add_argument('--names', default=None, type=str, nargs='+',
                    help='Names of model for plotting/saving')
args = parser.parse_args()

if (len(args.type) != len(args.file)) and (len(args.type) != len(args.names)):
    raise ValueError("Length of type, names, and file command line args must match!")

###################### Set parameters for evaluation #######################

# Net parameters
net_type = args.type
net_file = args.file
net_name = args.names
batch_size = args.batchSize

# Data parameters
filepath = "/pub/kgreif/samples/sample_1p5M_v7_test.root"
constit_branches = ['fjet_sortClusNormByPt_pt', 'fjet_sortClusCenterRotFlip_eta',
                  'fjet_sortClusCenterRot_phi', 'fjet_sortClusNormByPt_e']
hl_branches = ['fjet_Tau1_wta', 'fjet_Tau2_wta', 'fjet_Tau3_wta', 'fjet_Tau4_wta', 
               'fjet_Split12', 'fjet_Split23', 'fjet_ECF1', 'fjet_ECF2', 'fjet_ECF3', 
               'fjet_C2', 'fjet_D2', 'fjet_Qw', 'fjet_L2', 'fjet_L3', 'fjet_ThrustMaj']
extra_branches = ['fjet_pt']
max_constits = args.maxConstits

############################# Process Data ################################

# Now build dhs and use them to plot all branches of interest
print("Building data object...")
constit_dh = DataHandler(filepath, "FlatSubstructureJetTree", constit_branches,
                         extras=extra_branches, max_constits=max_constits)
hl_dh = DataHandler(filepath, "FlatSubstructureJetTree", hl_branches,
                    extras=extra_branches, max_constits=0)

# Figure out sample shape
constit_sample_shape = tuple([batch_size]) + constit_dh.sample_shape()
hl_sample_shape = tuple([batch_size]) + hl_dh.sample_shape()

# Now make np arrays out of the data
print("\nBuilding data arrays...")
constit_arrs = constit_dh.np_arrays()
hl_arrs = hl_dh.np_arrays()

# Delete dhs to save memory
del constit_dh
del hl_dh

# Split data arrays common to all model types
print("\nSplitting data arrays...")
constit_data = constit_arrs[0]
constit_labels = constit_arrs[1]
constit_pt = constit_arrs[2]
constit_events = len(constit_pt)
hl_data = hl_arrs[0]
hl_labels = hl_arrs[1]
hl_pt = hl_arrs[2]
hl_events = len(hl_pt)

# Verify that there are no NaNs in data
assert not np.any(np.isnan(constit_data))
assert not np.any(np.isnan(hl_data))

############################# Evaluation Loop #############################

for model_num, (file, type, name) in enumerate(zip(net_file, net_type, net_name)):

    # Load model
    print("\nLoading model ", name)
    model = tf.keras.models.load_model(file)

    # Either way print summary
    model.summary()

    # Process data depending on model type
    if type == 'dnn':
        input_shape = np.prod(constit_sample_shape[1:])
        valid_data = constit_data.reshape((constit_events, input_shape))
        valid_labels = constit_labels
        valid_pt = constit_pt
        num_events = constit_events 
        print("Shape of validation data: ", np.shape(valid_data))
    elif type == 'hldnn':
        valid_data = hl_data
        valid_labels = hl_labels
        valid_pt = hl_pt
        num_events = hl_events
        print("Shape of validation data: ", np.shape(valid_data))
    elif type == 'efn':
        valid_data = [constit_data[:,:,0], constit_data[:,:,1:3]]
        valid_labels = constit_labels
        valid_pt = constit_pt
        num_events = constit_events
        print("Shape of validation data (features): ", len(valid_data))
    elif type == 'pfn':
        valid_data = constit_data
        valid_labels = constit_labels
        valid_pt = constit_pt
        num_evens = constit_events
        print("Shape of validation data:", np.shape(valid_data))
    else:
        raise ValueError("Model type not recognized!")

    # Evaluate model
    print("\nEvaluation for ", name)
    preds = model.predict(valid_data, batch_size=batch_size)

    # Make a histogram of network output, separated into signal/background
    preds_sig = preds[valid_labels[:,1] == 1]
    preds_bkg = preds[valid_labels[:,1] == 0]
    hist_bins = np.linspace(0, 1.0, 100)

    plt.figure(1)
    plt.hist(preds_sig[:,1], bins=hist_bins, alpha=0.5, label='Signal')
    plt.hist(preds_bkg[:,1], bins=hist_bins, alpha=0.5, label='Background')
    plt.legend()
    plt.ylabel("Counts")
    plt.xlabel("Model output")
    plt.savefig('./outfiles/' + name + '.png', dpi=300)
    plt.clf()

    # Get total ROC curve and AUC
    fpr, tpr, thresholds = metrics.roc_curve(valid_labels[:,1], preds[:,1])
    auc = metrics.roc_auc_score(valid_labels[:,1], preds[:,1])
    fprinv = 1 / fpr

    # Plot inverse roc curve
    # Also want to fix color for this model
    if model_num == 0:
        # color = '#b8b8b8'
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][model_num]
    else:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][model_num-1]
    plt.figure(2)
    plt.plot(tpr, fprinv, label=name, color=color)

    # Find background rejection at tpr = 0.5, 0.8 working points
    wp_p5 = np.argmax(tpr > 0.5)
    wp_p8 = np.argmax(tpr > 0.8)

    # Print information on total model performance
    print("Background rejection at 0.5 signal efficiency: ", fprinv[wp_p5])
    print("Background rejection at 0.8 signal efficiency: ", fprinv[wp_p8])
    print("AUC score: ", auc)

    # Now we want to bin performance information into pt bins. Let's loop through
    # an array of bins. Note array defines bin edges so we want to go up to 
    # len - 1
    pt_bins = np.linspace(350000, 3150000, 15)
    jet_indeces = np.arange(0, num_events, 1)
    wp_50_array = np.zeros(len(pt_bins)-1)
    wp_80_array = np.zeros(len(pt_bins)-1)

    for i in range(len(pt_bins)-1):
        
        # Find indeces of predictions for jets in pt range
        condition = np.logical_and(valid_pt > pt_bins[i], valid_pt < pt_bins[i+1])
        bin_indeces = np.asarray(condition).nonzero()[0]

        # Now take a sub-sample of predictions within the pt bin
        bin_preds = preds[bin_indeces,1]
        bin_labels = valid_labels[bin_indeces,1]

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
        condition = np.logical_and(valid_pt > hand_pt_bins[i], valid_pt < hand_pt_bins[i+1])
        bin_indeces = np.asarray(condition).nonzero()[0]

        # Take sub-sample of predictions within pt bin
        bin_preds = preds[bin_indeces,1]
        bin_labels = valid_labels[bin_indeces,1]

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
