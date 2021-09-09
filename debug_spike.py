""" eval_efn.py - This script will build and evaluate an energy flow
network, implemented with the energyflow package. It will make use of the
DataDumper class to quickly generate np arrays of the jet data.

Author: Kevin Greif
7/9/21
python3
"""

import sys, os

import energyflow as ef
from energyflow.archs import EFN
import tensorflow as tf
import sklearn.metrics as metrics
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('~/mattyplotsalot/allpurpose.mplstyle')
from matplotlib.colors import LogNorm
import colorcet as cc

from data_handler import DataHandler


###################### Set parameters for training #######################

# Network parameters
modelpath = "training/mEFN/run_4/checkpoints"
batch_size = 100

# Data parameters
filepath = "/pub/kgreif/samples/sample_1p5M_nbpt_test.root"
constit_branches = ['fjet_sortClusNormByPt_pt', 'fjet_sortClusCenterRotFlip_eta',
                    'fjet_sortClusCenterRot_phi', 'fjet_sortClusNormByPt_e']
hl_branches = ['fjet_Tau1_wta', 'fjet_Tau2_wta', 'fjet_Tau3_wta', 'fjet_Split12',
                  'fjet_Split23', 'fjet_ECF1', 'fjet_ECF2', 'fjet_ECF3', 'fjet_C2',
                  'fjet_D2', 'fjet_Qw']
up_branches = ['fjet_sortClus_pt', 'fjet_sortClus_eta',
               'fjet_sortClus_phi', 'fjet_sortClus_e']
extra_branches = ['fjet_m', 'fjet_eta', 'fjet_phi']
my_max_constits = 80


############################# Process Data ################################

# Only need to look at validation data, trying to diagnose spiking issue

# Now build dds and make numpy arrays
print("Building data objects...")
dh_pp = DataHandler(filepath, "FlatSubstructureJetTree", constit_branches,
                    extras=None, max_constits=my_max_constits)
dh_hl = DataHandler(filepath, "FlatSubstructureJetTree", hl_branches,
                    extras=None, max_constits=my_max_constits)
dh_up = DataHandler(filepath, "FlatSubstructureJetTree", up_branches,
                    extras=extra_branches, max_constits=my_max_constits)

pp_arrs = dh_pp.np_arrays()
hl_arrs = dh_hl.np_arrays()
up_arrs = dh_up.np_arrays()

# Delete dds to save memory
del dh_pp
del dh_hl
del dh_up

# Now split up data array into correct inputs for efn
print("\nSplitting data arrays...")
pp_data = pp_arrs[0] 
labels_pp = pp_arrs[1]
raw_labels_pp = labels_pp[:,1]

hl_data = hl_arrs[0]
labels_hl = hl_arrs[1]
raw_labels_hl = labels_hl[:,1]

up_data = up_arrs[0]
fjet_m = up_arrs[2]
fjet_eta = up_arrs[3]
fjet_phi = up_arrs[4]

# Delete arrs object to save memory
del pp_arrs
del hl_arrs
del up_arrs

############################# Load EFN ##################################

print("\nLoading EFN model...")
efn = tf.keras.models.load_model(modelpath)

############################# Evaluate EFN ##############################

preds = efn.predict([pp_data[:,:,0], pp_data[:,:,1:3]], batch_size=batch_size)
# preds = efn.predict(hl_data, batch_size=batch_size)
raw_preds = preds[:,1]

# Make a histogram of network output, separated into signal/background
preds_sig = preds[raw_labels_pp == 1,1]
preds_bkg = preds[raw_labels_pp == 0,1]
hist_bins = np.linspace(0, 1.0, 100)

plt.clf()
n, bins, patches = plt.hist(preds_sig, bins=hist_bins, alpha=0.5, label='Signal')
plt.hist(preds_bkg, bins=hist_bins, alpha=0.5, label='Background')
plt.legend()
plt.ylabel("Counts")
plt.xlabel("EFN output")
plt.title("EFN output over validation set")
plt.savefig("./debug/final_output.png", dpi=300)

print(n)
print(bins)

# Find spike in output
botOutRange = 0.48
botRange = 0.4911
topRange = 0.4912
topOutRange = 0.515
indeces = np.where(np.logical_and(raw_preds > botRange, raw_preds < topRange))[0]
botOutIndeces = np.where(np.logical_and(raw_preds > botOutRange, raw_preds < botRange))[0]
topOutIndeces = np.where(np.logical_and(raw_preds > topRange, raw_preds < topOutRange))[0]
outIndeces = np.concatenate((botOutIndeces, topOutIndeces))
print("Number in spike:", len(indeces))
print("Number around spike:", len(outIndeces))
output_spike = raw_preds[indeces]
output_out = raw_preds[outIndeces]

assert not np.intersect1d(indeces, outIndeces).size

############ Preprocessed Jets ###############

# Separate spike and not spike events
pp_spike = pp_data[indeces,:,:]
pp_out = pp_data[outIndeces,:,:]
pp_z_spike = pp_spike[:,:,0]
pp_z_out = pp_out[:,:,0]
pp_eta_spike = pp_spike[:,:,1]
pp_eta_out = pp_out[:,:,1]
pp_phi_spike = pp_spike[:,:,2]
pp_phi_out = pp_out[:,:,2]
pp_e_spike = pp_spike[:,:,3]
pp_e_out = pp_out[:,:,3]
labels_spike = labels_pp[indeces,:]
labels_out = labels_pp[outIndeces,:]
jet_m_spike = fjet_m[indeces]
jet_m_out = fjet_m[outIndeces]
jet_eta_spike = fjet_eta[indeces]
jet_eta_out = fjet_eta[outIndeces]
jet_phi_spike = fjet_phi[indeces]
jet_phi_out = fjet_phi[outIndeces]

# Plot first spike jet
fpp_z_spike = pp_z_spike[0,:]
fpp_eta_spike = pp_eta_spike[0,:]
fpp_phi_spike = pp_phi_spike[0,:]
plt.clf()
plt.hist2d(fpp_eta_spike, fpp_phi_spike, bins=100, range=[[-np.pi, np.pi],[-np.pi, np.pi]], weights=fpp_z_spike)
plt.savefig("./debug/f_spike_jet.png", dpi=300)

# Plot all eta and phi (going to leave zeros in for now)
plt.clf()
plt.hist(np.ravel(pp_eta_spike), bins=100, label='Spike', range=[-np.pi, np.pi], alpha=0.5)
plt.hist(np.ravel(pp_eta_out), bins=100, label='Out', range=[-np.pi, np.pi])
plt.yscale('log')
plt.legend()
plt.title("Constituent Eta")
plt.savefig("./debug/pp_eta.png")

plt.clf()
plt.hist(np.ravel(pp_phi_spike), bins=100, label='Spike', range=[-np.pi, np.pi], alpha=0.5)
plt.hist(np.ravel(pp_phi_out), bins=100, label='Out', range=[-np.pi, np.pi], alpha=0.5)
plt.yscale('log')
plt.legend()
plt.title("Constituent Phi")
plt.savefig("./debug/pp_phi.png")

# Plot min/max eta and phi
pp_minEta_spike = np.min(pp_eta_spike, axis=1)
pp_maxEta_spike = np.max(pp_eta_spike, axis=1)
pp_minPhi_spike = np.min(pp_phi_spike, axis=1)
pp_maxPhi_spike = np.max(pp_phi_spike, axis=1)
pp_minEta_out = np.min(pp_eta_out, axis=1)
pp_maxEta_out = np.max(pp_eta_out, axis=1)
pp_minPhi_out = np.min(pp_phi_out, axis=1)
pp_maxPhi_out = np.max(pp_phi_out, axis=1)

plt.clf()
plt.hist(pp_minEta_spike, bins=100, range=[-np.pi, np.pi])
plt.title("Minimum eta in spike")
plt.savefig("./debug/pp_minEta_spike.png")

plt.clf()
plt.hist(pp_maxEta_spike, bins=100, range=[-np.pi, np.pi])
plt.title("Maximum eta in spike")
plt.savefig("./debug/pp_maxEta_spike.png")

plt.clf()
plt.hist(pp_minPhi_spike, bins=100, range=[-np.pi, np.pi])
plt.title("Minimum phi in spike")
plt.savefig("./debug/pp_minPhi_spike.png")

plt.clf()
plt.hist(pp_maxPhi_spike, bins=100, range=[-np.pi, np.pi])
plt.title("Maximum phi in spike")
plt.savefig("./debug/pp_maxPhi_spike.png")

plt.clf()
plt.hist(pp_minEta_out, bins=100, range=[-np.pi, np.pi])
plt.title("Minimum eta outside")
plt.savefig("./debug/pp_minEta_out.png")

plt.clf()
plt.hist(pp_maxEta_out, bins=100, range=[-np.pi, np.pi])
plt.title("Maximum eta outside")
plt.savefig("./debug/pp_maxEta_out.png")

plt.clf()
plt.hist(pp_minPhi_out, bins=100, range=[-np.pi, np.pi])
plt.title("Minimum phi outside")
plt.savefig("./debug/pp_minPhi_out.png")

plt.clf()
plt.hist(pp_maxPhi_out, bins=100, range=[-np.pi, np.pi])
plt.title("Maximum phi outside")
plt.savefig("./debug/pp_maxPhi_out.png")

# Plot min/max pt and e
pp_minz_spike = np.min(pp_z_spike, axis=1)
pp_maxz_spike = np.max(pp_z_spike, axis=1)
pp_mine_spike = np.min(pp_e_spike, axis=1)
pp_maxe_spike = np.max(pp_e_spike, axis=1)
pp_minz_out = np.min(pp_z_out, axis=1)
pp_maxz_out = np.max(pp_z_out, axis=1)
pp_mine_out = np.min(pp_e_out, axis=1)
pp_maxe_out = np.max(pp_e_out, axis=1)

plt.clf()
plt.hist(pp_minz_spike, bins=100)
plt.title("Minimum pT spike")
plt.savefig("./debug/pp_minz_spike.png")

plt.clf()
plt.hist(pp_maxz_spike, bins=100)
plt.title("Maximum pT spike")
plt.savefig("./debug/pp_maxz_spike.png")

plt.clf()
plt.hist(pp_mine_spike, bins=100)
plt.title("Minimum e spike")
plt.savefig("./debug/pp_mine_spike.png")

plt.clf()
plt.hist(pp_maxe_spike, bins=100)
plt.title("Maximum e spike")
plt.savefig("./debug/pp_maxe_spike.png")

plt.clf()
plt.hist(pp_minz_out, bins=100)
plt.title("Minimum pT out")
plt.savefig("./debug/pp_minz_out.png")

plt.clf()
plt.hist(pp_maxz_out, bins=100)
plt.title("Maximum pT out")
plt.savefig("./debug/pp_maxz_out.png")

plt.clf()
plt.hist(pp_mine_out, bins=100)
plt.title("Minimum e out")
plt.savefig("./debug/pp_mine_out.png")

plt.clf()
plt.hist(pp_maxe_out, bins=100)
plt.title("Maximum e out")
plt.savefig("./debug/pp_maxe_out.png")

# Plot all constituent pt, e in and out of spike

plt.clf()
plt.hist(np.ravel(pp_z_spike), bins=100)
plt.yscale('log')
plt.title("Constituent pT in spike")
plt.savefig("./debug/pp_z_spike.png")

plt.clf()
plt.hist(np.ravel(pp_e_spike), bins=100)
plt.yscale('log')
plt.title("Constituent e in spike")
plt.savefig("./debug/pp_e_spike.png")

plt.clf()
plt.hist(np.ravel(pp_z_out), bins=100)
plt.yscale('log')
plt.title("Constituent pT out of spike")
plt.savefig("./debug/pp_z_out.png")

plt.clf()
plt.hist(np.ravel(pp_e_out), bins=100)
plt.yscale('log')
plt.title("Constituent e out of spike")
plt.savefig("./debug/pp_e_out.png")

# Plot jet mass in and out of spike
plt.clf()
plt.hist(jet_m_spike, bins=100)
plt.yscale('log')
plt.title("Jet mass in spike")
plt.savefig("./debug/jet_m_spike.png")

plt.clf()
plt.hist(jet_m_out, bins=100)
plt.yscale('log')
plt.title("Jet mass out of spike")
plt.savefig("./debug/jet_m_out.png")

# Plot number of jet consituents in and out of spike
nconstits_spike = np.count_nonzero(pp_z_spike, axis=1)
nconstits_out = np.count_nonzero(pp_z_out, axis=1)

hist_bins = np.linspace(0, 80, 81)

plt.clf()
plt.hist(nconstits_spike, bins=hist_bins)
plt.yscale('log')
plt.title("Number of constituents in spike")
plt.savefig("./debug/nconstits_spike.png")

plt.clf()
plt.hist(nconstits_out, bins=hist_bins)
plt.yscale('log')
plt.title("Number of constituents out of spike")
plt.savefig("./debug/nconstits_out.png")

# Plot all jets in and out of spike
plt.clf()
plt.hist2d(np.ravel(pp_eta_spike), np.ravel(pp_phi_spike), bins=100, range=[[-np.pi, np.pi],[-np.pi, np.pi]], norm=LogNorm())
plt.title("All jets in spike")
plt.xlabel(r"$\eta$")
plt.ylabel(r"$\phi$")
plt.colorbar()
plt.savefig("./debug/pp_spike_jet.png", dpi=300)

plt.clf()
plt.hist2d(np.ravel(pp_eta_out), np.ravel(pp_phi_out), bins=100, range=[[-np.pi, np.pi],[-np.pi, np.pi]], norm=LogNorm())
plt.title("All jets out of spike")
plt.xlabel(r"$\eta$")
plt.ylabel(r"$\phi$")
plt.colorbar()
plt.savefig("./debug/pp_out_jet.png", dpi=300)

######################## Fancy Spike Tests ###############################

# Perturbation test setup
rng = np.random.default_rng()
sigma_ang = 1e-2
sigma_pt = 5e-4
pp_eta_noise = np.zeros_like(pp_eta_spike)
pp_phi_noise = np.zeros_like(pp_phi_spike)
pp_z_noise = np.zeros_like(pp_z_spike)

# Event loops for tests that need operations that can't be vectorized
for i, constits in enumerate(nconstits_spike):

    # Draw 3 samples from a normal distribution and fill noise arrays
    pp_eta_noise[i,:] = np.concatenate((rng.normal(0, sigma_ang, size=constits), np.zeros(my_max_constits - constits)))
    pp_phi_noise[i,:] = np.concatenate((rng.normal(0, sigma_ang, size=constits), np.zeros(my_max_constits - constits)))
    pp_z_noise[i,:] = np.concatenate((rng.normal(0, sigma_pt, size=constits), np.zeros(my_max_constits - constits)))
    
# Now we can use vectorized addition to add noise to existing data
pp_eta_pert = pp_eta_spike + pp_eta_noise
pp_phi_pert = pp_phi_spike + pp_phi_noise
pp_z_pert = pp_z_spike + pp_z_noise

# Now feed through network
angular_inputs = np.dstack([pp_eta_pert, pp_phi_pert])
pert_preds = efn.predict([pp_z_pert, angular_inputs], batch_size=batch_size)
raw_pert_preds = pert_preds[:,1]

# Find fraction of outputs that remain at spike output
num_remain = np.count_nonzero(np.logical_and(raw_pert_preds > botRange, raw_pert_preds < topRange))
fraction = num_remain / len(raw_pert_preds)
print("\nPerturbation tests...")
print("Fraction retained in spike:", fraction)

# Plot perturbed outputs
plt.clf()
hist_bins = np.linspace(0, 1, 100)
n, bins, patches = plt.hist(raw_pert_preds, bins=hist_bins)
plt.ylabel("Counts")
plt.xlabel("EFN output")
plt.title("EFN output over validation set")
plt.savefig("./debug/pert_output.png", dpi=300)

# Now we pick 1st spike jet, and calculate euclidean distance between all other jets
print("\nEuclidean distance tests...")

first_jet = np.concatenate((fpp_eta_spike, fpp_phi_spike, fpp_z_spike))

other_jet_sp = np.concatenate((pp_eta_spike[1:,:], pp_phi_spike[1:,:], pp_z_spike[1:,:]), axis=1)
other_jet_out = np.concatenate((pp_eta_out, pp_phi_out, pp_z_out), axis=1)
num_other_jets_sp = other_jet_sp.shape[0]
num_other_jets_out = other_jet_out.shape[0]
first_jet_sp = np.tile(first_jet, (num_other_jets_sp, 1))
first_jet_out = np.tile(first_jet, (num_other_jets_out, 1))

ed_sp = np.mean(np.power(other_jet_sp - first_jet_sp, 2), axis=1)
ed_out = np.mean(np.power(other_jet_out - first_jet_out, 2), axis=1)

plt.clf()
plt.hist(ed_sp, bins=100, label='Spike', alpha=0.5)
plt.hist(ed_out, bins=100, label='Outside', alpha=0.5)
plt.legend()
plt.ylabel('Counts')
plt.xlabel('Euclidean Distance from 1st Spike Jet')
plt.savefig('./debug/ed_calc.png', dpi=300)

############ Unprocessed Jets ###############

# Separate out spiked events
njets = 10000
up_spike = up_data[indeces,...]
up_out = np.delete(up_data, indeces, axis=0)
up_z_spike = np.ravel(up_spike[:njets,:,0])
up_z_out = np.ravel(up_out[:njets,:,0])
up_eta_spike = np.ravel(up_spike[:njets,:,1])
up_eta_out = np.ravel(up_out[:njets,:,1])
up_phi_spike = np.ravel(up_spike[:njets,:,2])
up_phi_out = np.ravel(up_out[:njets,:,2])

# Plot raw constituent eta/phi
plt.clf()
plt.hist(up_eta_spike, bins=100, label='Spike', range=[-np.pi, np.pi], alpha=0.5)
plt.hist(up_eta_out, bins=100, label='Out', range=[-np.pi, np.pi], alpha=0.5)
axes = plt.gca()
bottom, top = axes.get_ylim()
axes.set_ylim([1, top])
plt.yscale('log')
plt.legend()
plt.title("Raw Constituent Eta")
plt.savefig("./debug/up_eta.png")

plt.clf()
plt.hist(up_phi_spike, bins=100, label='Spike', range=[-np.pi, np.pi], alpha=0.5)
plt.hist(up_phi_out, bins=100, label='Out', range=[-np.pi, np.pi], alpha=0.5)
axes = plt.gca()
bottom, top = axes.get_ylim()
axes.set_ylim([1, top])
plt.yscale('log')
plt.legend()
plt.title("Raw Constituent Phi")
plt.savefig("./debug/up_phi.png")

# Plot jet eta/phi
plt.clf()
plt.hist(jet_eta_spike, bins=100, label='Spike', range=[-np.pi, np.pi], alpha=0.5)
plt.hist(jet_eta_out, bins=100, label='Out', range=[-np.pi, np.pi], alpha=0.5)
axes = plt.gca()
bottom, top = axes.get_ylim()
axes.set_ylim([1, top])
plt.yscale('log')
plt.legend()
plt.title("Jet Eta")
plt.savefig("./debug/jet_eta.png")

plt.clf()
plt.hist(jet_phi_spike, bins=100, label='Spike', range=[-np.pi, np.pi], alpha=0.5)
plt.hist(jet_phi_out, bins=100, label='Out', range=[-np.pi, np.pi], alpha=0.5)
axes = plt.gca()
bottom, top = axes.get_ylim()
axes.set_ylim([1, top])
plt.yscale('log')
plt.legend()
plt.title("Jet Phi")
plt.savefig("./debug/jet_phi.png")

# Plot all jets to visualize distribution in eta/phi plane
plt.clf()
plt.hist2d(up_eta_spike, up_phi_spike, bins=100, range=[[-np.pi, np.pi],[-np.pi, np.pi]], norm=LogNorm())
plt.title("All jets in spike")
plt.xlabel(r"$\eta$")
plt.ylabel(r"$\phi$")
plt.colorbar()
plt.savefig("./debug/up_spike_jet.png", dpi=300)

plt.clf()
plt.hist2d(up_eta_out, up_phi_out, bins=100, range=[[-np.pi, np.pi],[-np.pi, np.pi]], norm=LogNorm())
plt.title("All jets out of spike")
plt.xlabel(r"$\eta$")
plt.ylabel(r"$\phi$")
plt.colorbar()
plt.savefig("./debug/up_out_jet.png", dpi=300)






