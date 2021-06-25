# Standard imports
import os, sys
import subprocess

# Scientific imports
import numpy as np
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True


def train_weights (signal, pt, n_bins=200):
  from src.hep_ml.reweight import BinsReweighter
  # Prepate array for weights
  weights = np.ones((len(signal)))
  # Reweight signal and background separately
  for sig in [0, 1]:
    # Prepare data arrays
    msk = signal == sig
    # Get subarray for current class
    original = pt[msk]
    # Get range of pT spectrum
    xmin, xmax = original.min(), original.max()
    target = np.random.rand(original.size) * (xmax - xmin) + xmin
    # Fit bins-reweighter
    reweighter = BinsReweighter(n_bins=n_bins, n_neighs=3)
    reweighter.fit(original, target=target)
    # Predict new, flat-pT weight
    weights[msk] = reweighter.predict_weights(original)
    weights[msk] /= weights[msk].mean()
  return weights

def match_weights(signal_flag, pt, n_bins=200):
  from src.hep_ml.reweight import BinsReweighter
  # Initialize weights array
  weights = np.ones((len(signal_flag)))
  # Get signal/background
  signal = pt[signal_flag == 1]
  background = pt[signal_flag == 0]
  # Now fit bins-reweighter
  reweighter = BinsReweighter(n_bins=n_bins)
  reweighter.fit(background, target=signal)
  # Predict weights
  weights_tmp = reweighter.predict_weights(background)
  weights[signal_flag == 0] = weights_tmp / weights_tmp.mean()
  return weights

def correct_weight (signal, weights):
  # Reweight signal and background separately
  for sig in [0, 1]:
    # Prepare data arrays
    msk = signal == sig
    # Get correction factors
    weights[msk] /= weights[msk].mean()
  return weights


def remove_branch (fname, treename, branches):
  if not isinstance(branches, list): branches = [branches]
  fROOT_in = ROOT.TFile(fname)
  ttree_in = fROOT_in.Get(treename)
  fname_ou = fname.replace(".root", ".clone.root")
  fROOT_ou = ROOT.TFile(fname_ou, "RECREATE")
  if fROOT_ou.IsOpen():
    print("[\033[1mINFO\033[0m] File `%s` was sucessfully created" % fROOT_ou)
  # Disable branches to drop
  for branch in branches:
    print("[\033[1mINFO\033[0m] Branch `%s` has been disabled" % branch)
    ttree_in.SetBranchStatus(branch, 0)
  # Clone the input tree with branches disabled
  ttree_ou = ttree_in.CloneTree(-1, "fast")
  ttree_ou.Write()
  fROOT_ou.Close()
  fROOT_in.Close()
  subprocess.call("mv %s %s" % (fname_ou, fname), shell=True)

def add_branch (fname, treename, arr, branch_name):
  import array
  fROOT = ROOT.TFile(fname, "UPDATE")
  t_tree  = fROOT.Get(treename)
  address = array.array("d",[0])
  branch  = t_tree.Branch(branch_name, address, branch_name+"/D")
  for i in xrange(len(arr)):
    t_tree.GetEntry(i)
    address[0] = arr[i]
    branch.Fill()
  # overwrite the tree in the output file and close the file
  t_tree.Write("", ROOT.TFile.kOverwrite)
  fROOT.Write()
  fROOT.Close()


def resize (arr, n_max):

  print(arr)
  ndiff = arr.size - n_max
  return np.resize(np.pad(arr, (0, np.abs(ndiff)), "constant"), n_max)


def flatten (container):

  for i in container:
    if isinstance(i, (list,tuple)):
      for j in flatten(i):
        yield j
    else:
      yield i
