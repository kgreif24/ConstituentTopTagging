import os, sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_dir, ".."))

import ROOT
import array
import numpy as np
import utils


def roc_curve_binned(t_h1_sig, t_h1_bkg, side="gt", y_title="1/#varepsilon_{bkg}", sig_eff_min=0.01, n_divisions=1000, name=None):

# TODO: Must be changed to account for non-constant bin sizes!
#       Get x-array from hstogram ...
#  # New histogram with range -1, 1
#  _t_h1_sig = ROOT.TH1F("sig", "", t_h1_sig.GetNbinsX(), -1, 1)
#  _t_h1_bkg = ROOT.TH1F("bkg", "", t_h1_bkg.GetNbinsX(), -1, 1)
#
#  for b in range(0, _t_h1_sig.GetNbinsX()+2):
#    _t_h1_sig.SetBinContent(b, t_h1_sig.GetBinContent(b))
#  for b in range(0, t_h1_bkg.GetNbinsX()+2):
#    _t_h1_bkg.SetBinContent(b, t_h1_bkg.GetBinContent(b))

  # Normalize histograms
  _t_h1_sig = utils.norm_h(t_h1_sig)
  _t_h1_bkg = utils.norm_h(t_h1_bkg)

  # For the binned case, the number of devisions is the same as the number of bins
  if not n_divisions:
    n_divisions = min(_t_h1_sig.GetNbinsX(), _t_h1_bkg.GetNbinsX())

  # Set containers
  epsilon_sig = array.array("f", [0]*n_divisions)
  epsilon_bkg = array.array("f", [0]*n_divisions)

  # Fix some values
  epsilon_sig[0], epsilon_sig[n_divisions-1] = 0.0, 1.0
  epsilon_bkg[0], epsilon_bkg[n_divisions-1] = 1.0, 0.0

  # Loop over all threshold values
  eps = 0.0001
  for i, th in enumerate(np.linspace(sig_eff_min, 1, n_divisions)):

    # Compute efficiency in signal and background
    if side == "gt":
      epsilon_sig[i] = _t_h1_sig.Integral(_t_h1_sig.FindFixBin(th) - 1, _t_h1_sig.GetNbinsX() + 1)
      epsilon_bkg[i] = _t_h1_bkg.Integral(t_h1_bkg.FindFixBin(th) - 1, _t_h1_bkg.GetNbinsX() + 1)
    elif side == "lt":
      epsilon_sig[i] = _t_h1_sig.Integral(0, _t_h1_sig.FindFixBin(th) + 1)
      epsilon_bkg[i] = _t_h1_bkg.Integral(0, _t_h1_bkg.FindFixBin(th)+  1)

    if y_title == "1-#varepsilon_{bkg}":
      epsilon_bkg[i] = 1.0 - epsilon_bkg[i]
    elif y_title == "1/#varepsilon_{bkg}":
      if epsilon_sig[i] > sig_eff_min:
        if epsilon_bkg[i] > 0:
          epsilon_bkg[i] = 1/epsilon_bkg[i]
        else:
          epsilon_bkg[i] = 0
      else:
        epsilon_bkg[i] = 0
        epsilon_sig[i] = sig_eff_min

  tG = ROOT.TGraph(len(epsilon_sig), epsilon_sig, epsilon_bkg)
  if name: tG.SetName(name)

  # Set some basic properties
  if y_title == "#varepsilon_{bkg}":
    tG.SetTitle(";Signal efficiency #varepsilon_{sig};Background #varepsilon_{bkg};")
  else:
    tG.SetTitle(";Signal efficiency #varepsilon_{sig};Background rejection %s;" % y_title)
  tG.SetLineWidth(3)

  return tG


def arr2roc(self, pw_sig, pw_bkg, y_title="1/#varepsilon_{bkg}", n_divisions=1000, sig_eff_min=0.01, name=None):

  p_sig, w_sig = pw_sig
  p_bkg, w_bkg = pw_bkg

  # Set containers
  epsilon_sig = array.array("f", [0]*n_divisions)
  epsilon_bkg = array.array("f", [0]*n_divisions)

  # Fix some values
  epsilon_sig[0], epsilon_sig[n_divisions-1] = 0.0, 1.0
  epsilon_bkg[0], epsilon_bkg[n_divisions-1] = 1.0, 0.0

  # Loop over all threshold values
  for i, th in enumerate(np.linspace(0, 1, n_divisions)):

    # Get mask of events that pass th
    msk_sig_pass = p_sig > th
    msk_bkg_pass = p_bkg > th

    # Get efficiency values
    if weight_name != "":
      epsilon_sig[i] = np.sum(w_sig[msk_sig_pass]) / w_sig.sum()
      epsilon_bkg[i] = np.sum(w_bkg[msk_bkg_pass]) / w_bkg.sum()
    else:
      epsilon_sig[i] = len(p_sig[msk_sig_pass]) / float(len(p_sig))
      epsilon_bkg[i] = len(p_bkg[msk_bkg_pass]) / float(len(p_bkg))

    if y_title == "1-#varepsilon_{bkg}":
      epsilon_bkg[i] = 1.0 - epsilon_bkg[i]
    elif y_title == "1/#varepsilon_{bkg}":
      epsilon_bkg[i] = 1.0 / epsilon_bkg[i] if epsilon_bkg[i] else 0

  # Initialize graph to hold data
  tG = ROOT.TGraph(len(epsilon_sig), epsilon_sig, epsilon_bkg)

  # Set some basic properties
  if y_title == "#varepsilon_{bkg}":
    tG.SetTitle(";Signal efficiency #varepsilon_{sig};Background #varepsilon_{bkg};")
  else:
    tG.SetTitle(";Signal efficiency #varepsilon_{sig};Background rejection %s;" % y_title)
  tG.SetLineWidth(3)

  if name: tG.SetName(name)

  return tG


def rej_bkg_from_roc(roc_list, working_point):

  tMg = ROOT.TMultiGraph("rejBkg", "")

  if not isinstance(roc_list, ROOT.TMultiGraph):
    if not isinstance(roc_list, list):
      roc_list = [roc_list]
    for g in roc_list: tMg.Add(g.Clone())
  else:
    tMg = roc_list.Clone("clone_" + roc_list.GetName())

  # Get number of entries that corresponds to the number of bins
  n_entries = tMg.GetListOfGraphs().GetEntries()

  # List with rejection values to return
  rej_list = []

  # Loop over all roc curves and jet bkg rejection for working point
  for i_g in range(n_entries):
    # Get graph
    tG = tMg.GetListOfGraphs().At(i_g)
    # Get corresponding backround rejection for signal efficiency
    bkg_rej = tG.Eval(working_point)
    rej_list.append(bkg_rej)

  if len(rej_list) == 0:
    return 0
  elif len(rej_list) == 1:
    return rej_list[0]
  else:
    return rej_list
