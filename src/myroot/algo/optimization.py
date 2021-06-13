import os, sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_dir, "../../"))

import ROOT
import numpy as np
import myroot.utils


# Number of bins used in the optimization
__n_bins__ = 400

# Correct errors
ROOT.TH1.SetDefaultSumw2()


def _check_wp(working_point):

  # Check if `wp` is valid
  wp = float(working_point)
  if (wp > 1) and (wp < 100):
    print("[WARNING] The provided wp=`%s` is larger that 1 but smaller than 100" % wp)
    wp /= 100.
    print(" It appears that the wp was given in per cent; hence, the wp used for evaluation is wp=`%s`" % wp)
    print(" It's recommended not to use per cent for the wp, but provide a value within the range [0,1]")
  return wp


def get_cut_on_sig(t_h1_sig, working_point, side):

  if (side != "lt") and (side != "gt"):
    print("[ERROR] No valide argument for `side` was provided")
    sys.exit()

  # Get standardized working point
  wp = _check_wp(working_point)

  # Check if histogram is normalized
  _t_h1_sig = myroot.utils.norm_h(t_h1_sig)

  # Get the cut on the variable in t_h1_sig
  n_bins = _t_h1_sig.GetNbinsX()
  # Save efficiency for each bin
  eff = {}
  deff = 0
  for n_bin in range(1, n_bins+1):
    if side == "gt":
      tmp_frac = _t_h1_sig.Integral(n_bin, n_bins + 1)
    elif side == "lt":
      tmp_frac = _t_h1_sig.Integral(0, n_bins - n_bin + 1)
    # Set eff
    eff[n_bin+1] = tmp_frac
    # If signal eff falls below wp, terminate loop
    if (tmp_frac < wp):
      # If the difference `|eff(n_bin)-wp|` is smaller than `|eff(n_bin+1)-wp|`, take the first one
      if np.abs(eff[n_bin] - wp) < np.abs(eff[n_bin+1] - wp):
        return _t_h1_sig.GetXaxis().GetBinCenter(n_bin), 10E-09
      elif np.abs(eff[n_bin-1] - wp) < np.abs(eff[n_bin] - wp):
        return _t_h1_sig.GetXaxis().GetBinCenter(n_bin-1), 10E-09
      else:
        return _t_h1_sig.GetXaxis().GetBinCenter(n_bin+1), 10E-09
  return 0, 0


def cut2D(t_h2_sig, t_h2_bkg, var1, var2, working_point):

  # Get standardized working point
  wp = _check_wp(working_point)
  # Initial values
  best_sig_eff, best_bkg_eff = 1.0, 1.0
  i_var_best_cut, j_var_best_cut = 0.0, 0.0

  # Set number of bins
  assert t_h2_sig.GetNbinsY() == t_h2_bkg.GetNbinsY(), "Signal and background must have the same number of bins"
  n_bins = t_h2_sig.GetNbinsY()

  # First cut loop
  for i_var_cut in range(1, n_bins + 1):

    # Create projection to X
    pxfull = t_h2_bkg.ProjectionX("pxfull")

    # apply upper or lower cut
    if var1["side"] == "gt":
      sig_proj = t_h2_sig.ProjectionY("sig_proj", i_var_cut, pxfull.GetNbinsX() + 1)
      bkg_proj = t_h2_bkg.ProjectionY("bkg_proj", i_var_cut, pxfull.GetNbinsX() + 1)
    elif var1["side"] == "lt":
      sig_proj = t_h2_sig.ProjectionY("sig_proj", 0, pxfull.GetNbinsX() - i_var_cut + 1)
      bkg_proj = t_h2_bkg.ProjectionY("bkg_proj", 0, pxfull.GetNbinsX() - i_var_cut + 1)

    # Set number of bins
    assert sig_proj.GetNbinsX() == bkg_proj.GetNbinsX(), "Signal and background must have the same number of bins"
    n_bins = sig_proj.GetNbinsX()

    # Determine efficiencies
    sig_integral, bkg_integral = sig_proj.Integral(0, n_bins + 1), bkg_proj.Integral(0, n_bins + 1)
    # Temporaily store efficiencies
    sig_eff, bkg_eff = sig_integral, bkg_integral

    # Stop if signal efficiency is smaller than working point
    if sig_eff < wp: break

    # Second cut loop
    for j_var_cut in range(1, n_bins + 1):

      # Apply upper or lower cut
      if var2["side"] == "gt":
        sig_integral = sig_proj.Integral(j_var_cut, n_bins)
        bkg_integral = bkg_proj.Integral(j_var_cut, n_bins)
      elif var2["side"] == "lt":
        sig_integral = sig_proj.Integral(1, n_bins - j_var_cut + 1)
        bkg_integral = bkg_proj.Integral(1, n_bins - j_var_cut + 1)

      tmp_sig_eff = sig_integral

      # stop if signal efficiency is smaller than working point
      if tmp_sig_eff < wp:

        # Determine cuts if selection imporves performance
        if bkg_eff < best_bkg_eff:

          # Set efficiencies
          best_sig_eff, best_bkg_eff = sig_eff, bkg_eff
          # Set cuts
          xmin, xmax = float(var1["xmin"]), float(var1["xmax"])
          ymin, ymax = float(var2["xmin"]), float(var2["xmax"])
          if var1["side"] == "gt":
            i_var_best_cut  = xmin + ((( xmax - xmin ) / float(n_bins) ) * float(i_var_cut - 1) )
          elif var1["side"] == "lt":
            i_var_best_cut  = xmax - ((( xmax - xmin ) / float(n_bins) ) * float(i_var_cut - 1) )
          if var2["side"] == "gt":
            j_var_best_cut  = ymin + ((( ymax - ymin ) / float(n_bins) ) * float(j_var_cut - 2) )
          elif var2["side"] == "lt":
            j_var_best_cut  = ymax - ((( ymax - ymin ) / float(n_bins) ) * float(j_var_cut - 2) )

        break

      sig_eff, bkg_eff = sig_integral, bkg_integral
    # End second cut loop
  # End first cut loop
  return best_sig_eff, best_bkg_eff, i_var_best_cut, j_var_best_cut


def eff(t_h1_sig_tot, t_h1_bkg_tot, t_h1_sig_tag, t_h1_bkg_tag, working_point=None, verbose=True):

  # Define histograms for efficiency values
  h_tmp_eff = ROOT.TH1D("h_tmp_eff", "h_tmp_eff", 2, 0, 2)
  h_tmp_eff_total = ROOT.TH1D("h_tmp_eff_total", "h_tmp_eff_total", 2, 0, 2)

  # Evaluate total event yield and error
  sig_err_tot, bkg_err_tot = ROOT.Double(), ROOT.Double()
  n_sig_tot = t_h1_sig_tot.IntegralAndError(0, t_h1_sig_tot.GetNbinsX() + 1, sig_err_tot)
  n_bkg_tot = t_h1_bkg_tot.IntegralAndError(0, t_h1_bkg_tot.GetNbinsX() + 1, bkg_err_tot)

  # Evaluate event yield after applying smooth tagger cuts and error
  sig_err_tag, bkg_err_tag = ROOT.Double(), ROOT.Double()
  n_sig_tag = t_h1_sig_tag.IntegralAndError(0, t_h1_sig_tag.GetNbinsX() + 1, sig_err_tag)
  n_bkg_tag = t_h1_bkg_tag.IntegralAndError(0, t_h1_bkg_tag.GetNbinsX() + 1, bkg_err_tag)

  # Fill efficiencies into histos
  # -- Tagged
  h_tmp_eff.SetBinContent(1, n_sig_tag)
  h_tmp_eff.SetBinError(1, sig_err_tag)
  h_tmp_eff.SetBinContent(2, n_bkg_tag)
  h_tmp_eff.SetBinError(2, bkg_err_tag)
  # -- Total
  h_tmp_eff_total.SetBinContent(1, n_sig_tot)
  h_tmp_eff_total.SetBinError(1, sig_err_tot)
  h_tmp_eff_total.SetBinContent(2, n_bkg_tot)
  h_tmp_eff_total.SetBinError(2, bkg_err_tot)

  # Divide histograms to get efficiencies
  h_tmp_eff.Divide(h_tmp_eff_total)

  # Declare variables to hols efficiency values for signal and background
  sig_eff, bkg_eff = 0.0, 0.0
  sig_err, bkg_err = 0.0, 0.0

  # Store efficiencies and its associated errors
  sig_eff = h_tmp_eff.GetBinContent(1)
  sig_err = h_tmp_eff.GetBinError(1)
  if h_tmp_eff_total.GetBinContent(2) != 0:
    bkg_eff = h_tmp_eff.GetBinContent(2)
    bkg_err = h_tmp_eff.GetBinError(2)

  if verbose:
    if working_point:
      print("[OPTIMIZATION] (EffSig = %s - EffSigErr = %s; EffBkg = %s - EffBkgEff = %s) @ wp = %s" % (sig_eff, sig_err, bkg_eff, bkg_err, working_point))
    else:
      print("[OPTIMIZATION] (EffSig = %s - EffSigErr = %s; EffBkg = %s - EffBkgEff = %s)" % (sig_eff, sig_err, bkg_eff, bkg_err))

    # Return efficiencies and errors
    return sig_eff, sig_err, bkg_eff, bkg_err


def set_nbins(n_bins):

  global __n_bins__
  __n_bins__ = n_bins
  print("[INFO] `%s` bins will be used for optimization" % __n_bins__)

