#!/usr/bin/env python

import os, sys

# Scientific imports
import numpy as np
import ROOT

# Custom imports
import common
import myroot
import myroot.cut
import myutils.walker
import myutils.name
import myplt, myhep

from myplt.figure import Figure
from myroot.convert import conv2vec
from myroot.algo.optimization import eff
from myroot.algo.performance import roc_curve_binned, rej_bkg_from_roc


"""
  Some global configurations
"""

ROOT.TH1.SetDefaultSumw2()

# Path to this script
path2home = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Read config file
__sec_train, __sec_valid, __sec_meta = common.read_conf(os.path.join(path2home, "etc/nomen_est_omen.ini"), "modes", ["training", "validation", "metadata"])
__sec_train__, __sec_val__, __sec_test__ = common.read_conf(os.path.join(path2home, "etc/nomen_est_omen.ini"), "modes", ["training", "validation", "testing"])


def _cut_on_score (histos, pt_bins, working_point, hname):

  t_h1_cut = ROOT.TH1F(hname, hname, len(pt_bins)-1, pt_bins)
  # Get the cut on the dnn score in each pt bin
  from myroot.algo.optimization import get_cut_on_sig
  for i_bin in range(len(pt_bins)-1):
    # Get pT range
    pt_low, pt_high = myroot.cut.get_limits(i_bin, pt_bins, rtype=int, verbose=False)
    if (histos[i_bin]["Sig"].Integral() < 1E-10) or (histos[i_bin]["Sig"].GetEntries == 0):
      continue
    cut_score, cut_err = get_cut_on_sig(histos[i_bin]["Sig"], working_point, "gt")
    print("[EVAL] \033[1mlower cut=%s\033[0m - pT=[%s, %s] - wp=%s" % (cut_score, pt_low, pt_high, int(working_point*100)))
    t_h1_cut.SetBinContent(i_bin+1, cut_score)
    t_h1_cut.SetBinError(i_bin+1, cut_err)
  return t_h1_cut


def _fit_on_score (t_h1_cut, orders, wp100_str):

  from myroot.algo.fit import poly1D
  fits = []
  for order in orders:
    # Fit this histogram to get a smooth function of pT
    fits.append(poly1D(t_h1_cut, order))
    # Change name
    fits[-1].SetName( "fit_smoothCutDnnScore_wp%s_order_%s" % (wp100_str, str(order)))
    fits[-1].SetTitle("%s,,l" % order)
  return fits


def _norm (histos):

  # Normalize all histograms
  for i_bin in histos:
    for h in histos[i_bin]:
      histos[i_bin]["Sig"].Scale(1.0 / histos[i_bin]["Sig"].Integral())
      histos[i_bin]["Bkg"].Scale(1.0 / histos[i_bin]["Bkg"].Integral())


def _plt_score (h_sig, h_bkg, pt_low, pt_high, fname):

  myplt.settings.common_head_reset()
  myplt.settings.common_head() \
    .addRow(text="#sqrt{s}=13 TeV,  |#eta|<2.0,  m>40 GeV,  p_{T} #in [%s, %s] (GeV)" % (pt_low, pt_high)) \
    .setTextAlign("left")
  # Save the figure ...
  Figure(canv=(600,500)) \
    .setDrawOption("PLC PMC HIST") \
    .addHist([h_sig, h_bkg]) \
    .addAxisTitle(title_x="DNN classification score") \
    .addLegend(entries=["Signal;l", "Background;l"]) \
    .setLogY() \
    .setLogY(range_y=(1E-4, 1E0)) \
    .save(fname).close()


def _plt_score_vs_pt (h2, fname):

  myplt.settings.common_head_reset()
  myplt.settings.common_head() \
    .addRow(text="#sqrt{s}=13 TeV,  |#eta|<2.0,  m>40 GeV") \
    .setTextAlign("left")
  # Save the figure ...
  Figure(canv=(600,500)) \
    .setColorPalette(103) \
    .setDrawOption("COLZ") \
    .addHist(h2) \
    .addAxisTitle(
       title_x="Transverse momentum p^{true}_{T} [GeV]",
       title_y="DNN classification score",
       title_z="Number of weighted events") \
    .setPadMargin() \
    .setLogZ(0, h2.GetMaximum()) \
    .save(fname).close()


def _plt_roc_family (graphs, fname):

  myplt.settings.common_head_reset()
  myplt.settings.common_head() \
    .addRow(text="#sqrt{s}=13 TeV,  |#eta|<2.0,  m>40 GeV") \
    .setTextAlign("left")

  # Plot list of fitting curves
  Figure(canv=(600,500)) \
    .setDrawOption("AL PLC PMC") \
    .setColorPalette(87) \
    .addHist(graphs) \
    .addAxisTitle(
      title_x="Signal efficiency #varepsilon_{sig}",
      title_y="Background rejection 1/#varepsilon_{bkg}") \
    .setLogY(range_y=(1,1E5)) \
    .setRangeX(0.05,1.1) \
    .addLegend(xmin=0.8, ymin=0.12, xmax=1.13, ymax=0.925) \
    .setPadMargin(margin=0.2, side="right") \
    .save(fname).close()


def _plt_roc (graph, pt_low, pt_high, fname):

  myplt.settings.common_head_reset()
  myplt.settings.common_head() \
    .addRow(text="#sqrt{s}=13 TeV,  |#eta|<2.0,  m>40 GeV,  p_{T} #in [%s, %s] (GeV)" % (pt_low, pt_high)) \
    .setTextAlign("left")
  # Plot ROC curve
  Figure(canv=(600,500)) \
    .setDrawOption("AL PLC PMC") \
    .addHist(graph) \
    .addAxisTitle(
      title_x="Signal efficiency #varepsilon_{sig}",
      title_y="Background rejection 1/#varepsilon_{bkg}") \
    .setLogY(range_y=(1,1E5)) \
    .setRangeX(0.05,1.1) \
    .save(fname).close()


def _plt_cut (histo, wp100_str, fname):

  # Save the figure ...
  myplt.settings.common_head_reset()
  myplt.settings.common_head() \
    .addRow(text="#sqrt{s}=13 TeV,  |#eta|<2.0") \
    .setTextAlign("left")
  Figure(canv=(600,500)) \
    .setDrawOption("PLC PMC HIST") \
    .addHist(histo) \
    .addAxisTitle(
      title_x="Transverse momentum p^{true}_{T} [GeV]",
      title_y="Lower DNN-score cut @ #varepsilon_{sig}=" + wp100_str + "%") \
    .save(fname).close()


def _plt_fits (fits, wp100_str, fname):

  myplt.settings.common_head_reset()
  myplt.settings.common_head() \
    .addRow(text="#sqrt{s}=13 TeV,  |#eta|<2.0,  m>40 GeV,  DNN top tagger").setTextAlign("left")
  Figure(canv=(600,500)) \
    .setDrawOption("PLC PMC") \
    .addGraph(fits, range_y=(1.2*(1-float(wp100_str)/100.), 1.2)) \
    .addAxisTitle(
      title_x="Transverse momentum p^{true}_{T} [GeV]",
      title_y="Lower DNN-score cut @ #varepsilon_{sig}=" + wp100_str + "%") \
      .addLegend(title="#bf{Polynomial order}") \
    .save(fname).close()


def _plt_rej_bkg (h_bkg_rej, wp100_str, fname):

  myplt.settings.common_head_reset()
  myplt.settings.common_head() \
    .addRow(text="#sqrt{s}=13 TeV,  |#eta|<2.0,  m>40 GeV,  DNN top tagger").setTextAlign("left")
  # - Background rejection
  Figure(canv=(600,500)) \
    .setDrawOption("PLC PMC HIST") \
    .addHist(h_bkg_rej) \
    .addAxisTitle(
      title_x="Transverse momentum p^{true}_{T} [GeV]",
      title_y="Background rejection 1/#varepsilon_{bkg} @ #varepsilon_{sig}=" + wp100_str + "%") \
    .save(fname).close()


class EvalQuick (myutils.walker.Walker, myroot.fio.FileIo):

  def __init__ (self, datagen, model, outdir="evalquick"):

    self.__dict__.update(locals())
    # Initialize manager objects
    myutils.walker.Walker.__init__(self, outdir, write_mode="update")
    myroot.fio.FileIo.__init__(self, self.get("home"), write_mode="update")
    # Add files
    for fname in ["InputQuick.root", "OutputQuick.root", "FitFunctionsQuick.root"]:
      self.addFile(fname, write_mode="update")


  def setPtBins(self, pt_binning):

    import array
    self.pt_bins = array.array("d", pt_binning)
    return self


  def efnLatentSpace (self, efn, R=1, n=100, colors=["Reds", "Oranges", "Greens", "Blues", "Purples", "Greys"]):

    # Only valid for EFN; check if valid
    grads = np.linspace(0.45, 0.55, 4)
    # evaluate filters
    import matplotlib.pyplot as plt
    from energyflow.archs import PFN, EFN
    Phi_sizes, F_sizes = (200, 200, 200), (200, 200, 200)
    model = EFN(input_dim=2, Phi_sizes=Phi_sizes, F_sizes=F_sizes)
    model.load_weights(os.path.join("out/AK10UFOSD/rel22p0/EF/EFN/Phi200-200-200_F200-200-200/", "conf/weights.h5"))

    X, Y, Z = model.eval_filters(R, n=n)
    # Plot
    fig, axes = plt.subplots(1, 1, figsize=(8,8))
    # plot filters
    for i,z in enumerate(Z):
      axes.contourf(X, Y, z/np.max(z), grads, cmap=colors[i % len(colors)])
    axes.set_xticks(np.linspace(-R, R, 5))
    axes.set_yticks(np.linspace(-R, R, 5))
    axes.set_xticklabels(['-1', '-1/2', '0', '1/2', '1'])
    axes.set_yticklabels(['-1', '-1/2', '0', '1/2', '1'])
    axes.set_xlabel('Shifted pseudorapidity $\eta$')
    axes.set_ylabel('Shifted azimuthal angle $\phi$')
    axes.set_title('Anti-$k_t$ R=1.0 UFO jets, SD($\\beta=1.0, z_\mathrm{cut}=0.1$), CS+SK, $\sqrt{s}=13\,\mathrm{TeV}$, $|\eta|<2$', fontdict={'fontsize': 12})
    plt.savefig(os.path.join(self.get("home"), "plt/contour_trained.pdf"))
    return self


  def cutOnScore(self, working_point, batch_size=None, n_points=-1, n_bins=100, orders=[4,5,6,7,8,9,10]):

    # Some checks on input
    if not isinstance(orders, list): orders = [orders]

    # Current working point as string
    wp_str, wp100_str = str(working_point), str(int(working_point*100.))

    """
      1. Prepare a dictionary that defines a histogram fro each pT bin for
         signal and background
    """

    t_h2_sig = ROOT.TH2F("pt_vs_score_sig", "", len(self.pt_bins)-1, self.pt_bins, n_bins, 0, 1)
    t_h2_bkg = ROOT.TH2F("pt_vs_score_bkg", "", len(self.pt_bins)-1, self.pt_bins, n_bins, 0, 1)

    # Get data generator
    gen_valid = self.datagen.gen("valid", batch_size, n_points=n_points)

    # Get prediction by model
    count = 0
    for X, signal, _, pt, w in gen_valid():
      if count > gen_valid.n_batches: break
      score = self.model.predict(X)[:,1]
      # Split into sig + bkg
      msk = signal[:,1] == 1
      score_sig, pt_sig, w_sig = score[msk], pt[msk], w[msk]
      score_bkg, pt_bkg, w_bkg = score[~msk], pt[~msk], w[~msk]
      # Fill histograms for both processes
      [t_h2_sig.Fill(xx/1000., yy, ww) for xx, yy, ww in zip(pt_sig, score_sig, w_sig)]
      [t_h2_bkg.Fill(xx/1000., yy, ww) for xx, yy, ww in zip(pt_bkg, score_bkg, w_bkg)]
      count += 1

    # Dict for histograms
    histos = {}
    for i_bin in range(t_h2_sig.GetNbinsX()):
      pt_low, pt_high = myroot.cut.get_limits(i_bin, self.pt_bins, rtype=int, verbose=False)
      histos[i_bin] = \
      {
        "Sig": t_h2_sig.ProjectionY("dnn_score_sig_pt_%s-%s" % (pt_low, pt_high), i_bin+1, i_bin+2),
        "Bkg": t_h2_bkg.ProjectionY("dnn_score_bkg_pt_%s-%s" % (pt_low, pt_high), i_bin+1, i_bin+2)
      }
      self.addObj([ histos[i_bin]["Sig"],  histos[i_bin]["Bkg"]], "InputQuick.root")

    # Normalize all histograms
    _norm(histos)

    for i_bin in range(t_h2_sig.GetNbinsX()):
      pt_low, pt_high = myroot.cut.get_limits(i_bin, self.pt_bins, rtype=int, verbose=False)
      self.addObj([ histos[i_bin]["Sig"],  histos[i_bin]["Bkg"]], "InputQuick.root")
      _plt_score (histos[i_bin]["Sig"], histos[i_bin]["Bkg"], pt_low, pt_high,
        os.path.join(self.get("home"), "plt/score/dnnScoreQick.pt_%s-%s" % (pt_low, pt_high)))

    # Histogram to hold cuts for each pT bin
    t_h1_cut = _cut_on_score(histos, self.pt_bins, working_point, "dnn_dnnScoreCutLow_wp%s" % wp100_str)
    self.addObj(t_h1_cut, "OutputQuick.root")

    # Fit discrete weights to get a smooth function of pT
    fits = _fit_on_score(t_h1_cut, orders, wp100_str)
    self.addObj(fits, "FitFunctionsQuick.root")

    """
      Save correlations
    """

    _plt_score_vs_pt(t_h2_sig, os.path.join(self.get("home"), "plt/pt_vs_dnnScore_sig_quick"))
    _plt_score_vs_pt(t_h2_bkg, os.path.join(self.get("home"), "plt/pt_vs_dnnScore_bkg_quick"))
    self.addObj([t_h2_sig, t_h2_bkg], "OutputQuick.root")

    """
      Roc Curves
    """

    # List of graphs (ROC curves)
    graphs = []
    # Save ROC curves
    for i_bin in range(t_h2_sig.GetNbinsX()):
      graphs.append(roc_curve_binned(histos[i_bin]["Sig"], histos[i_bin]["Bkg"], sig_eff_min=0))
      pt_low, pt_high = myroot.cut.get_limits(i_bin, self.pt_bins, rtype=int, verbose=False)
      # Rename roc curve and save it to file
      graphs[-1].SetName("rocCurve_pt%s-%s" % (pt_low, pt_high))
      graphs[-1].SetTitle("#scale[0.7]{[%s, %s]},,l" % (pt_low, pt_high))
      self.addObj(graphs, "OutputQuick.root")
    # Plot family of ROC curves
    _plt_roc_family(graphs, os.path.join(self.get("home"), "plt/roc_family_quick"))


    """
      Background rejection
    """

    hname_eff = myutils.name.encode([("v", "dnn"), ("d", "effSig"), ("wp", wp100_str)])
    hname_rej = myutils.name.encode([("v", "dnn"), ("d", "rejBkg"), ("wp", wp100_str)])
    # -- Signal efficiency
    h_sig_eff = ROOT.TH1F(hname_eff, hname_eff, len(self.pt_bins)-1, self.pt_bins)
    # -- Background rejection
    h_bkg_rej = ROOT.TH1F(hname_rej, hname_rej, len(self.pt_bins)-1, self.pt_bins)

    # Background rejection from ROC curves
    rej_bkg = rej_bkg_from_roc(graphs, working_point)

    # Fill histograms
    for i_bin, rej in enumerate(rej_bkg):
      h_bkg_rej.SetBinContent(i_bin + 1, rej)
      h_bkg_rej.SetBinError(i_bin + 1, 0)
    self.addObj(h_bkg_rej, "OutputQuick.root")

    # Plot
    _plt_rej_bkg(h_bkg_rej, wp100_str, os.path.join(self.get("home"), "plt/rej_bkg_wp_%s_quick" % wp100_str))

    return self



class Eval (myutils.walker.Walker, myroot.fio.FileIo):


  def __init__(self, flist, treename="FlatSubstructureJetTree", outdir="evalout"):

    self.treename = treename
    self.flist = flist
    # Initialize manager objects
    myutils.walker.Walker.__init__(self, outdir, write_mode="update")
    myroot.fio.FileIo.__init__(self, self.get("home"), write_mode="update")
    # Add files
    for fname in ["Input.root", "Output.root", "FitFunctions.root"]:
      self.addFile(fname, write_mode="update")


  def setPtBins(self, pt_binning):

    import array
    self.pt_bins = array.array("d", pt_binning)
    print("[\033[1mINFO\033[0m] The following pT binning will be used:")
    for i_bin in range(len(self.pt_bins)):
      pt_low, pt_high = myroot.cut.get_limits(i_bin, self.pt_bins, rtype=int, verbose=False)
      print("       |- \033[1m[%s,%s)\033[0m" % (pt_low, pt_high))
    return self


  def cutOnScore(self, working_point, pt="fjet_truthJet_pt/1000.", weight="fjet_testing_weight_pt", n_bins=200, orders=[4,5,6,7,8,9,10]):

    RDF = ROOT.RDataFrame(self.treename, conv2vec(self.flist))

    # Some checks on input
    if not isinstance(orders, list): orders = [orders]

    # Current working point as string
    wp_str, wp100_str = str(working_point), str(int(working_point*100.))

    """
      1. Prepare a dictionary that defines a histogram fro each pT bin for
         signal and background
    """

    results = {}
    for i_bin in range(len(self.pt_bins)):
      results[i_bin] = {"Sig":None, "Bkg":None}
      # Get pT range
      pt_low, pt_high = myroot.cut.get_limits(i_bin, self.pt_bins, rtype=int, verbose=True)
      # Model for histograms
      th1_m_sig = ROOT.ROOT.RDF.TH1DModel("sig.dnnScore.pt_%s-%s" % (pt_low, pt_high), "", n_bins, 0, 1)
      th1_m_bkg = ROOT.ROOT.RDF.TH1DModel("bkg.dnnScore.pt_%s-%s" % (pt_low, pt_high), "", n_bins, 0, 1)
      # Check if those histograms already exist in file
      t_h1_sig_tag = self.getObj(th1_m_sig.fName, "Input.root")
      t_h1_bkg_tag = self.getObj(th1_m_bkg.fName, "Input.root")
      # RDF with pt cut/filter applied
      rdf_pt = RDF.Filter("%s <= %s" % (pt_low, pt)).Filter("%s < %s" % (pt, pt_high))
      # If histograms have been found, fill dictionary
      if t_h1_sig_tag:
        results[i_bin]["Sig"] = t_h1_sig_tag
      else:
        results[i_bin]["Sig"] = rdf_pt.Filter("fjet_signal == 1").Histo1D(th1_m_sig, "fjet_dnnScore", weight)
      if t_h1_bkg_tag:
        results[i_bin]["Bkg"] = t_h1_bkg_tag
      else:
        results[i_bin]["Bkg"] = rdf_pt.Filter("fjet_signal == 0").Histo1D(th1_m_bkg, "fjet_dnnScore", weight)

    # Add all objects to input file to save time during the next run
    self.addObj([results[i_bin][key] for i_bin in results for key in results[i_bin]], "Input.root")

    """
      2. Only the shape is relevant; normalize all histograms to unity
    """

    # Normalize all historams to unity and plot results
    for i_bin in results:
      # Get pT range
      pt_low, pt_high = myroot.cut.get_limits(i_bin, self.pt_bins, rtype=int, verbose=False)
      for key in results[i_bin]:
        # No need for over and underflow bin
        integral = results[i_bin][key].Integral()
        if integral < 1E-10: continue
        results[i_bin][key].Scale(1.0/results[i_bin][key].Integral())
        self.addObj(results[i_bin][key], "Output.root")

      fname = os.path.join(self.get("home"), "plt/score/dnnScore.pt_%s-%s" % (pt_low, pt_high))
      _plt_score (results[i_bin]["Sig"], results[i_bin]["Bkg"], pt_low, pt_high, fname)

    """
      3. Determine the cut on the DNN score such that the signal efficiency corresponds to the
         working point of the tagger
    """

    # Histogram to hold cuts for each pT bin
    hname = myutils.name.encode([("v", "dnn"), ("d", "dnnScoreCutLow"), ("wp", wp100_str)])
    t_h1_cut = ROOT.TH1F(hname, hname, len(self.pt_bins)-1, self.pt_bins)

    # Get the cut on the dnn score in each pt bin
    from myroot.algo.optimization import get_cut_on_sig
    for i_bin in range(len(self.pt_bins)-1):
      # Get pT range
      pt_low, pt_high = myroot.cut.get_limits(i_bin, self.pt_bins, rtype=int, verbose=False)
      if results[i_bin]["Sig"].Integral() < 1E-10: continue
      cut_score, cut_err = get_cut_on_sig(results[i_bin]["Sig"], working_point, "gt")
      print("[EVAL] \033[1mlower cut=%s\033[0m - pT=[%s, %s] - wp=%s" % (cut_score, pt_low, pt_high, wp100_str))
      t_h1_cut.SetBinContent(i_bin+1, cut_score)
      t_h1_cut.SetBinError(i_bin+1, cut_err)
    self.addObj(t_h1_cut, "Output.root")

    # Save the figure ...
    fname = os.path.join(self.get("home"), "plt/cuts.wp_%s" % wp100_str)
    _plt_cut(t_h1_cut, wp100_str, fname)

    """
      4. Now that the the background rejection has been computed for each bin, apply
         a fit to get a smooth function 
    """

    from myroot.algo.fit import poly1D
    fits = []
    for order in orders:
      # Fit this histogram to get a smooth function of pT
      fits.append(poly1D(t_h1_cut, order))
      # Change name
      fits[-1].SetName(myutils.name.encode([("v", "fit"), ("d", "smoothCutDnnScore"), ("wp", wp100_str), ("order", str(order))]))
      fits[-1].SetTitle("%s,,l" % order)
      # Add to output file
    self.addObj(fits, "FitFunctions.root")

    # Plot list of fitting curves
    fname = os.path.join(self.get("home"), "plt/fit_family.wp_%s" % wp100_str)
    _plt_fits(fits, wp100_str, fname)

    return self


  def performance(self, working_point, pt="fjet_truthJet_pt/1000.", weight="fjet_testing_weight_pt", n_bins=200, order=7):

    RDF = ROOT.RDataFrame(self.treename, conv2vec(self.flist))

    # Current working point as string
    wp_str, wp100_str = str(working_point), str(int(working_point*100.))

    hname_eff = myutils.name.encode([("v", "dnn"), ("d", "effSig"), ("wp", wp100_str), ("order", str(order))])
    hname_rej = myutils.name.encode([("v", "dnn"), ("d", "rejBkg"), ("wp", wp100_str), ("order", str(order))])
    # -- Signal efficiency
    h_sig_eff = ROOT.TH1F(hname_eff, hname_eff, len(self.pt_bins)-1, self.pt_bins)
    # -- Background rejection
    h_bkg_rej = ROOT.TH1F(hname_rej, hname_rej, len(self.pt_bins)-1, self.pt_bins)

    # List of graphs (ROC curves)
    graphs = []

    # Get the smooth fit
    t_f1_name = myutils.name.encode([("v", "fit"), ("d", "smoothCutDnnScore"), ("wp", wp100_str), ("order", str(order))])
    t_fit = self.getObj(t_f1_name, "FitFunctions.root")

    from myroot.algo.optimization import eff
    from myroot.algo.performance import roc_curve_binned
    for i_bin in range(len(self.pt_bins)-1):

      pt_low, pt_high = myroot.cut.get_limits(i_bin, self.pt_bins, rtype=int, verbose=True)
      # RDF with pt cut/filter applied
      rdf_sel = RDF.Filter("%s <= %s" % (pt_low, pt)).Filter("%s < %s" % (pt, pt_high))
      # Split into signal and background and get
      rdf_sig = rdf_sel.Filter("fjet_signal == 1")
      rdf_bkg = rdf_sel.Filter("fjet_signal == 0")

      """
        All events in this bin
      """

      th1_m_sig_tot = ROOT.ROOT.RDF.TH1DModel("", "", n_bins, 0, 1)
      th1_m_bkg_tot = ROOT.ROOT.RDF.TH1DModel("", "", n_bins, 0, 1)
      th1_m_sig_tot.fName = myutils.name.encode([("v", "dnn"), ("d", "dnnScoreTotSig"), ("pt", pt_low, pt_high)])
      th1_m_bkg_tot.fName = myutils.name.encode([("v", "dnn"), ("d", "dnnScoreTotBkg"), ("pt", pt_low, pt_high)])
      # Check if the histogram already exist
      t_h1_sig_tot, t_h1_bkg_tot = self.getObj([th1_m_sig_tot.fName,  th1_m_bkg_tot.fName], "Input.root")
      if not t_h1_sig_tot:
        t_h1_sig_tot = rdf_sig.Histo1D(th1_m_sig_tot, "fjet_dnnScore", weight)
        self.addObj(t_h1_sig_tot, "Input.root")
      if not t_h1_bkg_tot:
        t_h1_bkg_tot = rdf_bkg.Histo1D(th1_m_bkg_tot, "fjet_dnnScore", weight)
        self.addObj(t_h1_bkg_tot, "Input.root")

      """
        Apply tagger cut(s)
      """

      # Get cuts
      cuts = myroot.cut.CutStr()
      # Add selection cut from fit
      cuts.addPolyomialCut(pt, "fjet_dnnScore", t_fit)
      # Get weights and cuts as list
      cuts = cuts.getCutList()
      # Apply cut to rdf
      for cut in cuts:
        rdf_sig_tag = rdf_sig.Filter(cut)
        rdf_bkg_tag = rdf_bkg.Filter(cut)

      # Get histogram
      th1_m_sig_tag = ROOT.ROOT.RDF.TH1DModel("", "", n_bins, 0, 1)
      th1_m_bkg_tag = ROOT.ROOT.RDF.TH1DModel("", "", n_bins, 0, 1)
      th1_m_sig_tag.fName = myutils.name.encode([("v", "dnn"), ("d", "dnnScoreTaggedSig"), ("pt", pt_low, pt_high), ("wp", wp100_str), ("order", str(order))])
      th1_m_bkg_tag.fName = myutils.name.encode([("v", "dnn"), ("d", "dnnScoreTaggedBkg"), ("pt", pt_low, pt_high), ("wp", wp100_str), ("order", str(order))])
      t_h1_sig_tag, t_h1_bkg_tag = self.getObj([th1_m_sig_tag.fName, th1_m_bkg_tag.fName], "Input.root")
      if not t_h1_sig_tag:
        t_h1_sig_tag = rdf_sig_tag.Histo1D(th1_m_sig_tag, "fjet_dnnScore", weight)
        self.addObj(t_h1_sig_tag, "Input.root")
      if not t_h1_bkg_tag:
        t_h1_bkg_tag = rdf_bkg_tag.Histo1D(th1_m_bkg_tag, "fjet_dnnScore", weight)
        self.addObj(t_h1_bkg_tag, "Input.root")

      """
        Get efficiencies
      """

      # Get efficiencies
      sig_eff, sig_err, bkg_eff, bkg_err = eff(t_h1_sig_tot, t_h1_bkg_tot, t_h1_sig_tag, t_h1_bkg_tag, working_point=wp_str, verbose=True)

      """
        Fill histograms
      """

      # Signal efficiency
      h_sig_eff.SetBinContent(i_bin + 1, sig_eff)
      h_sig_eff.SetBinError(i_bin + 1, sig_err)

      # Background rejection
      if float(bkg_eff) > 0.0:
        # Preventing undefined background rejection
        h_bkg_rej.SetBinContent(i_bin + 1, 1.0 / float(bkg_eff))
        h_bkg_rej.SetBinError(i_bin + 1, (1.0 / float(bkg_eff)) * (bkg_err / bkg_eff))
      else:
        h_bkg_rej.SetBinContent(i_bin + 1, 0)
        h_bkg_rej.SetBinError(i_bin + 1, 1E-9)

      """
        ROC curve in this bin
      """

      graphs.append(roc_curve_binned(t_h1_sig_tot, t_h1_bkg_tot, sig_eff_min=0))
      # Rename roc curve and save it to file
      graphs[-1].SetName(myutils.name.encode([("v", "dnn"), ("d", "rocCurve"), ("pt", pt_low, pt_high)]))
      graphs[-1].SetTitle("#scale[0.7]{[%s, %s]},,l" % (pt_low, pt_high))
      self.addObj(graphs, "Output.root")

      fname = os.path.join(self.get("home"), "plt/rocs/roc.pt_%s-%s" % (pt_low, pt_high))
      _plt_roc(graphs[-1], pt_low, pt_high, fname)

    """
      Plotting
    """

    myplt.settings.common_head_reset()
    myplt.settings.common_head() \
      .addRow(text="#sqrt{s}=13 TeV,  |#eta|<2.0,  DNN top tagger").setTextAlign("left")
    # - Background rejection
    Figure(canv=(600,500)) \
      .setDrawOption("PLC PMC HIST") \
      .addHist(h_bkg_rej) \
      .addAxisTitle(
        title_x="Transverse momentum p^{true}_{T} [GeV]",
        title_y="Background rejection 1/#varepsilon_{bkg} @ #varepsilon_{sig}=" + wp100_str + "%") \
      .save(os.path.join(self.get("home"), "plt/rej_bkg.wp_%s" % wp100_str)).close()

    # - Signal efficiency
    Figure(canv=(600,500)) \
      .setDrawOption("PLC PMC HIST") \
      .addHist(h_sig_eff) \
      .addAxisTitle(
        title_x="Transverse momentum p^{true}_{T} [GeV]",
        title_y="Signal efficiency @ #varepsilon_{sig}=" + wp100_str + "%") \
      .save(os.path.join(self.get("home"), "plt/eff_sig.wp_%s" % wp100_str)).close()

    # Plot list of fitting curves
    Figure(canv=(600,500)) \
      .setDrawOption("AL PLC PMC") \
      .setColorPalette(87) \
      .addHist(graphs) \
      .addAxisTitle(
        title_x="Signal efficiency #varepsilon_{sig}",
        title_y="Background rejection 1/#varepsilon_{bkg}") \
      .setLogY(range_y=(1,1E5)) \
      .setRangeX(0.05,1.1) \
      .addLegend(xmin=0.8, ymin=0.12, xmax=1.13, ymax=0.925) \
      .setPadMargin(margin=0.2, side="right") \
      .save(os.path.join(self.get("home"), "plt/roc_family")).close()

    return self
