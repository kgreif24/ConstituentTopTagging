#!/usr/bin/env python

# Standard imports
import os, glob, yaml

# Scientific imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Custom imports
import src.common
import src.eval


# Every entry in this dictionary is a command line argument
__conf = \
{
  "input"        : True,
  "outdir"       : True
}


#class RejBkg (myutils.walker.Walker, myroot.fio.FileIo):
#
#  def __init__(self, flist, treename="FlatSubstructureJetTree", outdir="evalout"):
#
#    self.treename = treename
#    self.flist = flist
#    # Initialize manager objects
#    myutils.walker.Walker.__init__(self, outdir, write_mode="update")
#    myroot.fio.FileIo.__init__(self, self.get("home"), write_mode="update")
#    # Add files
#    for fname in ["Input.root", "Output.root", "FitFunctions.root"]:
#      self.addFile(fname, write_mode="update")
#
#
#  def setPtBins(self, pt_binning):
#    import array
#    self.pt_bins = array.array("d", pt_binning)
#    return self
#
#
#  def cutOnScore(self, working_point, pt="fjet_truthJet_pt/1000.", weight="fjet_testing_weight_pt", n_bins=200, orders=[4,5,6,7,8,9,10]):
#
#    RDF = ROOT.RDataFrame(self.treename, conv2vec(self.flist))
#
#    # Some checks on input
#    if not isinstance(orders, list): orders = [orders]
#
#    # Current working point as string
#    wp_str, wp100_str = str(working_point), str(int(working_point*100.))
#
#    """
#      1. Prepare a dictionary that defines a histogram fro each pT bin for
#         signal and background
#    """
#
#    results = {}
#    for i_bin in range(len(self.pt_bins)):
#      results[i_bin] = {"Sig":None, "Bkg":None}
#      # Get pT range
#      pt_low, pt_high = myroot.cut.get_limits(i_bin, self.pt_bins, rtype=int, verbose=True)
#      # Model for histograms
#      th1_m_sig = ROOT.ROOT.RDF.TH1DModel("sig.dnnScore.pt_%s-%s" % (pt_low, pt_high), "", n_bins, 0, 1)
#      th1_m_bkg = ROOT.ROOT.RDF.TH1DModel("bkg.dnnScore.pt_%s-%s" % (pt_low, pt_high), "", n_bins, 0, 1)
#      # Check if those histograms already exist in file
#      t_h1_sig_tag = self.getObj(th1_m_sig.fName, "Input.root")
#      t_h1_bkg_tag = self.getObj(th1_m_bkg.fName, "Input.root")
#      # RDF with pt cut/filter applied
#      rdf_pt = RDF.Filter("%s <= %s" % (pt_low, pt)).Filter("%s < %s" % (pt, pt_high))
#      # If histograms have been found, fill dictionary
#      if t_h1_sig_tag:
#        results[i_bin]["Sig"] = t_h1_sig_tag
#      else:
#        results[i_bin]["Sig"] = rdf_pt.Filter("fjet_signal == 1").Histo1D(th1_m_sig, "fjet_dnnScore", weight)
#      if t_h1_bkg_tag:
#        results[i_bin]["Bkg"] = t_h1_bkg_tag
#      else:
#        results[i_bin]["Bkg"] = rdf_pt.Filter("fjet_signal == 0").Histo1D(th1_m_bkg, "fjet_dnnScore", weight)
#
#    # Add all objects to input file to save time during the next run
#    self.addObj([results[i_bin][key] for i_bin in results for key in results[i_bin]], "Input.root")
#
#    """
#      2. Only the shape is relevant; normalize all histograms to unity
#    """
#
#    # Normalize all historams to unity and plot results
#    for i_bin in results:
#      # Get pT range
#      pt_low, pt_high = myroot.cut.get_limits(i_bin, self.pt_bins, rtype=int, verbose=False)
#      for key in results[i_bin]:
#        # No need for over and underflow bin
#        integral = results[i_bin][key].Integral()
#        if integral < 1E-10: continue
#        results[i_bin][key].Scale(1.0/results[i_bin][key].Integral())
#        self.addObj(results[i_bin][key], "Output.root")
#
#      myplt.settings.common_head_reset()
#      myplt.settings.common_head() \
#        .addRow(text="#sqrt{s}=13 TeV,  |#eta|<2.0,  p_{T} #in [%s, %s] (GeV)" % (pt_low, pt_high)) \
#        .setTextAlign("left")
#      # Save the figure ...
#      Figure(canv=(600,500)) \
#        .setDrawOption("PLC PMC HIST") \
#        .addHist([results[i_bin]["Sig"], results[i_bin]["Bkg"]]) \
#        .addAxisTitle(title_x="DNN classification score") \
#        .addLegend(entries=["Signal", "Background"]) \
#        .setLogY(range_y=(1E-4, 1E0)) \
#        .save(os.path.join(self.get("home"), "plt/score/dnnScore.pt_%s-%s" % (pt_low, pt_high))).close()
#
#    """
#      3. Determine the cut on the DNN score such that the signal efficiency corresponds to the
#         working point of the tagger
#    """
#
#    # Histogram to hold cuts for each pT bin
#    hname = myutils.name.encode([("v", "dnn"), ("d", "dnnScoreCutLow"), ("wp", wp100_str)])
#    t_h1_cut = ROOT.TH1F(hname, hname, len(self.pt_bins)-1, self.pt_bins)
#
#    # Get the cut on the dnn score in each pt bin
#    from myroot.algo.optimization import get_cut_on_sig
#    for i_bin in range(len(self.pt_bins)-1):
#      # Get pT range
#      pt_low, pt_high = myroot.cut.get_limits(i_bin, self.pt_bins, rtype=int, verbose=False)
#      if results[i_bin]["Sig"].Integral() < 1E-10: continue
#      cut_score, cut_err = get_cut_on_sig(results[i_bin]["Sig"], working_point, "gt")
#      print("[EVAL] \033[1mlower cut=%s\033[0m - pT=[%s, %s] - wp=%s" % (cut_score, pt_low, pt_high, wp100_str))
#      t_h1_cut.SetBinContent(i_bin+1, cut_score)
#      t_h1_cut.SetBinError(i_bin+1, cut_err)
#    self.addObj(t_h1_cut, "Output.root")
#
#    # Save the figure ...
#    myplt.settings.common_head_reset()
#    myplt.settings.common_head() \
#      .addRow(text="#sqrt{s}=13 TeV,  |#eta|<2.0") \
#      .setTextAlign("left")
#    Figure(canv=(600,500)) \
#      .setDrawOption("PLC PMC HIST") \
#      .addHist(t_h1_cut) \
#      .addAxisTitle(
#        title_x="Transverse momentum p^{true}_{T} [GeV]",
#        title_y="Lower DNN-score cut @ #varepsilon_{sig}=" + wp100_str + "%") \
#      .save(os.path.join(self.get("home"), "plt/cuts.wp_%s" % wp100_str)).close()
#
#    """
#      4. Now that the the background rejection has been computed for each bin, apply
#         a fit to get a smooth function 
#    """
#
#    from myroot.algo.fit import poly1D
#    fits = []
#    for order in orders:
#      # Fit this histogram to get a smooth function of pT
#      fits.append(poly1D(t_h1_cut, order))
#      # Change name
#      fits[-1].SetName(myutils.name.encode([("v", "fit"), ("d", "smoothCutDnnScore"), ("wp", wp100_str), ("order", str(order))]))
#      fits[-1].SetTitle("%s,,l" % order)
#      # Add to output file
#    self.addObj(fits, "FitFunctions.root")
#
#    # Plot list of fitting curves
#    Figure(canv=(600,500)) \
#      .setDrawOption("PLC PMC") \
#      .addGraph(fits, range_y=(1.2*(1-float(wp_str)), 1.2)) \
#      .addAxisTitle(
#        title_x="Transverse momentum p^{true}_{T} [GeV]",
#        title_y="Lower DNN-score cut @ #varepsilon_{sig}=" + wp100_str + "%") \
#        .addLegend(title="#bf{Polynomial order}") \
#      .save(os.path.join(self.get("home"), "plt/fit_family.wp_%s" % wp100_str)).close()
#
#    return self
#
#
#  def performance(self, working_point, pt="fjet_truthJet_pt/1000.", weight="fjet_testing_weight_pt", n_bins=200, order=7):
#
#    RDF = ROOT.RDataFrame(self.treename, conv2vec(flist))
#
#    # Current working point as string
#    wp_str, wp100_str = str(working_point), str(int(working_point*100.))
#
#    hname_eff = myutils.name.encode([("v", "dnn"), ("d", "effSig"), ("wp", wp100_str), ("order", str(order))])
#    hname_rej = myutils.name.encode([("v", "dnn"), ("d", "rejBkg"), ("wp", wp100_str), ("order", str(order))])
#    # -- Signal efficiency
#    h_sig_eff = ROOT.TH1F(hname_eff, hname_eff, len(self.pt_bins)-1, self.pt_bins)
#    # -- Background rejection
#    h_bkg_rej = ROOT.TH1F(hname_rej, hname_rej, len(self.pt_bins)-1, self.pt_bins)
#
#    # List of graphs (ROC curves)
#    graphs = []
#
#    # Get the smooth fit
#    t_f1_name = myutils.name.encode([("v", "fit"), ("d", "smoothCutDnnScore"), ("wp", wp100_str), ("order", str(order))])
#    t_fit = self.getObj(t_f1_name, "FitFunctions.root")
#
#    from myroot.algo.optimization import eff
#    from myroot.algo.performance import roc_curve_binned
#    for i_bin in range(len(self.pt_bins)-1):
#      # Get pT range
#      pt_low, pt_high = myroot.cut.get_limits(i_bin, self.pt_bins, rtype=int, verbose=True)
#
#      # RDF with pt cut/filter applied
#      rdf_sel = RDF.Filter("%s <= %s" % (pt_low, pt)).Filter("%s < %s" % (pt, pt_high))
#      # Split into signal and background and get
#      rdf_sig = rdf_sel.Filter("fjet_signal == 1")
#      rdf_bkg = rdf_sel.Filter("fjet_signal == 0")
#
#      """
#        All events in this bin
#      """
#
#      th1_m_sig_tot = ROOT.ROOT.RDF.TH1DModel("", "", n_bins, 0, 1)
#      th1_m_bkg_tot = ROOT.ROOT.RDF.TH1DModel("", "", n_bins, 0, 1)
#      th1_m_sig_tot.fName = myutils.name.encode([("v", "dnn"), ("d", "dnnScoreTotSig"), ("pt", pt_low, pt_high)])
#      th1_m_bkg_tot.fName = myutils.name.encode([("v", "dnn"), ("d", "dnnScoreTotBkg"), ("pt", pt_low, pt_high)])
#      # Check if the histogram already exist
#      t_h1_sig_tot, t_h1_bkg_tot = self.getObj([th1_m_sig_tot.fName,  th1_m_bkg_tot.fName], "Input.root")
#      if not t_h1_sig_tot:
#        t_h1_sig_tot = rdf_sig.Histo1D(th1_m_sig_tot, "fjet_dnnScore", weight)
#        self.addObj(t_h1_sig_tot, "Input.root")
#      if not t_h1_bkg_tot:
#        t_h1_bkg_tot = rdf_bkg.Histo1D(th1_m_bkg_tot, "fjet_dnnScore", weight)
#        self.addObj(t_h1_bkg_tot, "Input.root")
#
#      """
#        Apply tagger cut(s)
#      """
#
#      # Get cuts
#      cuts = myroot.cut.CutStr()
#      # Add selection cut from fit
#      cuts.addPolyomialCut(pt, "fjet_dnnScore", t_fit)
#      # Get weights and cuts as list
#      cuts = cuts.getCutList()
#      # Apply cut to rdf
#      for cut in cuts:
#        rdf_sig_tag = rdf_sig.Filter(cut)
#        rdf_bkg_tag = rdf_bkg.Filter(cut)
#
#      # Get histogram
#      th1_m_sig_tag = ROOT.ROOT.RDF.TH1DModel("", "", n_bins, 0, 1)
#      th1_m_bkg_tag = ROOT.ROOT.RDF.TH1DModel("", "", n_bins, 0, 1)
#      th1_m_sig_tag.fName = myutils.name.encode([("v", "dnn"), ("d", "dnnScoreTaggedSig"), ("pt", pt_low, pt_high), ("wp", wp100_str), ("order", str(order))])
#      th1_m_bkg_tag.fName = myutils.name.encode([("v", "dnn"), ("d", "dnnScoreTaggedBkg"), ("pt", pt_low, pt_high), ("wp", wp100_str), ("order", str(order))])
#      t_h1_sig_tag, t_h1_bkg_tag = self.getObj([th1_m_sig_tag.fName, th1_m_bkg_tag.fName], "Input.root")
#      if not t_h1_sig_tag:
#        t_h1_sig_tag = rdf_sig_tag.Histo1D(th1_m_sig_tag, "fjet_dnnScore", weight)
#        self.addObj(t_h1_sig_tag, "Input.root")
#      if not t_h1_bkg_tag:
#        t_h1_bkg_tag = rdf_bkg_tag.Histo1D(th1_m_bkg_tag, "fjet_dnnScore", weight)
#        self.addObj(t_h1_bkg_tag, "Input.root")
#
#      """
#        Get efficiencies
#      """
#
#      # Get efficiencies
#      sig_eff, sig_err, bkg_eff, bkg_err = eff(t_h1_sig_tot, t_h1_bkg_tot, t_h1_sig_tag, t_h1_bkg_tag, working_point=wp_str, verbose=True)
#
#      """
#        Fill histograms
#      """
#
#      # Signal efficiency
#      h_sig_eff.SetBinContent(i_bin + 1, sig_eff)
#      h_sig_eff.SetBinError(i_bin + 1, sig_err)
#
#      # Background rejection
#      if float(bkg_eff) > 0.0:
#        # Preventing undefined background rejection
#        h_bkg_rej.SetBinContent(i_bin + 1, 1.0 / float(bkg_eff))
#        h_bkg_rej.SetBinError(i_bin + 1, (1.0 / float(bkg_eff)) * (bkg_err / bkg_eff))
#      else:
#        h_bkg_rej.SetBinContent(i_bin + 1, 0)
#        h_bkg_rej.SetBinError(i_bin + 1, 1E-9)
#
#      """
#        ROC curve in this bin
#      """
#
#      graphs.append(roc_curve_binned(t_h1_sig_tot, t_h1_bkg_tot, sig_eff_min=0))
#      # Rename roc curve and save it to file
#      graphs[-1].SetName(myutils.name.encode([("v", "dnn"), ("d", "rocCurve"), ("pt", pt_low, pt_high)]))
#      graphs[-1].SetTitle("#scale[0.7]{[%s, %s]},,l" % (pt_low, pt_high))
#      self.addObj(graphs, "Output.root")
#
#      myplt.settings.common_head_reset()
#      myplt.settings.common_head() \
#        .addRow(text="#sqrt{s}=13 TeV,  |#eta|<2.0,  p_{T} #in [%s, %s] (GeV)" % (pt_low, pt_high)) \
#        .setTextAlign("left")
#      # Plot ROC curve
#      Figure(canv=(600,500)) \
#        .setDrawOption("AL PLC PMC") \
#        .addHist(graphs[-1]) \
#        .addAxisTitle(
#          title_x="Signal efficiency #varepsilon_{sig}",
#          title_y="Background rejection 1/#varepsilon_{bkg}") \
#        .setLogY(range_y=(1,1E5)) \
#        .setRangeX(0.05,1.1) \
#        .save(os.path.join(self.get("home"), "plt/rocs/roc.pt_%s-%s" % (pt_low, pt_high))).close()
#
#    """
#      Plotting
#    """
#
#    myplt.settings.common_head_reset()
#    myplt.settings.common_head() \
#      .addRow(text="#sqrt{s}=13 TeV,  |#eta|<2.0").setTextAlign("left")
#    # - Background rejection
#    Figure(canv=(600,500)) \
#      .setDrawOption("PLC PMC HIST") \
#      .addHist(h_bkg_rej) \
#      .addAxisTitle(
#        title_x="Transverse momentum p^{true}_{T} [GeV]",
#        title_y="Background rejection 1/#varepsilon_{bkg} @ #varepsilon_{sig}=" + wp100_str + "%") \
#      .save(os.path.join(self.get("home"), "plt/rej_bkg.wp_%s" % wp100_str)).close()
#
#    # - Signal efficiency
#    Figure(canv=(600,500)) \
#      .setDrawOption("PLC PMC HIST") \
#      .addHist(h_sig_eff) \
#      .addAxisTitle(
#        title_x="Transverse momentum p^{true}_{T} [GeV]",
#        title_y="Signal efficiency @ #varepsilon_{sig}=" + wp100_str + "%") \
#      .save(os.path.join(self.get("home"), "plt/eff_sig.wp_%s" % wp100_str)).close()
#
#    # Plot list of fitting curves
#    Figure(canv=(600,500)) \
#      .setDrawOption("AL PLC PMC") \
#      .addHist(graphs) \
#      .addAxisTitle(
#        title_x="Signal efficiency #varepsilon_{sig}",
#        title_y="Background rejection 1/#varepsilon_{bkg}") \
#      .setLogY(range_y=(1,1E5)) \
#      .setRangeX(0.05,1.1) \
#      .addLegend(xmin=0.8, ymin=0.12, xmax=1.13, ymax=0.925) \
#      .setPadMargin(margin=0.2, side="right") \
#      .save(os.path.join(self.get("home"), "plt/roc_family")).close()
#
#    return self


if __name__ == "__main__":

  # Command-line arguments pser
  args = src.common.get_parser(**__conf).parse_args()

  """
    Load meta data
  """

  with open(os.path.join(args.outdir, "conf/metadata.json")) as fJSON:
    info = yaml.safe_load(fJSON)

  """
    To be displayed on plots
  """

  import myhep.jets, myplt.settings
  myplt.settings.common_head() \
    .addRow(text="#sqrt{s}=14 TeV, |#eta|<2.0") \
    .setTextAlign("left")
  # Add some standard text to all plots
  jetInfo = myhep.jets.JetInfo("AntiKt10UFOCSSKSoftDropBeta100Zcut10Jets")
  myplt.settings.common_text() \
    .addExperiment() \
    .addText(jetInfo.getReconstructionAndObject()) \
    .addText(jetInfo.getGroomer()[1]) \
    .addText("Constituent-based Top Tagger") \
    .addText("Num. of constit. %s" % info["NConstit"])

  """
    Start evaluation
  """

  # Go paralel
  ROOT.ROOT.EnableImplicitMT()

  # Get all ROOT files in the given directory
  if not os.path.isfile(args.input):
    flist = glob.glob("%s*.root" % args.input)
  else:
    flist = [args.input]

  # Initialize an evaluation object
  evaluation = src.eval.Eval(flist, outdir=args.outdir)

  # Set pT binning
  evaluation.setPtBins([350, 450, 550, 650, 750, 850, 950, 1050, 1200, 1350, 1500, 1650, 1800, 1950, 2100, 2300, 2500, 2700,2900, 3150, 3500])

  # Start optimization
  evaluation.cutOnScore(0.5).cutOnScore(0.8)

  # Evaluate performance
  evaluation.performance(0.5).performance(0.8)
