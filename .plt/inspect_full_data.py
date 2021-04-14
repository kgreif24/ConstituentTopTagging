#!/usr/bin/env python

# System imports
import os, sys
proj_dir = os.path.abspath(os.path.dirname(__file__))
import argparse
import tqdm
sys.path.append("..")

# Scientific imports
import root_numpy
import numpy
import ROOT

# Custom imports
import src.common

# Import functions for RDF
from src.ginterpreter import *

# For plotting
from myplt.figure import Figure
import myplt.settings
import myroot.fio as fio

__fout__ = "full_ds_histos.root"
__data_tree__ = "FlatSubstructureJetTree"
__files_sig__ = "../dat/rel22p0/TopAntiKt10UFOCSSKSoftDropBeta100Zcut10Jets.list"
__files_bkg__ = "../dat/rel22p0/DijetAntiKt10UFOCSSKSoftDropBeta100Zcut10Jets.list"
__sig_tag__, __train_weight__ = src.common.read_conf(os.path.join(proj_dir, "../etc/nomen_est_omen.ini"), "variables", ["signalTag", "weightTrain"])


def _run (path2file):

  # Read the current selection from configuration file
  import configparser
  conf = configparser.ConfigParser(allow_no_value=True, encoding="ascii")
  conf.read([os.path.join(proj_dir, "../etc/selection.ini")])

  """
    Initialize RDFs and apply filters
  """

  # Get delimiter from config file(s)
  label = conf["rel22p0"]
  delim = conf["config"]["delim"]

  # Get selection for sig+bkg events (training and testing)
  selection = \
  {
    "common" : label["common"].split(delim),
    "sig"     : label["sig"].split(delim),
    "bkg"     : label["bkg"].split(delim),
    "train"   : " && ".join(conf["train"]["common"].split(delim)),
    "weight"  : label["weight"]
  }

  # Tell ROOT you want to go parallel
  ROOT.ROOT.EnableImplicitMT(50)

  # Sone conversion
  from myroot.reader import read_file_list
  files_sig_new = read_file_list(__files_sig__)
  files_bkg_new = read_file_list(__files_bkg__)

  # RDF for signal and background
  from myroot.reader import get_RDF
#  rdf_sig = get_RDF(__data_tree__, files_sig_new).Range(10000)
#  rdf_bkg = get_RDF(__data_tree__, files_bkg_new).Range(10000)
  rdf_sig = get_RDF(__data_tree__, files_sig_new)
  rdf_bkg = get_RDF(__data_tree__, files_bkg_new)

  # Add a branch to identify whether the event is sig or bkg
  rdf_sig = rdf_sig.Define(__sig_tag__, "1") \
    .Filter(" && ".join(selection["sig"]+["fjet_truthJet_pt/1000.>350"]).encode("ascii", "ignore")) \
    .Define("fjet_nufo", "fjet_clus_pt.size()").Define("fjet_ntrack", "fjet_tracks_pt.size()")
  rdf_bkg = rdf_bkg.Define(__sig_tag__, "0") \
    .Filter("fjet_truthJet_pt/1000.>350") \
    .Define("fjet_nufo", "fjet_clus_pt.size()").Define("fjet_ntrack", "fjet_tracks_pt.size()")

  cols  = [("fjet_sortClus_%s" % col.lower(), "pp::sort(fjet_clus_%s,%s,%s)" % (col, "fjet_clus_pt", "200")) for col in ["pt", "eta", "phi", "E"]]
  cols += [("fjet_sortClus_pt_0", "fjet_sortClus_pt[0]/1000."), ("fjet_sortClus_pt_1", "fjet_sortClus_pt[1]/1000.")]
  jet_str = "reco::jet(fjet_sortClus_pt, fjet_sortClus_eta, fjet_sortClus_phi, fjet_sortClus_e)"
  cols += [("fjet_reco_%s" % col, jet_str+".%s()" % col.title()) for col in ["eta", "phi"]]
  cols += [("fjet_reco_%s" % col, jet_str+".%s()/1000." % col.title()) for col in ["pt", "m"]]
  cols += [("fjet_ufo_m", "reco::constit_m(fjet_clus_pt, fjet_clus_eta, fjet_clus_phi, fjet_clus_E)")]

  # Add some columns
  for col in [("fjet_m_gev", "fjet_m/1000.")] + cols:
    rdf_sig = rdf_sig.Define(col[0], col[1])
    rdf_bkg = rdf_bkg.Define(col[0], col[1])

  """
    Book histograms and read data
  """

  # Histograms
  th = {}

  import myroot.utils
  # Number of UFOs
  t_h1_model_sig = myroot.utils.get_tH1_model("n_ufos_sig", "Signal,,l", 201, 0, 200)
  t_h1_model_bkg = myroot.utils.get_tH1_model("n_ufos_bkg", "Background,,l", 201, 0, 200)
  th["n_ufos"] = [rdf_sig.Histo1D(t_h1_model_sig, "fjet_nufo"), rdf_bkg.Histo1D(t_h1_model_bkg, "fjet_nufo")]

  # Number of tracks
  t_h1_model_sig = myroot.utils.get_tH1_model("n_tracks_sig", "Signal,,l", 201, 0, 200)
  t_h1_model_bkg = myroot.utils.get_tH1_model("n_tracks_bkg", "Background,,l", 201, 0, 200)
  th["n_tracks"] = [rdf_sig.Histo1D(t_h1_model_sig, "fjet_ntrack"), rdf_bkg.Histo1D(t_h1_model_bkg, "fjet_ntrack")]

  # Mass
  t_h1_model_sig = myroot.utils.get_tH1_model("mass_sig", "Signal,,l", 300, 0, 500)
  t_h1_model_bkg = myroot.utils.get_tH1_model("mass_bkg", "Background,,l", 300, 0, 500)
  th["mass"] = [rdf_sig.Histo1D(t_h1_model_sig, "fjet_m_gev" , "fjet_testing_weight_pt"), rdf_bkg.Histo1D(t_h1_model_bkg, "fjet_m_gev", "fjet_testing_weight_pt")]

  # Mass reconstructed from UFOs
  t_h1_model_sig = myroot.utils.get_tH1_model("reco_mass_sig", "Signal,,l", 300, 0, 500)
  t_h1_model_bkg = myroot.utils.get_tH1_model("reco_mass_bkg", "Background,,l", 300, 0, 500)
  th["reco_mass"] = [rdf_sig.Histo1D(t_h1_model_sig, "fjet_reco_m" , "fjet_testing_weight_pt"), rdf_bkg.Histo1D(t_h1_model_bkg, "fjet_m_gev", "fjet_testing_weight_pt")]

  # Mass of UFOs
  # - Range 0 - 100
  t_h1_model_sig = myroot.utils.get_tH1_model("ufo_m_0_100_sig", "Signal,,l", 300, 0, 100)
  t_h1_model_bkg = myroot.utils.get_tH1_model("ufo_m_0_100_bkg", "Background,,l", 300, 0, 100)
  th["ufo_m_0_100"] = [rdf_sig.Histo1D(t_h1_model_sig, "fjet_ufo_m" , "fjet_testing_weight_pt"), rdf_bkg.Histo1D(t_h1_model_bkg, "fjet_ufo_m", "fjet_testing_weight_pt")]
  # - Range 0 - 1
  t_h1_model_sig = myroot.utils.get_tH1_model("ufo_m_0_1_sig", "Signal,,l", 300, 0, 0.2)
  t_h1_model_bkg = myroot.utils.get_tH1_model("ufo_m_0_1_bkg", "Background,,l", 300, 0, 0.2)
  th["ufo_m_0_1"] = [rdf_sig.Histo1D(t_h1_model_sig, "fjet_ufo_m" , "fjet_testing_weight_pt"), rdf_bkg.Histo1D(t_h1_model_bkg, "fjet_ufo_m", "fjet_testing_weight_pt")]
  # - Range 0 - 1
  t_h1_model_sig = myroot.utils.get_tH1_model("ufo_m_pi_sig", "Signal,,l", 300, 0.139, 0.140)
  t_h1_model_bkg = myroot.utils.get_tH1_model("ufo_m_pi_bkg", "Background,,l", 300, 0.139, 0.140)
  th["ufo_m_pi"] = [rdf_sig.Histo1D(t_h1_model_sig, "fjet_ufo_m" , "fjet_testing_weight_pt"), rdf_bkg.Histo1D(t_h1_model_bkg, "fjet_ufo_m", "fjet_testing_weight_pt")]

  # Leading constituent pT versus sub-leading constituent pT
  t_h2_model_sig = myroot.utils.get_tH2_model("lufo_vs_slufo_pt_sig", "Signal,,l", 200, 0, 300, 200, 0, 300)
  t_h2_model_bkg = myroot.utils.get_tH2_model("lufo_vs_slufo_pt_bkg", "Background,,l", 200, 0, 300, 200, 0, 300)
  th["lufo_vs_slufo_pt"] = [rdf_sig.Histo2D(t_h2_model_sig, "fjet_sortClus_pt_0" , "fjet_sortClus_pt_1", "fjet_testing_weight_pt"),
    rdf_bkg.Histo2D(t_h2_model_bkg, "fjet_sortClus_pt_0" , "fjet_sortClus_pt_1", "fjet_testing_weight_pt")]

  """
    Finalize
  """

  # Save histogram in file
  file_io = fio.FileIo("dat", "update")
  file_io.addFile(__fout__, write_mode="update")
  for key in th:
    file_io.addObj([th[key][0], th[key][1]], __fout__)


def _plt ():

  """
    Read histograms
  """

  file_io = fio.FileIo("dat", "update")
  file_io.addFile(__fout__, write_mode="update")

  # Read histograms
  th1 = {"n_ufos":None, "n_tracks":None, "mass":None, "ufo_m_0_100":None, "ufo_m_0_1":None, "ufo_m_pi":None, "reco_mass":None}
  for key in th1:
    th1[key] = [file_io.getObj("%s_sig" % key, __fout__), file_io.getObj("%s_bkg" % key, __fout__)]
    th1[key][0].Scale(1.0 / th1[key][0].Integral())
    th1[key][1].Scale(1.0 / th1[key][1].Integral())

  th2 = {"lufo_vs_slufo_pt":None}
  for key in th2:
    th2[key] = [file_io.getObj("%s_sig" % key, __fout__), file_io.getObj("%s_bkg" % key, __fout__)]


  """
    Plotting
  """

  # Some global plotting settings
  myplt.settings.common_text() \
    .addExperiment()  \
    .addText("Anti-k_{t} R=1.0 jets") \
    .addText("UFO SD(#beta=1.0, z_{cut}=0.1), CS+SK")
  myplt.settings.legend.x1 = 0.6
  myplt.settings.legend.x2 = 0.9
  myplt.settings.common_head().reset()
  myplt.settings.common_head() \
    .addRow(text="#sqrt{s}=13 TeV, |#eta|<2.0, p_{T} > 350 GeV, Contained Top (rel. 22)") \
    .setTextAlign("left")

  # Number of UFOs
  Figure(canv=(600,500)) \
    .setDrawOption("PMC PLC HIST") \
    .addHist(th1["n_ufos"]) \
    .addAxisTitle(title_x="Number of UFOs (constituents) in jet") \
    .addLegend() \
    .save("out/full_ds/n_ufos").close()

  # Number of tracks
  Figure(canv=(600,500)) \
    .setDrawOption("PMC PLC HIST") \
    .addHist(th1["n_tracks"]) \
    .addAxisTitle(title_x="Number of tracks in jet") \
    .addLegend() \
    .save("out/full_ds/n_tracks").close()

  # Mass
  Figure(canv=(600,500)) \
    .setDrawOption("PMC PLC HIST") \
    .addHist(th1["mass"]) \
    .addAxisTitle(title_x="Reconstructed jet mass m_{jet} [GeV]") \
    .addLegend() \
    .save("out/full_ds/mass").close()

  # Jet mass from UFOs
  Figure(canv=(600,500)) \
    .setDrawOption("PMC PLC HIST") \
    .addHist(th1["reco_mass"]) \
    .addAxisTitle(title_x="Jet mass reconstructed from UFOs [GeV]") \
    .addLegend() \
    .save("out/full_ds/mass_from_ufos").close()

  # Mass UFO
  # - Range 0 - 100 GeV
  Figure(canv=(600,500)) \
    .setDrawOption("PMC PLC HIST") \
    .addHist(th1["ufo_m_0_100"]) \
    .addAxisTitle(title_x="Reconstructed UFO mass m_{UFO} [GeV]") \
    .addLegend() \
    .setLogY(range_y=(1E-12, 10)) \
    .save("out/full_ds/ufo_m_0_100").close()
  # - Range 0 - 1 GeV
  Figure(canv=(600,500)) \
    .setDrawOption("PMC PLC HIST") \
    .addHist(th1["ufo_m_0_1"]) \
    .addAxisTitle(title_x="Reconstructed UFO mass m_{UFO} [GeV]") \
    .addLegend() \
    .setLogY(range_y=(1E-7, 10)) \
    .save("out/full_ds/ufo_m_0_1").close()
  # - Range pion
  Figure(canv=(600,500)) \
    .setDrawOption("PMC PLC HIST") \
    .addHist(th1["ufo_m_pi"]) \
    .addAxisTitle(title_x="Reconstructed UFO mass m_{UFO} [MeV]") \
    .addLegend() \
    .setLogY(range_y=(1E-7, 10)) \
    .setRangeX(139, 140) \
    .save("out/full_ds/ufo_m_pi").close()

  # Correlation between leading and sub-leading constituent
  myplt.settings.auto.adjust_y = False
  # - Signal
  myplt.settings.common_head_reset()
  myplt.settings.common_head() \
    .addRow(text="#sqrt{s}=13 TeV, |#eta|<2.0, p_{T} > 350 GeV, #bf{Signal (top)}") \
    .setTextAlign("left")
  Figure(canv=(600,500)) \
    .setDrawOption("COLZ") \
    .setColorPalette(51) \
    .addHist(th2["lufo_vs_slufo_pt"][0]) \
    .addAxisTitle(
      title_x="Leading constituent pT [GeV]",
      title_y="Sub-leading constituent pT [GeV]",
      title_z="Number of events") \
    .setPadMargin() \
    .setLogZ() \
    .save("out/full_ds/lufo_vs_slufo_pt_sig").close()

  # - Background
  myplt.settings.common_head_reset()
  myplt.settings.common_head() \
    .addRow(text="#sqrt{s}=13 TeV, |#eta|<2.0, p_{T} > 350 GeV, #bf{Background (QCD)}") \
    .setTextAlign("left")
  Figure(canv=(600,500)) \
    .setDrawOption("COLZ") \
    .setColorPalette(51) \
    .addHist(th2["lufo_vs_slufo_pt"][1]) \
    .addAxisTitle(
      title_x="Leading constituent pT [GeV]",
      title_y="Sub-leading constituent pT [GeV]",
      title_z="Number of events") \
    .setPadMargin() \
    .setLogZ() \
    .save("out/full_ds/lufo_vs_slufo_pt_bkg").close()



if __name__ == "__main__":


  # Get input file from command line
  parser = argparse.ArgumentParser(description="Check if input looks reasonable")
  parser.add_argument("--input", metavar="I", type=str, help="Input file [ROOT]")
  parser.add_argument("--plt", action="store_true", help="Plot histogram." )
  args = parser.parse_args()

  # Compute data
  if not args.plt:
    _run(args.input)
  else:
    _plt()


