#!/usr/bin/env python

# System imports
import os, sys
import argparse
proj_dir = os.path.abspath(os.path.dirname(__file__))
home_dir = os.path.abspath(os.path.join(proj_dir, ".."))
sys.path.append(home_dir)

# Scientific imports
import ROOT

# Custom imports
import src.common


proj_dir = os.path.abspath(os.path.dirname(__file__))
__fout__ = "check_pt_weights.root"
__data_tree__ = "FlatSubstructureJetTree"
__files_sig__ = os.path.join(home_dir, "dat/rel22p0/TopAntiKt10UFOCSSKSoftDropBeta100Zcut10Jets.list")
__files_bkg__ = os.path.join(home_dir, "dat/rel22p0/DijetAntiKt10UFOCSSKSoftDropBeta100Zcut10Jets.list")
__sig_tag__, __train_weight__ = src.common.read_conf(os.path.join(home_dir, "etc/nomen_est_omen.ini"), "variables", ["signalTag", "weightTrain"])


def _run (path2file):

  # Read the current selection from configuration file
  import configparser
  conf = configparser.ConfigParser(allow_no_value=True, encoding="ascii")
  conf.read([os.path.join(home_dir, "etc/selection.ini")])

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
  rdf_sig = get_RDF(__data_tree__, files_sig_new)
  rdf_bkg = get_RDF(__data_tree__, files_bkg_new)

  # Add a branch to identify whether the event is sig or bkg
  rdf_sig = rdf_sig.Define(__sig_tag__, "1") \
    .Filter(" && ".join(selection["sig"]+["fjet_truthJet_pt/1000.>350"]).encode("ascii", "ignore")) \
    .Define("fjet_truthJet_pt_gev", "fjet_truthJet_pt/1000.") \
    .Define("fjet_m_gev", "fjet_m/1000.")
  rdf_bkg = rdf_bkg.Define(__sig_tag__, "0") \
    .Filter("fjet_truthJet_pt/1000.>350") \
    .Define("fjet_truthJet_pt_gev", "fjet_truthJet_pt/1000.") \
    .Define("fjet_m_gev", "fjet_m/1000.") \

  print("NEvents Sig: %s" % rdf_sig.Count().GetValue())
  print("NEvents Bkg: %s" % rdf_bkg.Count().GetValue())



if __name__ == "__main__":


  # Get input file from command line
  parser = argparse.ArgumentParser(description="Check if input looks reasonable")
  parser.add_argument("--input", metavar="I", type=str, help="Input file [ROOT]")
  parser.add_argument("--plt", action="store_true", help="Plot histogram." )
  args = parser.parse_args()

  _run(args.input)


