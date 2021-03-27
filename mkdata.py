#!/usr/bin/env python2

# Standard imports
import os, sys
import configparser

# Scientific imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Custom imports
import src.common
import src.data


"""
 Some global configurations
"""

# Path to this very script
path2home = os.path.abspath(os.path.dirname(__file__))

# Every entry in this dictionary is a command line argument
__conf = \
{
  "input_sig"   : True,
  "input_bkg"   : True,
  "fout"        : True,
  "truth_label" : True,
  "n-events"    : True,
}


def _selection (branches, truth_label):

  import myroot.cut
  # Read the current selection from configuration file
  conf = configparser.ConfigParser(allow_no_value=True, encoding="ascii")
  conf.read([os.path.join(path2home, "etc/selection.ini")])

  # Get delimiter from config file(s)
  label, delim = conf[truth_label], conf["config"]["delim"]

  # Get selection for sig+bkg events (training and testing)
  selection = \
  {
    "common" : label["common"].split(delim),
    "sig"    : label["sig"].split(delim),
    "bkg"    : label["bkg"].split(delim),
    "train"  : " && ".join(conf["train"]["common"].split(delim))
  }

  # From this selection, construct selection string
  sel_train = selection.pop("train")
  sel_true = myroot.cut.get_selection(selection, branches, filter_other=None)
  return (sel_true, sel_train)


def _run ():

  # Command-line arguments pser
  args = src.common.get_parser(**__conf).parse_args()

  """
    To be displayed on plots
  """

  import myhep.jets, myplt.settings
  myplt.settings.common_head() \
    .addRow(text="#sqrt{s}=13 TeV, |#eta|<2.0, training-validation data") \
    .setTextAlign("left")
  # Add some standard text to all plots
  jetInfo = myhep.jets.JetInfo("AntiKt10UFOCSSKSoftDropBeta100Zcut10Jets")
  myplt.settings.common_text() \
    .addExperiment() \
    .addText(jetInfo.getReconstructionAndObject()) \
    .addText(jetInfo.getGroomer()[1])

  # Read the current selection from configuration file
  conf = configparser.ConfigParser(allow_no_value=True, encoding="ascii")
  conf.read([ os.path.join(path2home, "etc/trim_slim.ini")])
  branches = [branch.encode("ascii","ignore") for branch in conf.get("slim", "branches").split(conf["config"]["delim"])]

  """
    Get selection supposed to be applied to data set.
  """

  sel_true, sel_train = _selection(branches, args.truth_label)

  """
    Initialize a `DataBuilder` that generates a compact
    data set which can be used for machine-learining
    applications

    The first argument to the constructor is the name of
    the output file. If not specified otherwise, both, a
    ROOT and a HDF5, are produced with identical content.
  """

  # Initialize data builder object
  db = src.data.DataBuilder(args.fout)

  # Use multi threading (auto. num of threads) to speed up generation of data
  db.useMT()

  # Specify the path to the list files (for signal and background)
  # that contain the full path to the ROOT files
  db.setFilesSig(args.input_sig) \
    .setFilesBkg(args.input_bkg)

  # Specify the total number of events in the final file. This number
  # refers to the combined number of events of the `training` and the
  # `validation` data set. Furthermore, give the fraction of the total
  # number of events in the `training` set, i.e., N_train = frac * N_tot
  # and N_valid = (1-frac) * N_tot.
  db.setNEvents(args.n_events).setFracTrain(0.7)

  # Usually, the respective ROOT files contain a large number of branches
  # not needed for the analysis. Use `cols2keep` to specify the branches
  # that should be included in the final data set
  db.cols2keep(branches)

  # Define the selection that is supposed to be applied to the data.
  # There are two selections provided:
  #
  # 1. The truth labeling that defines the signal and background, e.g., top and dijet
  db.setSelSig(sel_true["sig"]).setSelBkg(sel_true["bkg"])
  # 2. Specific cuts for the training of the tagger
  #    For instance, one could apply a cut in the mass of the jet (m>50GeV)
  db.setSelTrain(sel_train)
  # Important: All events in the final data set (training + validation) fulfill the
  # FULL selection in point 2. The truth labeling infomation is stored as metadata along
  # with the full selection and is used later in the evaluation step.
  # If no `full` seletion is specified, the truth labeling (or what has been specified in
  # point 1) is used instad.

  # Start building the data set; this may take a while.
  db.build()

  # Finally, make a histogram for each branch in the final data set
  db.inspect(outdir="out/data")


if __name__ == "__main__":
  _run()
