#!/usr/bin/env python2

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Standard imports
import os, sys
import uuid
import glob
import tqdm
import yaml
import collections
import configparser
import multiprocessing
import subprocess

## Scientific imports
import numpy as np
import root_numpy
import tensorflow as tf
import energyflow as ef

# Custom imports
import src.common
import myutils.orga
import myutils.profile


"""
  Some global variables and configurations
"""

# Logger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
tf.get_logger().setLevel("ERROR")

# Every entry in this dictionary is a command line argument
__conf__ = \
{
  "input"         : True,
  "fout"          : True,
  "max-processes" : True
}

# Some global variables
__path2dir__ = os.path.abspath(os.path.join(os.path.dirname(__file__)))
__dnn_score__, __sig_tag__ = src.common.read_conf(os.path.join(__path2dir__, "etc/nomen_est_omen.ini"), "variables", ["dnnScore", "signalTag"])
__event_pos__, __sort_col__ = src.common.read_conf(os.path.join(__path2dir__, "etc/nomen_est_omen.ini"), "variables", ["eventPos", "sortColumn"])
__ignore_list__ = ["Sig", "Bkg"]
__batch_size__ = 10000


inputs = ["fjet_sortClusNormByPt_pt", "fjet_sortClusCenterRotFlip_eta", "fjet_sortClusCenterRot_phi", "fjet_sortClusNormByPt_e"]


class TreeReader (ROOT.TFile):

  def __init__ (self, fname, treename, branches=None, cut=None):
    # Call constructor
    ROOT.TFile.__init__(self, fname, "READ")
    self.tree = self.Get(treename)


  def getEntries (self):
    return self.tree.GetEntries()


  def asNumpy (self, entry, cols):
    self.tree.GetEntry(entry)
    r_dict = collections.OrderedDict()
    for col in cols: r_dict[col] = np.asarray(getattr(self.tree, col))
    return r_dict



class FileSlimmer (multiprocessing.Process):

  comp = ["pt", "eta", "phi", "E"]

  def __init__ (self, vargs):

    # Base class constructor
    super(FileSlimmer, self).__init__()
    self._info = vargs


  def run (self):

    tRead = TreeReader(self._info["Path2File"], self._info["TreeName"])

    # Get number of entries in this Tree
    n_entries = tRead.getEntries()
    if n_entries == 0: return
    # Count number of batches
    n_batches = n_entries // __batch_size__

    # Get a slicing list
    steps = [(i*__batch_size__, (i+1)*__batch_size__) for i in range(n_batches)]
    # This list will be empty if n_batches == 0
    if not steps:
      steps += [(0, n_entries % __batch_size__)]
    elif (n_entries % __batch_size__) != 0:
      steps += [(steps[-1][1], steps[-1][1]+(n_entries % __batch_size__))]

    n_max = self._info["NConstit"]
    dtypes = [(col, "f8") for col in self._info["SlimList"] if col!=__sig_tag__] + [(__sig_tag__, "i4")]

    # Load the model
    from energyflow.archs import PFN
    with open(os.path.join(self._info["Path2Pro"], "conf/arch.json"), "r") as fJSON:
      model = tf.keras.models.model_from_json(fJSON.read())
      model.load_weights(os.path.join(self._info["Path2Pro"], "conf/weights.h5"))

    for i_batch, (start, stop) in enumerate(tqdm.tqdm(steps)):
      with myutils.profile.Profile("Start"):

        data = np.zeros(stop-start, dtype=dtypes)
        X = np.zeros((stop-start, n_max, 4))

        for j in range(stop-start):

          for col, x in tRead.asNumpy(start+j, self._info["SlimList"]).iteritems():
            data[col][j] = x

          # Get data to feed into tagger
          clus_pt, clus_eta, clus_phi, clus_e = tRead.asNumpy(start+j, inputs).values()

          n_const = len(clus_pt)
          if n_const > n_max: n_const = n_max
          X[j,:n_const,0] = clus_pt[:n_const]
          X[j,:n_const,1] = clus_eta[:n_const]
          X[j,:n_const,2] = clus_phi[:n_const]
          X[j,:n_const,3] = clus_e[:n_const]

        # Get predition by tagger
        data[__dnn_score__][:] = model.predict(X, verbose=1, use_multiprocessing=True)[:,1].flatten()
        root_numpy.array2root(data, self._info["Path2Fout"], self._info["TreeName"], mode="update")

    tRead.Close()
    subprocess.call("rm %s" % self._info["Path2File"], shell=True)


def _merge (fout, fin, clean=True):

  if not isinstance(fin, list): fin = [fin]
  subprocess.call("$ROOTSYS/bin/hadd -f -v %s %s" % (fout, " ".join(fin)), shell=True)
  if clean: subprocess.call("rm %s" % " ".join(fin), shell=True)


def _slim (info):

  # Get files
  from myroot.reader import read_file_list
  flist_sig = " ".join(read_file_list(info["FilesSig"]))
  flist_bkg = " ".join(read_file_list(info["FilesBkg"]))

  # Set filters
  filter_basic_sig = info["SelectionSig"].replace(" ", "")
  filter_basic_bkg = info["SelectionBkg"].replace(" ", "")
  filter_train     = info["SelectionTrain"].replace(" ", "")

  # Make sure tmp directory exists
  info["Path2Tmp"] = os.path.join("/tmp/csauer", info["Identify"])
  myutils.orga.mkdir(info["Path2Tmp"])

  # Argument string
  args =  "--branches %s --fin __fin__ --fout __fout__ -t %s --selTrue '__sel__' --selTrain '%s' --label __y__" % (" ".join(info["SlimList"]), info["TreeName"], filter_train)
  # - Signal
  print("[\033[1mINFO\033[0m] Start slimming for \033[1msignal\033[0m")
  fout_sig = os.path.join(info["Path2Tmp"], "sig.root")
  args_sig = args.replace("__fin__", flist_sig).replace("__fout__", fout_sig).replace("__sel__", filter_basic_sig).replace("__y__", "1")
  subprocess.call("lib/slim4pred/slim4pred %s" % args_sig, shell=True)
  print("[\033[1mINFO\033[0m] Start slimming for \033[1mnackground\033[0m")
  # - Background
  fout_bkg = os.path.join(info["Path2Tmp"], "bkg.root")
  args_bkg = args.replace("__fin__", flist_bkg).replace("__fout__", fout_bkg).replace("__sel__", filter_basic_bkg).replace("__y__", "0")
  subprocess.call("lib/slim4pred/slim4pred %s" % args_bkg, shell=True)

  # Merge two files that failed the tagger cuts
  failed = [fout_sig.replace(".root", ".fail.root"), fout_bkg.replace(".root", ".fail.root")]
  _merge(os.path.join(info["Path2Tmp"], "sig_bkg.fail.root"), " ".join(failed))


def _pred (info):

  fin_list = glob.glob("%s/*.pass.root" % info["Path2Tmp"])
  from multiprocessing import Manager
  inputs = []
  for path2file in fin_list:
    inputs.append({key: value for key, value in info.items() if all(ig not in key for ig in __ignore_list__)})
    inputs[-1]["Path2File"] = path2file
    inputs[-1]["Path2Fout"] = path2file.replace("pass", "eval")
    inputs[-1]["BatchNum"]  = len(inputs)
  from myutils.multi import run_batched
  with myutils.profile.Profile("Start evaluating all batches"):
    run_batched(FileSlimmer, inputs, max_processes=2)


def _run ():

  # Command-line arguments pser
  args = src.common.get_parser(**__conf__).parse_args()

  # Read the current selection from configuration file
  conf = configparser.ConfigParser(allow_no_value=True, encoding="ascii")
  conf.read([os.path.join(__path2dir__, "etc/slim4pred.ini")])

  # Load metadata
  with open(os.path.join(args.input, "conf/metadata.json")) as fJSON:
    info = yaml.safe_load(fJSON)

  # Add some entries to info dictionary
  info["Identify"] = uuid.uuid4().get_hex()
  info["SlimList"] = [name.encode("ascii") for name in conf.get("slim", "branches").split(conf.get("config", "delim"))]
  info["Path2Pro"] = os.path.abspath(args.input)
  info["TreeName"] = "FlatSubstructureJetTree"
  info["Features"] = ["fjet_sortNormByEnergyClus_pt", "fjet_sortShiftByMeanClus_eta", "fjet_sortShiftByMeanClus_phi", "fjet_sortNormByEnergyClus_e"]

  # Slim
  _slim(info)

  # Predict
  _pred(info)

  # Merge all files together
  _merge(args.fout, "%s/*.root" % info["Path2Tmp"])


if __name__ == "__main__":

  _run()

