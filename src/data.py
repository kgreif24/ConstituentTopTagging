# Standard imports
import os, sys
import uuid
import random
import itertools
import subprocess

# Scientific imports
import numpy as np
import uproot
import ROOT
import root_numpy
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Deleted energyflow imports, we won't need them

# Custom imports
import common
import utils
import src.myroot.reader
import src.myroot.convert
import src.myutils
import src.myutils.walker
import src.myutils.profile


# Path variables
path2file = os.path.abspath(os.path.dirname(__file__))
path2home = os.path.join(path2file, "..")

__sec_train__, __sec_valid__ = common.read_conf(os.path.join(path2home, "etc/nomen_est_omen.ini"), "modes", ["training", "validation"])
__sig_tag__, __train_weight__, __match_weight__ = common.read_conf(os.path.join(path2home, "etc/nomen_est_omen.ini"), "variables", ["signalTag", "weightTrain", "weightMatch"])


def _n_events (n, n_sig, n_bkg):
  n_keep_sig, n_keep_bkg = 1, 1
  if n != -1 and n < (n_sig + n_bkg):
    n_keep_sig, n_keep_bkg = int(2 *n_sig // n), int(2 *n_bkg // n)
    if n_keep_sig == 0: n_keep_sig = 1
    if n_keep_bkg == 0: n_keep_bkg = 1
  return (n_keep_sig, n_keep_bkg)


def _ascii (string):
  return string.encode("ascii")



class DataPipe (object):


  def __init__ (self, path2data, features=[], target=None, weight=None, other=[], outdir="./"):

    self.meta = {"PathTrainDat":path2data, "Features":features, "Target":target, "Weight":weight, "Other":other}
    self.data = {__sec_train__:None, __sec_valid__:None}
    self.path2out = outdir
    self.features = []


  def gen(self, key, batch_size=None, n_points=-1, categorical=True, shuffle=True):

    class Generator (object):

      def __init__(self, key, batch_size, n_points, meta):

        # Load the tree
        self.meta = meta
        self.key = key
        # List of all columns to read from ROOT file
        self.cols = list(utils.flatten(self.meta["Features"]))
        self.cols += [self.meta["Target"]] + self.meta["Other"] + ([self.meta["Weight"]] if self.meta["Weight"] is not None else [])
        # Load tree
        self.tree = uproot3.open(self.meta["PathTrainDat"])[key]
        # Set number of points
        self.n_points = n_points
        if (self.tree.numentries < self.n_points) or (self.n_points == -1):
          print("[INFO] More points than available have been requested. Using %s points" % self.tree.numentries)
          self.n_points = self.tree.numentries
        # Read the entire data set if no batch size has been specified
        self.batch_size = batch_size
        if self.batch_size is None: self.batch_size = self.n_points
        # Compute number of batches
        self.n_batches = (self.n_points if self.tree.numentries > self.n_points else self.tree.numentries) // self.batch_size
        # Shape of the data set
        self.shape = (self.batch_size, self.meta["NConstit"])
        # Cache
        self.cache = uproot3.cache.ArrayCache(np.float64(0).nbytes*100000*1024)

      def __call__(self, categorical=True, shuffle=True, randomize=False):

        def _resize (arr, n_max=self.meta["NConstit"]):
          ndiff = arr.size - n_max
          return np.resize(np.pad(arr, (0, np.abs(ndiff)), "constant"), n_max)

        i_batch = 0
        batches = None
        while True:

          if (i_batch >= (self.n_batches-1)) or batches is None:
            i_batch = 0
            idx = random.randint(0, self.tree.numentries - self.n_points)
            batches = np.array_split(np.arange(idx, self.n_points+idx), self.n_batches)
            np.random.shuffle(batches)
          if not randomize:
            data = self.tree.lazyarrays(self.cols, entrysteps=self.batch_size,
              entrystart=i_batch*self.batch_size, entrystop=(i_batch+1)*self.batch_size, cache=self.cache)
          else:
            data = self.tree.lazyarrays(self.cols, entrysteps=len(batches[i_batch]),
              entrystart=batches[i_batch][0], entrystop=batches[i_batch][-1], cache=self.cache)
          if data.size == 0:
            i_batch = 0
            continue

          # Feature array
          X = []
          if len(self.meta["Features"]) == 4:
            for feature in self.meta["Features"]:
              X.append([])
              for arr in data[feature]:
                X[-1].append(_resize(arr))
            X = np.dstack(tuple(X))
            # Due to preprocessing, there might be an invalid data point from time to time. Set NaNs and Infs to zero
            X = np.where(np.isnan(X), 0, X)
            if self.meta["Flattening"]:
              X = X.reshape(self.batch_size, -1)
          elif len(self.meta["Features"]) == 2:
            XX = []
            for arr in data[self.meta["Features"][0]]:
              XX.append(_resize(arr))
            X.append(np.array(XX))
            XX = []
            for feature in self.meta["Features"][1]:
              XX.append([])
              for arr in data[feature]:
                XX[-1].append(_resize(arr))
            X.append(np.dstack(tuple(XX)))
            X[0] = np.where(np.isnan(X[0]), 0, X[0])
            X[1] = np.where(np.isnan(X[1]), 0, X[1])

          # Get targets
          Y = np.array(data[self.meta["Target"]])
          if categorical:
            # Y = to_categorical(Y, num_classes=2)
            print("Feature removed so I could run")
            print("Best of luck if you ever see this lol")

          # Get weights
          if self.meta["Weight"] is not None:
            W = np.array(data[self.meta["Weight"]])
          else:
            W = np.ones(Y.shape[0])

          # Get weights
          O = []
          if len(self.meta["Other"]) != 0:
            for o in self.meta["Other"]:
              O.append(np.array(data[o]))

          # Shuffle
          pos = np.arange(Y.shape[0])
          if shuffle:
            np.random.shuffle(pos)
          i_batch += 1

          if self.key == "train":
            if len(self.meta["Features"]) == 4:
              yield (X[pos], Y[pos], W[pos])
            elif len(self.meta["Features"]) == 2:
              yield ([X[0][pos], X[1][pos]], Y[pos], W[pos])
          else:
            if len(self.meta["Features"]) == 4:
              out = [X[pos], Y[pos], W[pos]] + O
            elif len(self.meta["Features"]) == 2:
              out = [[X[0][pos], X[1][pos]], Y[pos], W[pos]] + O
            yield tuple(out)

    return Generator(key, batch_size, n_points, self.meta)


  def saveMetadata (self, fname):

    with open(fname, "w") as fJSON:
      json.dump(self.meta, fJSON)


  def shape (self, flat=False):

    if not flat:
      return (len(self.meta["Features"]),)
    else:
      def flatten(container):
        for i in container:
          if isinstance(i, (list,tuple)):
            for j in flatten(i): yield j
          else: yield i
      return len(list(flatten(self.meta["Features"])))


  def setNConstit (self, n_constit):

    self.meta["NConstit"] = n_constit
    print("[\033[1mINFO\033[0m] Number of constituents has been set to \033[1m`%s`\033[0m" % self.meta["NConstit"])


  def addFeature(self, feature, flat=False):

    if not isinstance(feature, list): feature = [feature]
    for feat in feature:
      self.meta["Features"].append(feat)
    print("[\033[1mINFO\033[0m] Added \033[1m`%s`\033[0m (%s) to list of features" % (self.meta["Features"], len(self.meta["Features"])))
    self.meta["Flattening"] = flat
    return self


  def setTarget(self, target):

    self.meta["Target"] = target
    print("[\033[1mINFO\033[0m] Target has been set to \033[1m`%s`\033[0m" % self.meta["Target"])
    return self


  def setWeight(self, weight):

    self.meta["Weight"] = weight
    print("[\033[1mINFO\033[0m] Weight has been set to \033[1m`%s`\033[0m" % self.meta["Weight"])
    return self


  def addOther(self, name):

    self.meta["Other"].append(name)


  def display(self):

    print("[\033[1m\033[1mINFO\033[0m\033[0m] [\033[1mPipeline\033[0m] Using the following variables:")
    print("[    ] [\33[34mFeatures\033[0m]")
    for v in self.meta["Features"]: print("[    ] \033[1m%s\033[0m" % v)
    print("[    ] [\33[34mTarget\033[0m]")
    print("[    ] \033[1m%s\033[0m" % self.meta["Target"])
    print("[    ] [\33[34mWeight\033[0m]")
    print("[    ] \033[1m%s\033[0m" % self.meta["Weight"])
    if len(self.meta["Other"]) != 0:
      print("[    ] [\33[34mOther\033[0m]")
      for v in self.meta["Other"]: print("[    ] \033[1m%s\033[0m" % v)
    return self



class DataBuilder(src.myutils.walker.Walker):


  def __init__ (self, fout, files_sig=None, files_bkg=None, treename="FlatSubstructureJetTree"):

    self.fout = fout
    self.path2out = os.path.dirname(self.fout)
    src.myutils.walker.Walker.__init__(self, self.path2out, write_mode="update")
    # Some other members
    self.n_tot = int(1E6)
    self.n_threads = 0
    self.branches = ROOT.vector("string")()
    # Generate a unique id for this run
    self.stamp = uuid.uuid4().hex

    # This metadata will be stored
    self.info = \
    {
      "FilesSig"         : files_sig,
      "FilesBkg"         : files_bkg,
      "RandSeed"         : random.randint(1, 1E5),
      "TreeName"         : treename,
      "PathTrainDat"     : None,
      "SelectionSig"     : None,
      "SelectionBkg"     : None,
      "SelectionTrain"   : None
    }


  def _prep (self):

    # TODO: Dirty hack; fix this Christof :).
    if self.info["SelectionTrain"] is None:
      self.info["SelectionTrain"] = "(1==1)"
    # Add some columns
    self.branches.push_back(__sig_tag__)


  def _slim (self):

    # Get files
    flist_sig = src.myroot.reader.read_file_list(self.info["FilesSig"])
    flist_bkg = src.myroot.reader.read_file_list(self.info["FilesBkg"])

    # Shuffle files
    random.Random(self.info["RandSeed"]).shuffle(flist_sig)
    random.Random(self.info["RandSeed"]).shuffle(flist_bkg)

    # RDF for signal and background
    rdf_sig = ROOT.RDataFrame(self.info["TreeName"], src.myroot.convert.conv2vec(flist_sig))
    rdf_bkg = ROOT.RDataFrame(self.info["TreeName"], src.myroot.convert.conv2vec(flist_bkg))

    # Set filters
    filter_basic_sig = self.info["SelectionSig"]
    filter_basic_bkg = self.info["SelectionBkg"]
    filter_train = self.info["SelectionTrain"]

    # Apply basic selection (truth label, training cut etc.)
    # - Signal
    rdf_sig = rdf_sig.Define(__sig_tag__, "1") \
                     .Filter(filter_basic_sig, filter_basic_sig) \
                     .Filter(filter_train, filter_train)

    # - Background
    rdf_bkg = rdf_bkg.Define(__sig_tag__, "0") \
                     .Filter(filter_basic_bkg, filter_basic_bkg) \
                     .Filter(filter_train, filter_train)

    # Check if selection cuts are indeed correct
    print("[\033[1mINFO\033[0m] Filters for \033[1msignal\033[0m")
    for filter_name in rdf_sig.GetFilterNames():
      print("       |- \33[1m%s\33[0m" % filter_name)
    print("[\033[1mINFO\033[0m] Filters for \033[1mbackground\033[0m")
    for filter_name in rdf_bkg.GetFilterNames():
      print("       |- \33[1m%s\33[0m" % filter_name)

    # We want an equal number of signal and background events
    n_sig, n_bkg = rdf_sig.Count().GetValue(), rdf_bkg.Count().GetValue()
    print("[\033[1mINFO\033[0m] Total number of events:")
    print("       |- Sig: %s" % n_sig)
    print("       |- Bkg: %s" % n_bkg)

    # Deterine number of rows to keep
    n_keep_sig, n_keep_bkg = _n_events (self.n_tot, n_sig, n_bkg)
    print("[\033[1mINFO\033[0m] Number of events to keep:")
    print("       |- Sig: %s - Events in file: %s" % (n_keep_sig, n_sig // n_keep_sig))
    print("       |- Bkg: %s - Events in file: %s" % (n_keep_bkg, n_bkg // n_keep_bkg))

    # Filter events and save snapshot (tmp files will be deleted)
    files_slim = [self.get("home", "slim.sig.%s.root" % self.stamp),
      self.get("home", "slim.bkg.%s.root" % self.stamp)]

    # Save temporary snapshots of data
    rdf_sig.Filter("rdfentry_ % {} == 0".format(n_keep_sig)).Snapshot(self.info["TreeName"], files_slim[0], self.branches)
    rdf_bkg.Filter("rdfentry_ % {} == 0".format(n_keep_bkg)).Snapshot(self.info["TreeName"], files_slim[1], self.branches)

    # Split both files (sig + bkg) in many smaller files and shuffle those
    subprocess.call("%s/lib/shuffle/shuffle --path2tmp %s --fraction 0.01 -t %s --fin %s --fout '' " % (path2home, self.get("sig"), self.info["TreeName"], files_slim[0]), shell=True)
    subprocess.call("%s/lib/shuffle/shuffle --path2tmp %s --fraction 0.01 -t %s --fin %s --fout '' " % (path2home, self.get("bkg"), self.info["TreeName"], files_slim[1]), shell=True)
    subprocess.call("rm %s" % " ".join(files_slim), shell=True)
   # Get a list of all files
    import glob
    files_slim = glob.glob("%s/*" % self.get("sig")) + glob.glob("%s/*" % self.get("bkg"))
    random.Random(self.info["RandSeed"]).shuffle(files_slim)

    # Merge the two files and remove previous two
    print("[\033[1mINFO\033[0m] Merging ROOT files ...")
    subprocess.call("$ROOTSYS/bin/hadd -f -v %s %s" % (self.fout, " ".join(files_slim)), shell=True)
    subprocess.call("rootls -t %s" % self.fout, shell=True)
    subprocess.call("rm %s" % " ".join(files_slim), shell=True)


  def _shuffle (self, steps=100):

    subprocess.call("%s/lib/shuffle/shuffle --path2tmp %s/tmp --fraction 0.01 -t %s --fin %s --fout %s" % (path2home, path2home, self.info["TreeName"], self.fout, self.fout), shell=True)

    # Real shuffeling
    ROOT.ROOT.DisableImplicitMT()
    RDF = ROOT.RDataFrame(self.info["TreeName"], self.fout)
    files_shuffled = []
    for step in range(steps):
      files_shuffled.append(self.fout.replace(".root", ".%s.root" % step))
      RDF.Filter("(rdfentry_ + {}) % {} == 0".format(step, steps)).Snapshot(self.info["TreeName"], files_shuffled[-1])
      print("[\033[1mINFO\033[0m] Current batch %s/%s" % (step+1, steps))
    random.Random(42).shuffle(files_shuffled)
    subprocess.call("$ROOTSYS/bin/hadd -f -v %s %s" % (self.fout, " ".join(files_shuffled)), shell=True)
    subprocess.call("rm %s" % " ".join(files_shuffled), shell=True)

    subprocess.call("%s/lib/shuffle/shuffle --path2tmp %s/tmp --fraction 0.01 -t %s --fin %s --fout %s" % (path2home, path2home, self.info["TreeName"], self.fout, self.fout), shell=True)


  def _preprocess (self):

    subprocess.call("%s/lib/preprocessing/preprocessing -t %s --fin %s --fout %s" % (path2home, self.info["TreeName"], self.fout, self.fout), shell=True)


  def _add_weights (self, pt="fjet_truthJet_pt"):

    # Get the pT as we want a flat pT spectra
    arr = root_numpy.root2array(self.fout, self.info["TreeName"], branches=[pt, "fjet_signal", "fjet_testing_weight_pt"])
    # utils.remove_branch(self.fout, self.info["TreeName"], "fjet_testing_weight_pt")

    # Get correction factors for testing weights
    w = utils.correct_weight(arr["fjet_signal"], arr["fjet_testing_weight_pt"])
    utils.add_branch(self.fout, self.info["TreeName"], w, "fjet_correct_testing_weight_pt")

    # Compute weights to flatten plt spectra
    w = utils.train_weights(arr["fjet_signal"], arr[pt])
    utils.add_branch(self.fout, self.info["TreeName"], w, __train_weight__)

    # Compute weights to match bkg to sig spectra
    w = utils.match_weights(arr["fjet_signal"], arr[pt])
    utils.add_branch(self.fout, self.info["TreeName"], w, __match_weight__)

    # Some events come with very large weights(?); make a cut
    ROOT.ROOT.EnableImplicitMT(self.n_threads)
    RDF = ROOT.RDataFrame(self.info["TreeName"], self.fout) \
      .Filter("%s<100" % __train_weight__).Filter("%s<100" % "fjet_correct_testing_weight_pt") \
      .Snapshot(self.info["TreeName"], self.fout.replace(".root", ".limit_weights.root"))
    subprocess.call("mv %s %s" % (self.fout.replace(".root", ".limit_weights.root"), self.fout), shell=True)
 

  def _save2root (self):

    # Add metadata to ROOT file
    from src.myroot.fio import TFile
    with TFile(self.fout, "UPDATE") as fROOT:
      # Get info object associated to tree
      for obj in fROOT.GetListOfKeys():
        for key in self.info:
          fROOT.Get(obj.GetName()).GetUserInfo().Add(ROOT.TNamed(key, str(self.info[key])))
        # Update user info in tree
        fROOT.Get(obj.GetName()).Write()
        print("[\033[1mINFO\033[0m] Added meta data to TTree `%s` in file `%s`" % (obj.GetName(), self.fout))


  def _split (self):

    keep4train = 3
    # !Multithreading must be disabled for this step!
    ROOT.ROOT.EnableImplicitMT(0)
    # RDF for signal and background
    RDF = ROOT.RDataFrame(self.info["TreeName"], self.fout)
    RDF.Filter("rdfentry_ % {} != 0".format(keep4train)) \
       .Snapshot(__sec_train__, self.fout.replace(".root", ".train.root"))
    RDF.Filter("rdfentry_ % {} == 0".format(keep4train)) \
       .Snapshot(__sec_valid__, self.fout.replace(".root", ".valid.root"))
    # Remove previous tree and merge train and valid
    subprocess.call("rm %s" % self.fout, shell=True)
    subprocess.call("$ROOTSYS/bin/hadd -v %s %s %s" % (self.fout, self.fout.replace(".root", ".train.root"), self.fout.replace(".root", ".valid.root")), shell=True)
    subprocess.call("rm %s % s" % (self.fout.replace(".root", ".train.root"), self.fout.replace(".root", ".valid.root")), shell=True)


  def inspect(self, outdir="out/data", pt="fjet_truthJet_pt", weights="fjet_testing_weight_pt"):

    # Create histograms
    if not os.path.exists(outdir): os.makedirs(outdir)
    subprocess.call("%s/tool/inspect_data.py --fin %s --outdir %s --weight-test fjet_testing_weight_pt --treename %s" % (path2home, self.fout, outdir, __sec_train__), shell=True)

    # Get list of branches in file
    RDF = ROOT.RDataFrame(__sec_train__, self.fout)
    cols  = [col for col in RDF.GetColumnNames()]
    cols += ["UFO_m_0_100", "UFO_m_0_0p2", "UFO_m_0p139_0p140", "pt_weighted_train", "pt_weighted_test", "nUFO"]

    # Start plotting
    fout_sig = os.path.join(outdir, "inspection_sig.root")
    fout_bkg = os.path.join(outdir, "inspection_bkg.root")
    from src.myroot.fio import FileIo
    fio = FileIo()
    fio.addFile(fout_sig, "update")
    fio.addFile(fout_bkg, "update")
    from myplt.figure import Figure
    # Plot all branches
    for branch in cols:
      o_sig, o_bkg = fio.getObj(branch, fout_sig), fio.getObj(branch, fout_bkg)
      if o_sig is None or o_bkg is None: continue
      fig = Figure(canv=(600,500)).setDrawOption("PMC PLC HIST") \
        .addHist([o_sig, o_bkg]).addLegend(entries=["Signal;l", "Background;l"]) \
        .addAxisTitle(title_x=branch).setLogY().save(os.path.join(outdir, "plt", branch)).close()

    # Average jet image
    o_sig, o_bkg = fio.getObj("avrg_jet", fout_sig), fio.getObj("avrg_jet", fout_bkg)
    for o, name in [(o_sig, "top"), (o_bkg, "qcd")]:
      Figure(canv=(600,500)).setColorPalette(103).setDrawOption("COLZ") \
        .addHist(o).setLogZ().setRangeZ(1E-4, o.GetMaximum()) \
        .addAxisTitle(
          title_x="Shifted pseudorapidity #eta",
          title_y="Shifted azimuthal angle #phi",
          title_z="Accumulated energy [GeV]") \
        .setPadMargin().addLegend(entries=["#scale[1.5]{#bf{%s}};p" % name], xmin=0.6, ymin=0.75) \
        .save(os.path.join(outdir, "plt", "avrg_jet_%s" % name)).close()


  def cols2keep (self, branches):

    if not isinstance(branches, list): branches = [branches]
    for branch in branches: self.branches.push_back(branch)
    print("[INFO] Following columns will be included in the final data set")
    for col in self.branches: print("       |- %s" % col)


  def setTreeName (self, treename):

    self.info["TreeName"] = treename
    return self


  def setFilesSig (self, flist):

    self.info["FilesSig"] = flist
    return self


  def setFilesBkg (self, flist):

    self.info["FilesBkg"] = flist
    return self


  def setNEvents (self, n_req):

    assert self.info["FilesSig"] and self.info["FilesBkg"], "[ERROR] Please provide the input files first"
    # Create temporary RDF to check if requested number exeeds the available one
    flist = src.myroot.reader.read_file_list(self.info["FilesSig"]) + src.myroot.reader.read_file_list(self.info["FilesBkg"])
    n_avail = ROOT.RDataFrame(self.info["TreeName"], src.myroot.convert.conv2vec(flist)).Count().GetValue()
    self.n_tot = int(n_req)
    if n_avail < self.n_tot:
      print("[WARNING] %s events have been requested, but only %s are available." % (self.n_tot, n_avail))
      self.n_tot = n_avail
    return self


  def setFracTrain (self, frac):

    assert frac <= 1, "[ERROR] Fraction of training events must be smaller than 1."
    self.info["FraTrain"] = frac
    return self


  def setSelSig(self, selection):

    self.info["SelectionSig"] = selection.encode("ascii", "ignore")
    return self


  def setSelBkg(self, selection):

    self.info["SelectionBkg"] = selection.encode("ascii", "ignore")
    return self


  def setSelTrain (self, selection):

    self.info["SelectionTrain"] = selection.encode("ascii", "ignore")


  def build (self, save_h5=False):

    # Make some preparations before starting
    with src.myutils.profile.Profile("Prepare run"):
      self._prep()

    # Slim the data set
    with src.myutils.profile.Profile("Slim data"):
      self._slim()

    # Shuffle data
    with src.myutils.profile.Profile("Shuffle"):
      self._shuffle()

    # Preprocess constituent data and add preprocessed columns
    with src.myutils.profile.Profile("Preprocess data, add columns"):
      self._preprocess()

    # Compute weights to get flat pt
    with src.myutils.profile.Profile("Compute training weights"):
      self._add_weights()

    # Split into training and testing set
    with src.myutils.profile.Profile("Split data into training and validation data"):
      self._split()

    # Save as ROOT file
    with src.myutils.profile.Profile("Save data to ROOT file"):
      self._save2root()


  def useMT (self, n_threads=0):

    print("[\033[1mINFO\033[0m] Requested multi threading with `%s` threads" % n_threads)
    if n_threads > 50: print("[\033[1m    \033[0m] You are quite greedy :)")
    self.n_threads = n_threads
    ROOT.ROOT.EnableImplicitMT(n_threads)
