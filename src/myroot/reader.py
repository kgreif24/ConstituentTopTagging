import os, sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

import termcolor
import utils
import cut as cut_module
import fio
import ROOT
import common.profile


# RDF-related variables
__n_empty_cache__ = 100

# Some global variables
__t_file__ = None
__t_dir__ = "/"
__t_tree__ = ""
__samples__ = {}

# Histogram cache appends all generated histograms to a list if requested
__hist_cache__ = []

# `Private` global variables
__rdf__ = {}
__ptr_cache__ = {}
__hist2cache__ = True


def add_h1_to_cache(rdf_name, branch_exp, t_h1_model, weight="", cut="", verbose=True):

  """
  Add a 1D histogram to histogram cache. All histograms in this list are read in parallel.

  This function appends a 1D histogram to to global variable `__hist_cache__`. The histogram readout is not executed before `__n_empty_cache__` or `read_cache` is triggered. 

  :param rdf_name: name of the RDF that is supposed to be populated
  :param branch_exp: name of the branch/variable to be read from ROOT file that has been specified in root.read.set_df(tree_name, samples(list of ROOT files))
  :param t_h1_model: a struct (cf. ROOT.RDF.TH1DModel) that defines the paramters of a TH1D
  :param weights: event weights
  :param cut: boolean cuts applied on an event-on-event base
  :type branch_exp: str
  :type th1_model: ROOT.RDF.TH2DModel (https://root.cern/doc/master/structROOT_1_1RDF_1_1TH1DModel.html)
  :type weight: str or list of str
  :type cut: str or list of str
  """

  # Do some checks
  if not __rdf__[rdf_name]:
    print("[ERROR] No RDF instance has been found")
    sys.exit()

  # Standardize expressions
  weight_std = cut_module.standardize_exp(weight, "*", rtype=str)
  cut_std = cut_module.standardize_exp(cut, "&&", rtype=list)
  if weight_std == "": weight_std = "1"

  # Create a new temporary RDF branch
  rdf_tmp = __rdf__[rdf_name].Define("__branch__", branch_exp) \
                             .Define("__weight__", weight_std)
  print("[INFO] Branch (__branch__): %s - event weight (__weight__): %s" % (branch_exp, weight_std))

  # Apply filters/cuts to temporary RDF
  for filter_name in cut_std:
    if not filter_name:
      continue
    else:
      rdf_tmp = rdf_tmp.Filter(filter_name, filter_name)

  # Make a clone of the model to not overwrite its properties
  t_h1_model_copy = utils.get_tH1_model(t_h1_model.fName, t_h1_model.fTitle, t_h1_model.fNbinsX, t_h1_model.fXLow, t_h1_model.fXUp)

  # Set histogram name
  if not t_h1_model_copy.fName:
    t_h1_model_copy.fName = utils.get_h_name(branch_exp, t_h1_model_copy, weight, cut)

  if verbose:
    print("[INFO] Metadata for (%s): %s" % (rdf_name, t_h1_model_copy.fName))
    print("       ]> List of Columns:")
    for col in rdf_tmp.GetDefinedColumnNames():
      print("       |- \33[1m%s\33[0m" % col)
    print("       ]> List of filters:")
    for filter_name in rdf_tmp.GetFilterNames():
      print("       |- \33[1m%s\33[0m" % filter_name)
  # Get histogram
  t_h1_ptr = rdf_tmp.Histo1D(t_h1_model_copy, "__branch__", "__weight__")  

  global __ptr_cache__
  __ptr_cache__[t_h1_model_copy.fName] = t_h1_ptr

  # Check if cache should be read
  if len(__ptr_cache__) == __n_empty_cache__ and __t_file__: read_cache(__t_file__)


def add_h2_to_cache(rdf_name, x_branch_exp, y_branch_exp, t_h2_model, weight="", cut="", verbose=True):

  """
  Add a 2D histogram to histogram cache. All histograms in this list are read in parallel.

  This function appends a 1D histogram to to global variable `__hist_cache__`. The histogram readout is not executed before `__n_empty_cache__` or `read_cache` is triggered. 

  :param rdf_name: name of the RDF that is supposed to be populated
  :param x_branch_exp: expression of the first branch/variable to be read from ROOT file that has been specified in root.read.set_df(tree_name, samples(list of ROOT files))
  :param y_branch_exp: expression of the second branch/variable to be read from ROOT file that has been specified in root.read.set_df(tree_name, samples(list of ROOT files))
  :param t_h2_model: a struct (cf. ROOT.RDF.TH2DModel) that defines the paramters of a TH2D
  :param weights: event weights
  :param cut: boolean cuts applied on an event-on-event base
  :type x_branch_exp: str
  :type y_branch_exp: str
  :type th1_model: ROOT.RDF.TH2DModel (https://root.cern/doc/master/structROOT_1_1RDF_1_1TH1DModel.html)
  :type weight: str or list of str
  :type cut: str or list of str
  """

  # Do some checks
  if not __rdf__[rdf_name]:
    print("[ERROR] No RDF instance has been found")
    sys.exit()

  # Standardize expressions
  weight_std = cut_module.standardize_exp(weight, "*", rtype=str)
  cut_std = cut_module.standardize_exp(cut, "&&", rtype=list)
  if weight_std == "": weight_std = "1"

  # Create a new temporary RDF branch
  rdf_tmp = __rdf__[rdf_name].Define("__x_branch__", x_branch_exp) \
                             .Define("__y_branch__", y_branch_exp) \
                             .Define("__weight__", weight_std)

  # Apply filters/cuts to temporary RDF
  for filter_name in cut_std:
    if not filter_name: continue
    else: rdf_tmp = rdf_tmp.Filter(filter_name, filter_name)

  # Make a clone of the model to not overwrite its properties
  import copy
  t_h2_model_copy = copy.copy(t_h2_model)

  # Set histogram name
  if not t_h2_model_copy.fName:
    t_h2_model_copy.fName = utils.get_h_name("%s:%s" % (x_branch_exp, y_branch_exp), t_h2_model_copy, weight, cut)

  if verbose:
    print("[INFO] Metadata for (%s): %s" % (rdf_name, t_h2_model_copy.fName))
    print("       ]> List of Columns:")
    for col in rdf_tmp.GetDefinedColumnNames():
      print("       |- \33[1m%s\33[0m" % col)
    print("       ]> List of filters:")
    for filter_name in rdf_tmp.GetFilterNames():
      print("       |- \33[1m%s\33[0m" % filter_name)

  # Get histogram
  t_h2_ptr = rdf_tmp.Histo2D(t_h2_model_copy, "__x_branch__", "__y_branch__", "__weight__")  

  global __ptr_cache__ 
  __ptr_cache__[t_h2_model_copy.fName] = t_h2_ptr

  # Check if cache should be read
  if len(__ptr_cache__) == __n_empty_cache__ and __t_file__: read_cache(__t_file__)


def clear_hist_cache():

  """
  Clear all histograms that are currently defined in `__hist_cache__`.

  This function is automatically called after `__n_empty_cache__` or `read_cache` hase been triggered.
  """

  global __hist_cache__
  __hist_cache__ = []


def get_h1_from_file(t_file, tree_name, branch_name, x_range, selection="", name="t_h1_tmp", title=""):

  """
  Read 1D histogram from a singe TFile with selections (event weights and/or cuts) via TTree.Draw.

  :param t_file: name of the ROOT file or a TFile object
  :param tree_name: name of the tree or ntuple name in ROOT file
  :param branch_name: name of the branch or variable to read from tree in ROOT file
  :param x_range: string that defines the hoistogram binning and range (n_bins,x_min,x_max)
  :param selection: even weights and cuts to be applied on an event-on-event basis
  :param name: name of the histogram
  :type t_tfile: ROOT.TFile (https://root.cern.ch/doc/master/classTFile.html)
  :type tree_name: str
  :type branch: str
  :type x_range: str
  :type selection: str
  :type name: str
  :rtype TH1D
  """

  print("[INFO] Drawing (TH1): %s" % t_file)
  print("[INFO] Populating histogram: %s" % name)
  # Open ROOT file in this scope
  if isinstance(t_file, ROOT.TFile):
    t_file_tmp = t_file
  else:
    t_file_tmp = ROOT.TFile.Open(t_file, "READ")
  tree_tmp = t_file_tmp.Get(tree_name)
  # Get histogram with selection criteria applied
  tree_tmp.Draw("%s>>t_h1_tmp(%s)" % (branch_name, x_range), selection, "e")
  t_h1_tmp = ROOT.gDirectory.Get("t_h1_tmp")
  t_h1_tmp.SetName(name)
  t_h1_tmp.SetTitle(title)
  # Do not write this histogram to any file yet
  t_h1_tmp.SetDirectory(0)
  # Done reading; close file and return histogram
  t_file_tmp.Close()
  return t_h1_tmp


def get_h2_from_file(t_file, tree_name, x_branch_name, y_branch_name, x_range, y_range, selection="", h_name="t_h2_tmp", n=None):

  """
  Read 2D histogram from a singe TFile with selections (event weights and/or cuts) via TTree.Draw.

  :param t_file: name of the ROOT file or a TFile object
  :param tree_name: name of the tree or ntuple name in ROOT file
  :param x_branch_name: name of the first branch or variable to read from tree in ROOT file
  :param y_branch_name: name of the second branch or variable to read from tree in ROOT file
  :param x_range: string that defines the hoistogram binning and range (n_binsx,x_min,x_max,n_binsy, y_min,y_max)
  :param selection: even weights and cuts to be applied on an event-on-event basis
  :param h_name: name of the histogram
  :type t_file: ROOT.TFile (https://root.cern.ch/doc/master/classTFile.html)
  :type tree_name: str
  :type x_branch_name: str
  :type y_branch_name: str
  :type x_range: str
  :type selection: str
  :type h_name: str
  :rtype TH2D
  """

  print("[INFO] Drawing (TH2): %s" % t_file)
  # Open ROOT file in this scope
  if not isinstance(t_file, ROOT.TFile):
    t_file_tmp = ROOT.TFile.Open(t_file, "READ")
  else:
    t_file_tmp = t_file
  tree_tmp = t_file_tmp.Get(tree_name)
  # Get histogram with selection criteria applied
  if n == None:
    tree_tmp.Draw(x_branch_name+":"+y_branch_name+">>t_h2_tmp("+x_range+","+y_range+")", selection, "e")
  else:
    tree_tmp.Draw(x_branch_name+":"+y_branch_name+">>t_h2_tmp("+x_range+","+y_range+")", selection, "e", n)
  t_h2_tmp = ROOT.gDirectory.Get("t_h2_tmp")
  t_h2_tmp.SetName(h_name)
  # Do not write this histogram to any file yet
  t_h2_tmp.SetDirectory(0)
  # Done reading; close file and return histogram
  t_file_tmp.Close()
  print("[INFO] Done populating histogram: %s" % t_h2_tmp)
  return t_h2_tmp


def get_h1_from_files(samples, tree_name, branch_name, t_h1_model, selection="", normalize=False, t_file=__t_file__, directory=__t_dir__):

  """
  Read 1D histogram from a singe or multiple TFile(s) with selections (event weights and/or cuts) via TTree.Draw.

  :param samples: name of the ROOT files or a TFile objects
  :param tree_name: name of the tree or ntuple name in ROOT file
  :param branch_name: name of the branch or variable to read from tree in ROOT file
  :param t_h1_model: a struct (cf. ROOT.RDF.TH1DModel) that defines the paramters of a TH1D
  :param selection: even weights and cuts to be applied on an event-on-event basis
  :param normalize: normalize the histogram that is returned (the histogram saved to the root file is not normalized)
  :param t_file: a ROOT file to save the generated histograms
  :param dir: directory in `t_file` in which the(summed) histogram will be saved
  :type t_tfile: str of list of str or ROOT.TFile or list of ROOT.TFile
  :type tree_name: str
  :type branch_name: str
  :type th1_model: ROOT.RDF.TH2DModel (https://root.cern/doc/master/structROOT_1_1RDF_1_1TH1DModel.html)
  :type selection: str
  :type normalize: bool
  :type t_file: ROOT.TFile (https://root.cern.ch/doc/master/classTFile.html)
  :type dir: str
  :rtype (summed) TH1D
  """

  # Convert to list
  if not isinstance(samples, list): samples = [samples]

  # Set file
  t_file = t_file if t_file is not None else __t_file__

  # Get name of histogram
  x_range = "%d,%s,%s" % (t_h1_model.fNbinsX, t_h1_model.fXLow, t_h1_model.fXUp)
  if not t_h1_model.fName: t_h1_model.fName = utils.get_std_name(branch_name, x_range, selection)

  # If a root file is given, check if this particular histogram already exists
  t_h1 = None
  if t_file:
    t_h1 = fio.browse_root_file(t_file, t_h1_model.fName)
    print("[INFO] The histogram `%s` has been found in file `%s`" % (t_h1.GetName(), t_file))

  # If histogram has not been found, read from file
  if not t_h1 or not t_file:
    print("[INFO] \33[3m> Start reading histograms <\33[0m")
    t_h1 = get_h1_from_file(samples[0], tree_name, branch_name, x_range, selection, h_name=t_h1_model.fName)
    for i_sample in range(len(samples)-1):
      if i_sample+1 > 0:
        t_h1_tmp = get_h1_from_file(samples[i_sample+1], tree_name, branch_name, x_range, selection, h_name=t_h1_model.fName)
        t_h1.Add(t_h1_tmp)

  # Add to file
  if t_file: fio.add_obj_to_file(obj, t_file, directory)
  # Normalize if requested
  if normalize: t_h1 = utils.norm_h(t_h1)

  return t_h1


def get_h1_from_array(x, nbins=None, range_x=(None, None), title_x="", title_y="", normalize=True, name=None, title="", weights=[]):

  """
  Generate 1D histogram from arry

  :param x: an array of data points
  :param nbins: number of bins
  :param range_x: range of x axis
  :param title_x: title of x axis
  :param title_y: title of y axis
  :param normalize: is the histogram supposed to be normalized to unity?
  :param name: name of the histogram that os returned
  :param weights: a list of weights for each eacht in x
  :type x: list or numpy array
  :type nbins: int
  :type range_x: tuple
  :type title_x: str
  :type title_y: str
  :type normalize: bool
  :type name: str
  :type weights list or numpy array

  """

  # Get some parameters of the historam
  if not name: name = "unnamed"
  if not nbins: nbins = len(x)
  xmin, xmax = range_x[0], range_x[1]
  if xmin is None: xmin = min(x)
  if xmax is None: xmax = max(x)

  # Get histogram
  t_h1 = ROOT.TH1F(name, title, nbins, xmin, xmax)
  t_h1.SetLineWidth(3)
  # Fill histogram
  if len(weights) == 0:
    for xx in x:
      t_h1.Fill(xx)
  else:
    for xx, ww in zip(x, weights):
      t_h1.Fill(xx, ww)
  # Normalize? Ifo so, also update title
  if normalize:
    t_h1 = utils.norm_h(t_h1)
    title_y = "Entries normalized to unity / %.3f" % t_h1.GetBinWidth(1)
  t_h1.GetXaxis().SetTitle(title_x)
  t_h1.GetYaxis().SetTitle(title_y)
  return t_h1


def get_h2_from_array(x, nbins_x=None, range_x=(None, None), nbins_y=None, range_y=(None, None), title_x="", title_y="", normalize=False, name=None, title="", weights=[]):

  """
  Generate 2D histogram from arry

  :param x: an array of data points
  :param nbins_x: number of bins
  :param nbins_y: number of bins
  :param range_x: range of x axis
  :param range_y: range of y axis
  :param title_x: title of x axis
  :param title_y: title of y axis
  :param normalize: is the histogram supposed to be normalized to unity?
  :param name: name of the histogram that os returned
  :param weights: a list of weights for each eacht in x
  :type x: list or numpy array
  :type nbins: int
  :type range_x: tuple
  :type title_x: str
  :type title_y: str
  :type normalize: bool
  :type name: str
  :type weights list or numpy array

  """

  # Get some parameters of the historam
  if not name: name = "unnamed"
  if not nbins_x: nbins_x = x.shape[-2]
  if not nbins_y: nbins_y = x.shape[-1]
  xmin, xmax = range_x[0], range_x[1]
  ymin, ymax = range_y[0], range_y[1]
  if xmin is None: xmin = 1
  if xmax is None: xmax = x.shape[-2]
  if ymin is None: ymin = 1
  if ymax is None: ymax = x.shape[-1]

  # Get histogram
  t_h2 = ROOT.TH2F(name, title, nbins_x, xmin, xmax, nbins_y, ymin, ymax)
  # Fill histogram
  for i in range(x.shape[-2]):
    for j in range(x.shape[-1]):
      t_h2.SetBinContent(i+1, j+1, x[i][j])
  # Normalize? Ifo so, also update title
  if normalize:
    t_h2 = utils.norm_h(t_h2)
  t_h2.GetXaxis().SetTitle(title_x)
  t_h2.GetYaxis().SetTitle(title_y)
  return t_h2



def get_graph_from_array(x, y, title_x="", title_y="", name=None):

  """
  Generate TGraph from an array

  :param x: an array of data points
  :param y: an array of data points
  :param title_x: title of x axis  
  :param title_y: title of y axis  
  :param name: name of the histogram that os returned
  :type x: list or numpy array
  :type y: list or numpy array
  :type title_x: str
  :type title_y: str
  :type name: str
  """

  # Get some parameters of the historam
  if not name: name = "unnamed"

  import array
  ax, ay = array.array("d", x), array.array("d", y)
  n = len(ax)
  # create tgraph
  tG = ROOT.TGraph(n, ax, ay)
  if name: tG.SetName(name)
  tG.GetXaxis().SetTitle(title_x), tG.GetYaxis().SetTitle(title_y)
  return tG


def get_h2_from_files(samples, tree_name, x_branch_name, y_branch_name, t_h2_model, selection="", normalize=False, t_file=__t_file__, directory=__t_dir__):

  """
  Read 2D histogram from a singe or multiple TFile(s) with selections (event weights and/or cuts) via TTree.Draw.

  :param samples: name of the ROOT files or a TFile objects
  :param tree_name: name of the tree or ntuple name in ROOT file
  :param x_branch_name: name of the first branch or variable to read from tree in ROOT file
  :param y_branch_name: name of the second branch or variable to read from tree in ROOT file
  :param t_h2_model: a struct (cf. ROOT.RDF.TH1DModel) that defines the paramters of a TH1D
  :param selection: even weights and cuts to be applied on an event-on-event basis
  :param normalize: normalize the histogram that is returned (the histogram saved to the root file is not normalized)
  :param t_file: a ROOT file to save the generated histograms
  :param dir: directory in `t_file` in which the(summed) histogram will be saved
  :type t_tfile: str of list of str or ROOT.TFile or list of ROOT.TFile
  :type tree_name: str
  :type x_branch_name: str
  :type y_branch_name: str
  :type th2_model: ROOT.RDF.TH2DModel (https://root.cern/doc/master/structROOT_1_1RDF_1_1TH2DModel.html)
  :type selection: str
  :type normalize: bool
  :type t_file: ROOT.TFile (https://root.cern.ch/doc/master/classTFile.html)
  :type dir: str
  :rtype (summed) TH2D
  """

  # Convert to list
  if not isinstance(samples, list): samples = [samples]

  # Set file
  t_file = t_file if t_file is not None else __t_file__

  # Get name of histogram
  x_range = "%d,%s,%s" % (t_h2_model.fNbinsX, t_h2_model.fXLow, t_h2_model.fXUp)
  y_range = "%d,%s,%s" % (t_h2_model.fNbinsY, t_h2_model.fYLow, t_h2_model.fYUp)
  if not t_h2_model.fName: t_h2_model.fName = utils.get_std_name("%s:%s" % (x_branch_name, y_branch_name), "%s%s" % (x_range, y_range), selection)

  # If a root file is given, check if this particular histogram already exists
  t_h2 = None
  if t_file:
    t_h2 = fio.browse_root_file(t_file, t_h2_model.fName)
    print("[INFO] The histogram `%s` has been found in file `%s`" % (t_h2.GetName(), t_file))

  # If histogram has not been found, read from file
  if not t_h2 or not t_file:
    print("[INFO] \33[3m> Start reading histograms <\33[0m")
    t_h2 = get_h2_from_file(samples[0], tree_name, x_branch_name, y_branch_name, x_range, y_range, selection, h_name=t_h2_model.fName)
    for i_sample in range(len(samples)-1):
      if i_sample+1 > 0:
        t_h2_tmp = get_h2_from_file(samples[i_sample+1], tree_name, x_branch_name, y_branch_name, x_range, y_range, selection, h_name=t_h2_model.fName)
        t_h2.Add(t_h2_tmp)

  # Add to file
  if t_file: fio.add_obj_to_file(obj, t_file, directory)
  # Normalize if requested
  if normalize: t_h2 = utils.norm_h(t_h2)

  return t_h2


def get_RDF (tree_name, samples, default_branches=[]):

  """
    Use ROOT's RDF MT for fast read of samples files (this is super fast!)
    Get the selection cuts for signal and background (truth label, acceptance cuts ect.)

      Parameters:
        tree_name (str): Name of the TTree in the ROOT File supposed to be read
        samples (str/list of str) Names of files to be read
        default_branches (list of str): What to read from the tree

      Returns:
        RDF (ROOT.RDataFrame): Returns a ROOT.RDataFrame pointing to the files
  """

  # Samples may be just one file name or a list of those
  if not isinstance(samples, list): samples = [samples]
  if not isinstance(default_branches, list): default_branches = [default_branches]

  # Init RDF
  rdf = None

  # RDF needs an std::vector<string>
  if all([isinstance(element, str) for element in samples]):
    samples_list = ROOT.vector("string")()
    default_branches_list = ROOT.vector("string")()
    for sample in samples:samples_list.push_back(sample.replace(" ", ""))
    for branch in default_branches: default_branches_list.push_back(sample)
    # Set RDF with constructor RDataFrame(tree_name, std::vector<std::string> samples)
    rdf = ROOT.RDataFrame(tree_name, samples_list, default_branches_list)

  if all([isinstance(element, ROOT.TFile) for element in samples]):
    t_chain = ROOT.TChain(tree_name)
    for sample in samples: t_chain.Add(samples)
    for branch in default_branches: default_branches_list.push_back(sample)
    # Set RDF with constructor RDataFrame(ROOT.TChain)
    rdf = ROOT.RDataFrame(t_chain, default_branches_list)

  return rdf


def ls_hist_cache():

  """
  List all histograms currently in `__hist_cache__`.
  """

  print("[INFO] Current content of `__hist(ogram)_cache__`:")
  for h in __hist_cache__:
    print("       |- %s" % h.GetName())


def ls_config():

  """
  List the current configuration of all global variables.
  """

  # Get a list of all global variables in this module
  list_of_globals = [glob for glob in globals().keys() if glob.startswith("__") and glob.endswith("__")]

  # Remove some standards global variables that are part of all python modules
  for item in ["__file__", "__builtins__", "__samples__", "__doc__", "__name__"]: list_of_globals.remove(item)

  # Print output
  print("[INFO] Current configuration of global variables:")
  for item in list_of_globals:
    # Check if var is iterable
    print("       |- %s: %s" % (item, globals()[item]))


def ls_branches(fname, treename, verbose=True):

  t_f = ROOT.TFile(fname)
  # Retrieve branches from tree in file
  branches = [branch.GetName() for branch in t_f.Get(treename).GetListOfBranches()]
  if verbose:
    print("[INFO] Found the following branches in file %s in tree %s" % (termcolor.colored(t_f.GetName(), attrs=["bold"]), termcolor.colored(treename, attrs=["bold"])))
    for branch in branches:
      print("|- %s" % branch)
  t_f.Close()
  return branches


@common.profile.profile
def read_cache(t_file=__t_file__):

  """
  Read all histograms (1D,2D and 3D) from current content of `__ptr_cache__` to file and, if requested, to `__hist_cache`.

  :param t_file: a ROOT file to save the generated histograms
  :type t_tfile: str of ROOT.TFile
  """

  # Set file
  global __t_file__
  t_file = t_file if t_file is not None else __t_file__

  if not t_file: print("[WARNING] No file has been provided. The histograms will be returned as a list")
  else: print("[INFO] The following historams will be saved to file: %s" % t_file.GetName())

  # Parallel magic happens here
  global __ptr_cache__

  # If ptr cache is empty, done
  if not __ptr_cache__: return

  # Print some infos
  for hist_name in __ptr_cache__:
    print("       |- %s" % hist_name)

  # Store histograms in list if no file is provided
  t_h_list = []

  # Check if any histogram in `__ptr_cache__` is already present in file (if it is given)
  if t_file:
    n_el_in_ptr_cache = len(__ptr_cache__)
    print("[INFO] Check on pre-existing histograms in: `%s`" % t_file.GetName())
    # Loop over all histogram names and chek if they are in file
    for hist_name in dict(__ptr_cache__):
      t_h_found = fio.browse_root_file(t_file, hist_name)
      if t_h_found:
        del __ptr_cache__[hist_name]
        t_h_list.append(t_h_found)
        if t_h_list[-1] in __hist_cache__:
          __hist_cache__.append(t_h_list[-1])
        print("       |- Found `%s` ==> remove from `__ptr_cache__` " % hist_name)
    # If elements have been removed, print content once again
    if n_el_in_ptr_cache != len(__ptr_cache__):
      print("[INFO] Preexisting histograms have been removed. The following objects will be saved to file: %s" % t_file.GetName())
      # Print some infos
      for hist_name in __ptr_cache__:
        print("       |- %s" % hist_name)

  # read histograms in parallel
  for hist_name in __ptr_cache__:
    if t_file:
      t_file.cd(__t_dir__)
      __ptr_cache__[hist_name].Write()
      t_h_list.append(__ptr_cache__[hist_name].GetValue())
      t_file.Flush()
    else:
      t_h_list.append(__ptr_cache__[hist_name].GetValue())
    if not t_file or __hist2cache__:
        if t_h_list[-1] in __hist_cache__:
          __hist_cache__.append(t_h_list[-1])

  # Reinitialize for the next run
  __ptr_cache__ == {}

  if not t_file: return t_h_list
  else: print("[INFO] Output has been written to file: %s" % t_file.GetName())


def read_sample_file(fname):

  """
  Read location of ntuples from *.list files

  :param fname: path to *.list file that holds location of ntuples
  :type dir: str
  :rtype str
  """

  sample_list = []
  if os.path.isdir(fname):
    for f in os.listdir(fname):
      if f.endswith(".root"): sample_list.append(os.path.join(fname, f))
  if os.path.isfile(fname):
    with open(fname, "r") as f:
      for line in f: sample_list.append(line.replace("\n", ""))
  # Print list
  print("[INFO] Found the following signal files:")
  for f in sample_list:
    print("  |- %s" % f)
  return sample_list 



def set_df(name, tree_name, samples):

  """
  Initialize RDF (https://root.cern/doc/master/classROOT_1_1RDataFrame.html) with a tree_name and a list of ROOT file.

  :param tree_name: name of the tree to read from samples
  :param tree_name: list of samples
  :type tree_name: str
  :type samples: str of list of str or ROOT.TFile or list of ROOT.TFile
  """

  # Samples may be just one file name or a list of those
  if not isinstance(samples, list): samples = [samples]

  global __rdf__, __samples__
  # RDF needs an std::vector<string>
  if all([isinstance(element, str) for element in samples]):
    samples_list = ROOT.vector("string")()
    for sample in samples: samples_list.push_back(sample)
    # Set RDF with constructor RDataFrame(tree_name, std::vector<std::string> samples)
    __rdf__[name] = ROOT.RDataFrame(tree_name, samples_list)
    __samples__[name] = samples

  if all([isinstance(element, ROOT.TFile) for element in samples]):
    t_chain = ROOT.TChain(tree_name)
    for sample in samples: t_chain.Add(samples)
    # Set RDF with constructor RDataFrame(ROOT.TChain)
    __rdf__[name] = ROOT.RDataFrame(t_chain)
    __samples__[name] = t_chain.GetListOfFiles()

  # Reset cache
  global __ptr_cache__
  __ptr_cache__ == {}

  # Set some other global variables
  global __t_tree__
  __t_tree__ = tree_name

  # Print status
  print("[INFO] DataFrame has been successfully initialized.")
  print("       |- Tree name: %s" % tree_name)
  print("       |_ Sample files:")
  for fname in __samples__[name]:
    print("         |- %s " % fname)


def set_current_file(t_file, directory="/"):

  """
  Set the current and global ROOT file that will be used to save all objects.

  :param t_file: the ROOT.File to save all objects
  :param directory: directory in the ROOT file in which all objects are saved
  :type t_file: ROOT.TFile
  :type directory: str
  """

  # If t_file is a file name, read a file
  if isinstance(t_file, str):
    t_file = ROOT.TFile(t_file, "READ")

  global __t_file__
  __t_file__ = t_file
  global __t_dir__
  __t_dir__ = directory

  # Set directory of this ROOT file to `directory` (default is root `/`)
  fio.cd_in_file(t_file=__t_file__, directory=__t_dir__)


def cd_dir(directory="/"):

  """
  Change directory of current root file `__t_file__`.

  :param directory: directory in the ROOT file in which all objects are saved
  :type directory: str
  """

  global __t_dir__
  __t_dir__ = directory


def set_n_empty_cache(n_empty_cache=100):

  """
  Set the capacity of `__ptr_cache__`. If `__n_empty_cache__` is reached, all objects are read and saved to file.

  :param n_empty_cache: Set the number of objects in `__ptr_cache` at which `root.reader.read_cache` is triggered
  :type n_empty_cache: int
  """

  global __n_empty_cache__
  __n_empty_cache__ = n_empty_cache


def use_hist2cache(use=True):

  """
  Save all generated histograms in the global container `__hist_cache__`.

  :param use: to use or not to use, this is the question
  :type use: bool
  """

  if not isinstance(use, bool):
    print("[ERROR] `ue_hist2cache` must be a boolean")
    sys.exit()

  global __hist2cache__
  __hist2cache__ = use
  

def use_multi_threading(n_threads=0):

  """
  Use multi threading for reading histograms.

  :param n_threads: number of threads to use for multi threading
  :type n_threads: int
  """

  print("[INFO] \33[3mUSING MULTI-THREADING\33[0m")
  print("[WARNING] Make sure that ROOT has been build with -Dimt=ON to use multi-threading features, e.g., on LXPLUS: source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.18.04/x86_64-centos7-gcc48-opt/bin/thisroot.sh")
  if n_threads == 0: print("[INFO] Using default number of threads")
  else: print("[WARNING] Using %s threads. The actual number of threads may differ depending on your system" % n_threads)

  # Activate mt
  ROOT.ROOT.EnableImplicitMT(n_threads)


def read_by_tags(tags, t_file=__t_file__, delimiter_tag="_"):

  """
  Read objects from root files based on an arbitrary number of tags

  :param tags: list of tags
  :param t_file: a root file that contains drawable TObjects 
  :param delimiter_tag: separator 
  :type tags: list of str
  :type t_file: ROOT.TFile
  :type delimiter_tag: str
  :rtype list of TObjects
  """

  # Set file
  global __t_file__
  t_file = t_file if t_file is not None else __t_file__

  # read ALL objects from file that match at least one tag
  o_list = []
  for tag in tags:
    if delimiter_tag not in tag: tag = tag+delimiter_tag
    t_o = fio.browse_root_file(t_file, hist_name, mode="loose")
    if t_o: o_list.append(t_o)

  # Only keep objects that match ALL patterns
  for o in o_list:
    if not all(tag+delimiter_tag in o for tag in tags):
      o_list.remove(o)

  # Print output
  print("[INFO] The following TObjects have been found that match the requested pattern:")
  for o in t_list:
    print("       |- %s" % o.GetName())

  return o_list


def read_file_list(fname, sort=True, extension_in=".list", extension_out=".root"):

  """
  Read a file that contains a paths to ntuples into a list for further processing

  :param fname: name of the file that contains teh paths to ntuples 
  :param sort: sort the content of the final list
  :param extension_in: file extension of the input file
  :param extension_out: file extension of the files listed in fname
  :type str: str
  :type sort: bool
  :type extension_in: str
  :type extension_out: str
  :rtype list of str
  """

  # Could also be just a file
  if fname.endswith(extension_out):
    return [fname]

  # First, check if input is a file
  sample_list = []
  if os.path.isfile(fname) and fname.endswith(extension_in):
    with open(fname, "r") as f:
      for line in f:
        sample_list.append(line.replace("\n", ""))
  
  # Get all files
  if os.path.isdir(fname):
    for root, subdirs, files in os.walk(fname):
      if files.endswith(extension_out):
        sample_list.append(os.path.join(root, files).replace(" ", ""))
  
  # If requested,sort the output
  if sort:
    return sorted(sample_list)
  else:
    return samples_list

