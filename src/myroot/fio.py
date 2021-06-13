import os, sys
import warnings
import ROOT

# Custom imports
import utils


def browse_root_file(t_file, obj_name, mode="strict"):

  """
  Browse all directories and sub-directories of the ROOT file
  and look for a TObject with name `obj_name`
  """

  if mode not in ["strict", "loose"]:
    print("[ERROR] Invalide choice of `mode` [strict/loose]")
    sys.exit()

  # Define iterator to browse all keys in file
  next_iter = ROOT.TIter(t_file.GetListOfKeys())
  key = next_iter()

  # Loop over all keys and look for object with name `obj_name`
  while (key):
    cl = ROOT.gROOT.GetClass(key.GetClassName())
    # Could be the right object. Check its name
    if cl.InheritsFrom("TH1") or cl.InheritsFrom("TGraph") or cl.InheritsFrom("TF1"):
      if mode == "strict":
        if key.GetName() == obj_name:
          return key.ReadObj()
      elif mode == "loose":
        if (key.GetName() in obj_name) or (obj_name in key.GetName()):
          return key.ReadObj()
    # The current key is a directory, and no object has been found; go one level up
    elif key.GetClassName() == "TDirectoryFile":
      t_file.cd(key.GetName())
      sub_dir = ROOT.gDirectory
      # Call this function again
      return browse_root_file(sub_dir, obj_name)
#    else:
#      return None
    # Next key in root file
    key = next_iter()


def filter_by_name(t_file, obj_name, mode="strict"):

  if isinstance(t_file, ROOT.TFile):
    # Get all object(s) in ROOT file that match name pattern
    return browse_root_file(t_file, obj_name, mode)
  else:
    f = ROOT.TFile(t_file, "READ")
    obj = browse_root_file(f, obj_name, mode)
    return (f, obj)


def filter_by_tag(t_file, tags):

  # Some checks on input
  if not isinstance(tags, list): tags = [tags]

  # A container that holds all objects that pass teh filtering
  o_list = []

  import hep.utils.name
  import numbers
  # A list that holds all strings that the object name must contain
  patterns = []
  # Loop over all tags or tag -key pairs
  for element in tags:
    # If element is a tuple, encode the tag-key pair
    if type(element) == tuple:
      patterns.append(hep.utils.name.encode(element))
    elif type(element) == str:
      patterns.append(hep.utils.name.encode(element))
    elif isinstance(element, numbers.Number):
      patterns.append(hep.utils.name.encode(str(element)))

  # Look for object(s) in file that contain at least one patterns
  for pattern in patterns:
    t_o = filter_by_name(t_file, pattern, mode="loose")
    if t_o:
      o_list.append(t_o)
      print("pattern", pattern, t_o)

  # Get rid of Nones
  o_list = filter(lambda x: x != None, o_list)

  # Filter list based on patterns
  for o in o_list:
    if not all(pattern in o.GetName() for pattern in patterns):
      o_list.remove(o)

  # Now get the object(s) taht contain all patterns
  return o_list


def cd_in_file(t_file, directory="/"):

  # Get all sub-directories
  dirs = list(filter(None, os.path.normpath(directory).split(os.sep)))
  # Tree of directories that will be build
  dir_tree = ""
  # Loop over all directories till tree level
  for d in dirs:
    dir_tree = os.path.join(dir_tree, d)
    # Check if this directory exists
    if not t_file.FindKey(dir_tree): t_file.mkdir(dir_tree)
    t_file.cd(dir_tree)
  print("[INFO] Changed directory of file `%s` to `%s`" % (t_file.GetName(), dir_tree))


def add_obj_to_file(obj, t_file, directory="/", verbose=False):

  # Check inputs
  if not isinstance(obj, list): obj = [obj]

  if directory != "/":
    # Get all sub-directories
    dirs = list(filter(None, os.path.normpath(directory).split(os.sep)))
    # Tree of directories that will be build
    dir_tree = ""
    # Loop over all directories till tree level
    for d in dirs:
      dir_tree = os.path.join(dir_tree, d)
      # Check if this directory exists
      if not t_file.FindKey(dir_tree): t_file.mkdir(dir_tree)
      t_file.cd(dir_tree)
  else:
    t_file.cd()

  # Add object to file
  for o in obj:
    o.Write(o.GetName().replace("_clone", ""), ROOT.TObject.kOverwrite)
    if verbose:
      print("[INFO] Added `%s` to file `%s`." % (o.GetName(), fname))

  # Back to root dir
  t_file.cd()


class TFile (ROOT.TFile):

  def __init__(self, fname, write_mode="RECREATE"):

    """
    Simple class derived ROOT.TFile to be used in comb. with a with statement

      Parameters:
        fname (str): Name of the file
        write_mode (str): Write mode of this file [READ, CREATE, RECREATE]
    """

    ROOT.TFile.__init__(self, fname, write_mode)

  def __enter__(self):

      return self

  def __exit__(self, type, value, traceback):

      self.Close()


class FileIo(object):

  """
  This class implements a simple interface to ROOT's TFile
  """

  def __init__(self, out_dir="./", write_mode="update"):

    # Initialize some members
    self.out_dir = out_dir
    self.files = []
    self.current_file = None

    # Check if output directory exists
    if not os.path.exists(self.out_dir):
      print("[INFO] Created directory: %s" % self.out_dir)
      os.makedirs(self.out_dir)


  def addFile(self, fname, write_mode="update", verbose=True):

    # Some checks on input(s)
    if write_mode.lower() not in ["recreate", "create", "update", "new"]:
      warnings.warn("The write mode `%s` is not defined. Using `recreate` instead." % write_mode)
      write_mode = "recreate"
    if not fname.endswith(".root"): fname = fname + ".root"

    # Check if file does not exist, but write_mode is update
    if not os.path.isfile(os.path.join(self.out_dir, fname)) and write_mode.lower() == "update":
      warnings.warn("File does not exist, but write mode `update` has been requested. Create a new file.")
      write_mode = "recreate"

    # Create or load file
    self.files.append(ROOT.TFile(os.path.join(self.out_dir, fname), write_mode.upper()))
   
    # The title will be used as an identifier
    self.files[-1].SetTitle(fname.replace(".root",""))
    if verbose:
      print("[INFO] Added ROOT file: %s" % self.files[-1].GetName())
    self.current_file = self.files[-1]
    if verbose:
      print("[INFO] Current file is: %s" % self.current_file.GetName())


  def addObj(self, obj, fname, directory="/", verbose=False):

    # Check if file name exists
    if not any(fname.replace(".root","") == f.GetTitle() for f in self.files):
      print("[ERROR] The file `%s` has not been defined. Call FileIo.add('%s') to add it." % (fname, fname))
      sys.exit()

    # Get the ROOT file corresponding to `fname`
    t_file = [f for f in self.files if fname.replace(".root", "") == f.GetTitle()][0]

    # Add TObject to file
    add_obj_to_file(obj, t_file, directory)


  def close(self, fname=None):

    # If not file name has been soecified, close all open files
    if not fname:
      for f in self.files:
        f.Close()
        if not f.IsOpen():
          print("[INFO] File `%s` has been closed" % f.GetName())
    else:
      # Get the ROOT file corresponding to `fname`
      t_file = [f for f in self.files if fname == f.GetTitle()][0]
      t_file.Close()
      if not t_file.IsOpen():
        print("[INFO] File `%s` has been closed" % t_file.GetName())


  def flush(self):

    for f in self.files:
      f.Flush()


  def getFile(self, fname):

    # Get the ROOT file corresponding to `fname`
    t_file = [f for f in self.files if fname == f.GetTitle() or fname.replace(".root","") == f.GetTitle()][0]
    return t_file


  def getObj(self, obj_name, fname, directory="/", normalize=False):

    # Check if file name exists
    if not any(fname.replace(".root", "") in f.GetTitle() for f in self.files):
      print("[ERROR] The file `%s` has not been defined. Call FileIo.add('%s') to add it." % (fname, fname))
      sys.exit()

    if not isinstance(obj_name, list):
      obj_name = [obj_name]

    # Get the ROOT file corresponding to `fname`
    t_file = [f for f in self.files if fname.replace(".root", "") == f.GetTitle()][0]

    # Check if this object exist in any directory of the file
    import utils
    r_list = []
    for o_name in obj_name:
      t_o = browse_root_file(t_file, o_name)
      if t_o:
        if normalize: r_list.append(utils.norm_h(t_o))
        else: r_list.append(t_o)
      else: r_list.append(None)
 
    if len(r_list) == 1: return r_list.pop()
    else: return r_list


  def ls(self, fname, pattern=""):

    # Get the ROOT file corresponding to `fname`
    t_file = [f for f in self.files if fname == f.GetTitle()][0]

    print("[INFO] Content of file `%s` that matches pattern `%s`" % (t_file.GetName(), pattern))
    print(t_file.ls(pattern))


  def savePngInRootFile(self, png, fname, directory="/"):

    # Get the ROOT file corresponding to `fname`
    t_file = [f for f in self.files if fname == f.GetTitle()][0]

    # Save PNG on canvas and vectorized version
    t_c = ROOT.TCanvas("classifier_model")
    t_img = ROOT.TImage.Open(fname)
    t_img.Vectorize()
    t_img.Draw("X")
    add_obj_to_file([t_c, t_img], t_file, directory)
    t_c.Delete() # <-- important! Avoid seg fault


  def setCurrentFile(self, fname):

    # Get the ROOT file corresponding to `fname`
    t_file = [f for f in self.files if fname == f.GetTitle()][0]

    # Update current file
    print("[INFO] Current file has been updated: %s -> %s" % (self.current_file.GetName(), t_file.GetName()))
    self.curent_file = t_file


class NtupleIO(ROOT.TFile):

  def __init__(self, filename="output.root", treename="tree", mode="recreate"):

    # Check if the directory exists
    if not os.path.exists(os.path.dirname(os.path.realpath(filename))):
      os.makedirs(os.path.dirname(os.path.realpath(filename)))

    # Initialize ROOT file
    super(NtupleIO, self).__init__(filename, mode)
    if not self.IsOpen():
      print("[ERROR] Something went wrong; ROOT file could not be initialized")
      sys.exit()

    # Initialize TTrees
    self.t_tree = ROOT.TTree(treename, treename)

    # A dictionary to hold default values
    self.defaults = {}

  def __enter__(self):

    return self

  def __exit__(self, exc_type, exc_value, traceback):

    self.cd()
    self.Write()
    self.Close()
    print("[%s] [INFO] Closed file: %s" % (self.__exit__.__name__, self.GetName()))

  def addBranch(self, branchname, obj, leaflist, default=True):

    # Add memeber to class
    setattr(self, branchname, obj)
    # Set branch
    self.t_tree.Branch(branchname, getattr(self, branchname), leaflist)
    if default: self.defaults["_default_"+branchname] = obj[:]

  def _clear(self):

    for branch in self.t_tree.GetListOfBranches():
      if "_default_"+branch.GetName() in self.defaults:
        for i in range(len(self.defaults["_default_"+branch.GetName()])):
          self.__dict__.get(branch.GetName())[i] = self.defaults["_default_"+branch.GetName()][i]

  def addEvent(self, **kwargs):

    for key in kwargs:
     for i in range(len(kwargs[key])):
       if i < len(self.__dict__.get(key)):
         self.__dict__.get(key)[i] = kwargs[key][i]

    # Fill to tree
    self.t_tree.Fill()
    # Reinitialize variables
    self._clear()

