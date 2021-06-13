import os, sys
import os.path
import ROOT


def add_branch(name, function, data, fill_value=-1, usemask=False):

  # If the branch already exists, do nothing
  if name in data.dtype.names: return data

  import numpy as np
  from numpy.lib.recfunctions import append_fields
  new_branch = function(data)
  data = append_fields(data, name, new_branch, fill_value=fill_value, usemask=usemask)
  return data


def add_field(data_old, name, data_new, fill_value=-1, usemask=False):

  # If the branch already exists, do nothing
  if name in data_old.dtype.names: return data_old

  import numpy as np
  from numpy.lib.recfunctions import append_fields
  data_old = append_fields(data_old, name, data_new, fill_value=fill_value, usemask=usemask)
  return data_old


def drop_field(data, name):

   # Some checks on input
   if not isinstance(name, list): name = [name]

   import numpy as np
   import numpy.lib.recfunctions

   for field in name:
     if field in data.dtype.names:
       data = np.lib.recfunctions.drop_fields(data, field)

   return data


def get_std_name(branch, _range, selection, delimiter=";"):

  if "TH1DModel" in type(_range).__name__:
    _range = "%d,%s,%s" % (_range.fNbinsX, _range.fXLow, _range.fXUp)
  elif "TH2DModel" in type(_range).__name__:
    _range = "%d,%s,%s,%d,%s,%s" % (_range.fNbinsX, _range.fXLow, _range.fXUp,
      _range.fNbinsY, _range.fYLow, _range.fYUp)

  return "%s(%s)%s%s" % (branch.replace("/","|"), _range, delimiter, selection.replace("/","|"))


def get_h_name(branch_exp, t_h_model, weight="", cut=""):

  # Standardize expressions
  weight_std = standardize_exp(weight, "*", rtype=str)
  cut_std = standardize_exp(cut, "&&", rtype=list)
  if weight_std == "": weight_std = "1"
  # Set histogram name
  selection = get_selection(weight_std, cut_std)
  return get_std_name(branch_exp, t_h_model, selection)


def get_tH1_model(name="", title="", nbins=100, xmin=-1, xmax=1):

  return ROOT.ROOT.RDF.TH1DModel(name, title, nbins, xmin, xmax)


def get_tH2_model(name="", title="", nbins_x=100, xmin=-1, xmax=1, nbins_y=100, ymin=-1, ymax=1):

  return ROOT.ROOT.RDF.TH2DModel(name, title, nbins_x, xmin, xmax, nbins_y, ymin, ymax)


def norm_h(hist, norm=1.0):

  if "TH1" in hist.ClassName(): integral = hist.Integral(0, hist.GetNbinsX() + 1)
  elif "TH2" in hist.ClassName(): integral = hist.Integral(0, hist.GetNbinsX() + 1, 0, hist.GetNbinsY() + 1)
  h_tmp = hist.Clone()
  # Normalize
  if float(integral) > 10E-05:
    h_tmp.Scale(norm/float(integral))
  else:
    print("[WARNING] Area under the histogram below 10E-05. The histogram will not be normalized to avoid `division by zero`")
  return h_tmp


def count_entries (path2file, tree_name):

  from fio import TFile 
  n = 0
  with TFile(path2file, write_mode="READ") as f:
    n = f.Get(tree_name).GetEntries()
  return n
