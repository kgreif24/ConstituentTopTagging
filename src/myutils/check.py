
def is_normalized(t_h, eps=1E-04):

  import ROOT

  integral=1.0
  if "TH1" in t_h.ClassName(): integral = t_h.Integral(1, t_h.GetNbinsX())
  elif "TH2" in t_h.ClassName(): integral = t_h.Integral(1, t_h.GetNbinsX(), 1, t_h.GetNbinsY())
  if abs(integral-1.0)<eps:
    return True
  else:
    return False


def is_file(fname):

  import os

  if os.path.isfile(fname):
    return True
  else:
    return False


def inherits_from(x, base_class):

   if not isinstance(base_class, (tuple, list)): base_class = (base_class)

   for b in base_class:
     if not hasattr(x, "InheritsFrom"):
       return False
     if x.InheritsFrom(b):
       return True
   return False
