
def change_unit(var_name, in_unit, out_unit):

  if "mev" in in_unit.lower() and "gev" in out_unit.lower():
    return "%s/1000." % var_name
  else:
    return var_name


def cut_label(var_name, var_unit, cut_low=None, cut_high=None):

  if cut_low and cut_high:
    return "%s %s < %s < %s %s" % (cut_low, var_unit, var_name, cut_high, var_unit)
  elif cut_low and not cut_high:
    return "%s > %s %s" % (var_name, cut_low, var_unit)
  elif not cut_low and cut_high:
    return "%s < %s %s" % (var_name, cut_high, var_unit)
  else:
    return ""


def set_batch_mode(mode=True):

  import ROOT
  ROOT.gROOT.SetBatch(mode)

