import ROOT
import numpy as np


def poly1D(t_h1, order, option="S", exponent="TMath::Power"):

  if (exponent != "TMath::Power") and (exponent != "^"):
    print("[ERROR] No valide argument for `exponent` was provided [TMath::Power/^]")
    sys.exit()

  # get a string expression depending on the order of the polynomial
  if exponent == "^":
    func_str = "[0]+" + "+".join(["[%s]*x^(%s)" % (i, i) for i in range(1, order+1)])
  else:
    func_str = "[0]+" + "+".join(["[%s]*TMath::Power(x,%s)" % (i, i) for i in range(1, order+1)])
  print("[INFO] Using the following function for fitting: %s" % func_str)

  # Get min and max value
  x_min, x_max = t_h1.GetXaxis().GetXmin(), t_h1.GetXaxis().GetXmax()
  t_f1_fit = ROOT.TF1(func_str, func_str , x_min, x_max)

  # Fit polynomial to histogram in given range
  t_h1.Fit(t_f1_fit, option)

  # return fit
  return t_f1_fit
