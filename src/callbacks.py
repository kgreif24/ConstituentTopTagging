# Standard imports
import os, sys

# Scientific imports
import sklearn.metrics
import tensorflow as tf
import math
import ROOT
import numpy

import myplt.settings
import myplt.figure as myfigure

import src.common

#
#def plot_metric (metric, outfile, x_title="Number of epochs", y_title="", name="", path2root=None):
#
#   # Graph
#   tG = ROOT.TGraph(len(metric))
#   for i, m in enumerate(metric):
#     if m is None: return
#     tG.SetPoint(i, i+1, m)
#   tG.SetName(name)
#
#   # Silence ROOT (https://root.cern.ch/root/roottalk/roottalk10/1007.html)
#   fig = myfigure.Figure().addGraph(tG, title_x=x_title, title_y=y_title).save(outfile)
#   if path2root is not None:
#     fig.dumpToROOT(fname=path2root, verbose=False)
#   fig.close()
#
#
#class WeightsSaver (keras.callbacks.Callback):
#
#  def __init__(self, frequency=10, outdir="./"):
#
#    self.freq = frequency
#    self.outdir = outdir
#
#  def on_epoch_end (self, epoch, logs={}):
#
#    if epoch % self.freq == 0 or epoch == 0:
#      name = os.path.join(self.outdir, "weight.ep_%08d.h5") % (epoch+1)
#      self.model.save_weights(name)
#    epoch += 1
#
#  def on_train_end (self, logs={}):
#
#    name = os.path.join(self.outdir, "final.h5")
#    self.model.saveConfig(name)
#
#
#class Metrics (keras.callbacks.Callback):
#
#  def __init__ (self, dp, frequency=1, outdir="./", path2root=None):
#
#    super(keras.callbacks.Callback, self).__init__()
#    self.dp = dp
#    self.freq = frequency
#    self.outdir = outdir
#    self.path2root = path2root
#
#  def on_train_begin (self, logs={}):
#
#    self.losses = []
#    self.accuracy = []
#    self.auc = []
#
#  def on_epoch_end (self, epoch, logs={}):
#
#    # Get validation data
#    x, y, w = self.dp.getData()
#    # Evaluate metrics
#    # - Loss
#    self.losses.append(logs.get("loss"))
#    # - Accuracy
#    self.accuracy.append(logs.get("accuracy"))
#    # - ROC and AUC
#    p = self.model.predict(x, verbose=0).flatten()
#    fpr, tpr, _ = sklearn.metrics.roc_curve(y, p, sample_weight=w)
#    current = sklearn.metrics.auc(fpr, tpr)
#    self.auc.append(current)
#    # Only plot of requency is fulfilled
#    if (epoch+1) % self.freq == 0 or epoch == 0:
#      # Plot if frquency is matched
#      plot_metric(self.losses, os.path.join(self.outdir, "loss.pdf"), y_title="Loss BCE", path2root=self.path2root, name="loss")
#      plot_metric(self.accuracy, os.path.join(self.outdir, "accuracy.pdf"), y_title="Accuracy", path2root=self.path2root, name="accuracy")
#      plot_metric(self.auc, os.path.join(self.outdir, "auc.pdf"), y_title="AUC", path2root=self.path2root, name="auc")
#
#
#class ROC (keras.callbacks.Callback):
#
#  def __init__ (self, dp, frequency=1, outdir="./", path2root=None):
#
#    super(keras.callbacks.Callback, self).__init__()
#    self.dp = dp
#    self.freq = frequency
#    self.outdir = outdir
#    self.path2root = path2root
#
#  def on_train_begin (self, logs={}):
#
#    self.rocs = []
#
#  def on_epoch_end (self, epoch, logs={}):
#
#    if (epoch+1) % self.freq == 0 or epoch == 0:
#      # Get validation data
#      x, y, w = self.dp.getData()
#      # Get prediction by model
#      p = self.model.predict(x, verbose=0).flatten()
#      # Split into signal and backgound
#      pw_sig = (p[y == 1], w[y == 1])
#      pw_bkg = (p[y == 0], w[y == 0])
#      from myroot.performance import arr2roc
#      self.rocs.append(arr2roc(self, pw_sig, pw_bkg, sig_eff_min=0.1))
#      self.rocs[-1].SetTitle("Ep. %d;l" % (epoch+1))
#      self.rocs[-1].SetName("roc.ep%05d" % (epoch+1))
#      # Plot if frquency is matched
##      ROOT.gErrorIgnoreLevel = 5999
#      fig = myfigure.Figure().addGraph(self.rocs, title_x="Signal efficiency #varepsilon_{sig}", title_y="background efficiency 1/#varepsilon_{bkg}") \
#        .setRangeX(0.1, 1.1).setLogY().addLegend(ymin=0.3, entries=[o.GetTitle() for o in self.rocs]).save(os.path.join(self.outdir, "rocs.pdf"))
#      if self.path2root is not None:
#        fig.dumpToROOT(fname=self.path2root, verbose=False)
#      fig.close()
##      ROOT.gErrorIgnoreLevel = -1


class DnnScore (tf.keras.callbacks.Callback):

  def __init__ (self, test_data, freq=1, outdir="./"):

    super(tf.keras.callbacks.Callback, self).__init__()
    self.test_data = test_data
    self.freq = freq
    self.outdir = outdir

  def on_train_begin (self, logs={}):
    pass

  def on_epoch_end (self, epoch, logs={}):

    if (epoch+1) % self.freq == 0 or epoch == 0:
      x, y, w = next(self.test_data)
      # Get prediction by model
      p = self.model.predict(x, verbose=0)[:,1].flatten()
      # Split into signal and backgound
      pw_sig, w_sig = p[y == 1], w[y == 0]
      pw_bkg, w_bkg = p[y == 0], w[y == 1]
      # Plot if frequency is matched
      myplt.settings.auto.OoMthreshold = 1
      fig = myfigure.Figure() \
        .setDrawOption("PLC PMC HIST") \
        .hist1DFromArray(pw_sig, weights=w_sig, nbins=50, range_x=(0,1), name="dnnScore.sig.ep%05d" % (epoch+1)) \
        .hist1DFromArray(pw_bkg, weights=w_bkg, nbins=50, range_x=(0,1), name="dnnScore.bkg.ep%05d" % (epoch+1)) \
        .addAxisTitle(title_x="DNN classification score") \
        .addLegend(title="Ep. %d;l" % (epoch+1), entries=["Signal", "Background"]) \
        .save(os.path.join(self.outdir, "dnnScore.ep%05d.pdf" % (epoch+1))) \
        .close()
