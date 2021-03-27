#!/usr/bin/env python

# Standard imput
import os, sys
import h5py
sys.path.append("../..")

# Scientific imports
import numpy as np
import tensorflow as tf

# Custom imports
import src.algo

# Some global variables
__inputs__ = ["fjet_clus_pt", "fjet_clus_eta", "fjet_clus_phi", "fjet_clus_E"]
__path2data__ = "/eos/user/c/csauer/data/boostedJetTaggers/constitTopTaggerDNN/training/dataset.train-test-val.antiKt10UFOCSSKSoftDropBeta100Zcut10Jets.nConstit_400.rel220.h5py"
__n_train__, __n_test__ = 5000, 5000
__n_const__ = 50


def smooth(y, box_pts):

  box = np.ones(box_pts)/box_pts
  y_smooth = np.convolve(y, box, mode="same")
  return y_smooth


def list2tgraph(fpr, tpr, inv=True, skip=100):

  import ROOT
  if inv: fpr[fpr!= 0] = np.reciprocal(fpr[fpr!= 0])
  ziped = zip(tpr, fpr)
  ziped.sort()
  tpr, fpr = zip(*ziped)
  tpr, fpr = smooth(tpr,9), smooth(fpr,9)
  ziped = zip(tpr, fpr)
  tG = ROOT.TGraph(len(fpr))
  tG.SetTitle("dsds,,l")
  count = 0
  for i, xy in enumerate(ziped):
    if i % skip != 0: continue
    x, y = xy
    if y < 1: continue
    tG.SetPoint(count, x, y)
    count += 1
  tG.SetPoint(count, 1, 1)
  return tG


def _run ():

  """
    Load data
  """

  with h5py.File(__path2data__, "r") as f:
    # Train and test data
    ds_train = np.zeros((4, __n_train__, __n_const__))
    for i, key in enumerate(__inputs__):
      ds_train[i] = f[os.path.join("train", key)][0:__n_train__,0:__n_const__]
    ds_test = np.zeros((4, __n_test__, __n_const__))
    for i, key in enumerate(__inputs__):
      ds_test[i] = f[os.path.join("val", key)][0:__n_test__,0:__n_const__]

    # Targets
    signal_train, signal_test = f["train/fjet_signal"][0:__n_train__], f["val/fjet_signal"][0:__n_test__]
    # Weights
    weights_train, weights_test = f["train/fjet_weight_train"][0:__n_train__], f["val/fjet_testing_weight_pt"][0:__n_test__]
    weights_test2 = f["train/fjet_testing_weight_pt"][0:__n_test__]
    # Pt
    pt_train, pt_test = f["train/fjet_truthJet_pt"][0:__n_train__]/100000., f["val/fjet_truthJet_pt"][0:__n_test__]/100000.

    print(pt_train)

    # PP
    ds_train[0] /= 700000
    for i in range(__n_train__):
      eta = f["train/fjet_eta"][i]
      for j in range(__n_const__):
        ds_train[1][i][j] -= eta
    for i in range(__n_train__):
      phi = f["train/fjet_phi"][i]
      for j in range(__n_const__):
        ds_train[2][i][j] -= phi
    ds_train[3] /= 700000

    ds_test[0] /= 700000
    for i in range(__n_test__):
      eta = f["val/fjet_eta"][i]
      for j in range(__n_const__):
        ds_test[1][i][j] -= eta
    for i in range(__n_test__):
      phi = f["val/fjet_phi"][i]
      for j in range(__n_const__):
        ds_test[2][i][j] -= phi
    ds_test[3] /= 700000

  # Reshape datasets
  ds_train, ds_test = np.swapaxes(np.swapaxes(ds_train, 0, 2), 0, 1), np.swapaxes(np.swapaxes(ds_test, 0, 2), 0, 1)

  # Make sure the correct number of constituents is used
  ds_train, ds_test = ds_train[:,0:__n_const__,:], ds_test[:,0:__n_const__,:]

  """
     Classification model
  """

  input_co = tf.keras.Input(shape=(__n_const__, 4), name="img")
  input_pt = tf.keras.Input(shape=(1), name="pt")
  x = tf.keras.layers.Flatten(input_shape=ds_train[0].shape)(input_co)
  x = tf.keras.layers.concatenate([x, input_pt])
  x = tf.keras.layers.Dense(20, activation="relu")(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  x = tf.keras.layers.Dense(10, activation="relu")(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
  m_clf = tf.keras.Model([input_co , input_pt], [x])

  # Print and compile model
  m_clf.summary()
  m_clf.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

  """
    Train model
  """

  m_clf.fit([ds_train, pt_train], signal_train, sample_weight=None, epochs=100, shuffle=True, batch_size=100)

  """
    Evaluate
  """

  # Get prediction and save to file
  pred = m_clf.predict([ds_test, pt_test]).flatten()



  import sklearn.metrics
#  fpr, tpr, thresholds = sklearn.metrics.roc_curve(np.nan_to_num(signal_test), np.nan_to_num(pred), sample_weight=weights_test)
  fpr, tpr, thresholds = sklearn.metrics.roc_curve(np.nan_to_num(signal_test), np.nan_to_num(pred), sample_weight=weights_test)
  t_g = list2tgraph(fpr, tpr)
  t_g.SetTitle("%s,,l" % ds_test.shape[1])
  t_g.SetName("roc")

  """
    Start plotting
  """

  import myplt as myplt
  import myplt.settings
  myplt.settings.file.extension = ["pdf", "png"]
  # Some global plotting settings (this will be visible on all plots)
  myplt.settings.file.extension = ["pdf", "png"]
  myplt.settings.textframe.dx = 0.1
  myplt.settings.common_head() \
    .addRow(text="#sqrt{s}=14 TeV, |#eta|<2.0, p_{T} #in [550, 650] (GeV)").setTextAlign("left")
  myplt.settings.common_text() \
    .addText('#bf{#scale[1.3]{DNN-based classifier}}') \

  from myplt.figure import Figure
  # ROC curve
  Figure(canv=(600,500)) \
    .setColorPalette(87, invert=True) \
    .setDrawOption("AL PMC PLC") \
    .addGraph(t_g) \
    .setLogY(range_y=(1,1E3)) \
    .addAxisTitle(title_x="Signal efficiency #varepsilon_{sig}", title_y="Background rejection 1/#varepsilon_{bkg}", title_z="") \
    .save("out/constit/plt/roc") \
    .dumpToROOT("%s/constit/out.root" % "out") \
    .close()

  # DNN score
  Figure(canv=(600,500)) \
    .setColorPalette(87, invert=True) \
    .setDrawOption("PMC PLC HIST") \
    .hist1DFromArray(pred[signal_test==1], nbins=50, range_x=(0, 1), weights=[]) \
    .hist1DFromArray(pred[signal_test==0], nbins=50, range_x=(0, 1), weights=[]) \
    .addAxisTitle(title_x="DNN classification score") \
    .setLogY(range_y=(1E-0, 10E3)) \
    .save("out/constit/plt/score_unweighted") \
    .close()

  # DNN score
  Figure(canv=(600,500)) \
    .setColorPalette(87, invert=True) \
    .setDrawOption("PMC PLC HIST") \
    .hist1DFromArray(pred[signal_test==1], nbins=50, range_x=(0, 1), weights=weights_test[signal_test==1]) \
    .hist1DFromArray(pred[signal_test==0], nbins=50, range_x=(0, 1), weights=weights_test[signal_test==0]) \
    .addAxisTitle(title_x="DNN classification score") \
    .setLogY(range_y=(1E-0, 10E3)) \
    .save("out/constit/plt/score_weighted") \
    .close()

#  # Mass
#  M = np.array([src.algo.constit2jet(jet).M()/1000. for jet in ds_test])
#  M_sig, M_bkg = M[signal_test==1], M[signal_test==0]
#  Figure(canv=(600,500)) \
#    .setColorPalette(87, invert=True) \
#    .setDrawOption("PMC PLC HIST") \
#    .hist1DFromArray(M_sig, nbins=50, range_x=(0, 500)) \
#    .hist1DFromArray(M_bkg, nbins=50, range_x=(0, 500)) \
#    .addAxisTitle(title_x="Reconstructed jet mass m") \
#    .save("out/plt/mass") \
#    .dumpToROOT("%s/out.root" % "out") \
#    .close()
#
##    .setLogY(range_y=(1E-3, 1E4)) \
#
  """
    Save predicted data
  """

  with h5py.File("out/data.h5", "w") as f:
    f.create_dataset("dnn_score", data=pred)
    f.create_dataset("input", data=ds_test)
    f.create_dataset("signal", data=signal_test)



if __name__ == "__main__":

  _run()
