#!/usr/bin/env python2

# Standard imports
import os, sys
import configparser
path2home = os.path.abspath(os.path.dirname(__file__))


"""
 Some global configurations
"""

# Every entry in this dictionary is a command line argument
__conf__ = \
{
  "input"         : True,
  "outdir"        : True,
  "batch-size"    : True,
  "n-train"       : True,
  "n-epoch"       : True,
  "n-constit"     : True,
  "sample-weight" : True,
  "target"        : True,
  "load"          : True,
  "architecture"  : True,
  "Phi-sizes"     : True,
  "F-sizes"       : True
}

def _run ():

  import src.common
  # Command-line arguments pser
  args = src.common.get_parser(**__conf__).parse_args()

  """
    To be displayed on plots
  """

  import myhep.jets, myplt.settings
  myplt.settings.common_head() \
    .addRow(text="DNN top tagger, #bf{Training}") \
    .setTextAlign("left")
  jetInfo = myhep.jets.JetInfo("AntiKt10UFOCSSKSoftDropBeta100Zcut10Jets")
  myplt.settings.common_text() \
    .addExperiment() \
    .addText(jetInfo.getReconstructionAndObject()) \
    .addText(jetInfo.getGroomer()[1])

  """
    Build project directory structure
  """

  import myutils.walker
  orga = myutils.walker.Walker(args.outdir)

  """
    Read and preprocess data
  """

  import src.data
  dp = src.data.DataPipe(args.input, outdir=orga.get("data"))
  # Add features (inputs) used to train the DNN
  if args.architecture == "PFN":
    dp.addFeature(
      ["fjet_sortClusNormByPt_pt",
      "fjet_sortClusCenterRotFlip_eta",
      "fjet_sortClusCenterRot_phi",
      "fjet_sortClusNormByPt_e"]
    )
  elif args.architecture == "RNN":
    dp.addFeature(
      ["fjet_sortClusNormByPt_pt",
      "fjet_sortClusCenterRotFlip_eta",
      "fjet_sortClusCenterRot_phi",
      "fjet_sortClusNormByPt_e"]
    )
  elif args.architecture == "EFN":
    dp.addFeature(
      ["fjet_sortClusNormByPt_pt",
      ["fjet_sortClusCenterRotFlip_eta",
      "fjet_sortClusCenterRot_phi"]]
    )
  if args.architecture == "DNN":
    dp.addFeature(
      ["fjet_sortClusNormByPt_pt",
      "fjet_sortClusCenterRotFlip_eta",
      "fjet_sortClusCenterRot_phi",
      "fjet_sortClusNormByPt_e"],
      flat=True
    )
  # Add target
  dp.setTarget(args.target)
  # Add training weight
  dp.setWeight(args.sample_weight)
  # Set number of constituents
  dp.setNConstit(args.n_constit)
  # Print config
  dp.display()

  """
    Initialize the DNN-based model and train it on data
  """

  import src.model
  import tensorflow as tf
  opt = tf.keras.optimizers.Adam(learning_rate=0.001)
  if args.architecture == "PFN":
    from energyflow.archs import PFN, EFN
    dnn = PFN(input_dim=4, Phi_sizes=tuple(args.Phi_sizes), F_sizes=tuple(args.F_sizes), summary=(0==0), optimizer=opt)
  elif args.architecture == "EFN":
    from energyflow.archs import EFN
    dnn = EFN(input_dim=2, Phi_sizes=tuple(args.Phi_sizes), F_sizes=tuple(args.F_sizes), summary=(0==0), optimizer=opt)
  elif args.architecture == "DNN":
    dnn = tf.keras.models.Sequential()
    dnn.add(tf.keras.layers.Dense(200, input_dim=4*args.n_constit, activation='relu'))
    dnn.add(tf.keras.layers.BatchNormalization())
    dnn.add(tf.keras.layers.Dense(50, activation='relu'))
    dnn.add(tf.keras.layers.BatchNormalization())
    dnn.add(tf.keras.layers.Dense(10, activation='relu'))
    dnn.add(tf.keras.layers.BatchNormalization())
    dnn.add(tf.keras.layers.Dense(2, activation='softmax'))
    dnn.summary()
    # Compile model
  elif args.architecture == "RNN":
    dnn = tf.keras.models.Sequential()
    dnn.add(tf.keras.layers.Masking(mask_value=0.0, input_shape=(args.n_constit, 4)))
    dnn.add(tf.keras.layers.LSTM(25, return_sequences=False))
    dnn.add(tf.keras.layers.Dense(2, activation='softmax'))
    dnn.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=["accuracy"])


  model = src.model.DNN(dnn)
  if args.load:
    model.load(orga.get("conf/arch.json"), orga.get("conf/weights.h5"))
  else:
    model.saveDiagram(orga.get("fig")) \
      .train(dp, n_train=args.n_train, n_epochs=args.n_epoch, batch_size=args.batch_size) \
       .saveModel(orga.get("conf")) \
       .saveMetadata(orga.get("conf")) \
       .saveHistory(orga.get("data")) \
       .pltTrainingMetrics(orga.get("fig"))


  """
    Save the metadata for this model
  """

  import json, ROOT
  with open(orga.get("conf", "metadata.json"), "w") as fJSON:
    fROOT = ROOT.TFile(args.input, "READ")
    meta = {obj.GetName():obj.GetTitle() for obj in fROOT.Get("train").GetUserInfo()}
    meta["NConstit"] = args.n_constit
    meta["Architecture"] = args.architecture
    json.dump(meta, fJSON)
    fROOT.Close()

  """
    Do a quick evaluation of the trained model
  """

  # Add some other columns needed for evaluation
  dp.addOther("fjet_truthJet_pt")
  dp.addOther("fjet_testing_weight_pt")
  import src.eval
  evl = src.eval.EvalQuick(dp, model.model, orga.get("home"))
  evl.setPtBins([350, 450, 550, 650, 750, 850, 950, 1050, 1200, 1350, 1500, 1650, 1800, 1950, 2100, 2300, 2500, 2700,2900, 3150, 3500]) \
     .cutOnScore(0.5)

#     .efnLatentSpace(dnn) \

if __name__ == "__main__":

  _run()
