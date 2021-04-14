#!/usr/bin/env python2

# System imports
import os, sys
import json
import ast

# Scientific imports
import numpy as np
import ROOT

# ML related imports
import tensorflow as tf

# Visualization imports
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Custom imports
import common
import myplt


def _run ():

  args = common.get_args()

  """
    Load the model
  """

  # Load the model
  from energyflow.archs import PFN, EFN
#  with open(os.path.join(args.input, "conf/model.json"), "r") as fJSON:
#    model = tf.keras.models.model_from_json(fJSON.read(), {"EFN":EFN})
#    model.load_weights(os.path.join(args.input, "conf/weights.h5"))
#    model.summary()

  Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)
  model = EFN(input_dim=2, Phi_sizes=Phi_sizes, F_sizes=F_sizes)
  model.load_weights(os.path.join(args.input, "conf/weights.h5"))

  # plot settings
  R, n = 0.4, 100
  colors = ['Reds', 'Oranges', 'Greens', 'Blues', 'Purples', 'Greys']
  grads = np.linspace(0.45, 0.55, 4)

  # evaluate filters
  X, Y, Z = model.eval_filters(R, n=n)

  fig, axes = plt.subplots(1, 1, figsize=(8,8))

  # plot filters
  for i,z in enumerate(Z):
      axes.contourf(X, Y, z/np.max(z), grads, cmap=colors[i%len(colors)])

  axes.set_xticks(np.linspace(-R, R, 5))
  axes.set_yticks(np.linspace(-R, R, 5))
  axes.set_xticklabels(['-R', '-R/2', '0', 'R/2', 'R'])
  axes.set_yticklabels(['-R', '-R/2', '0', 'R/2', 'R'])
  axes.set_xlabel('Translated Rapidity y')
  axes.set_ylabel('Translated Azimuthal Angle phi')
  axes.set_title('Energy Flow Network Latent Space', fontdict={'fontsize': 10})

  plt.savefig("contour_trained.pdf")


if __name__ == "__main__":

  _run()

