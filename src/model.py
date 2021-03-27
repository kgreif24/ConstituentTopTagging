import os, sys
import json, pickle
proj_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")

# ML imports
import tensorflow as tf

# Custom imports
import common


# Some global settings
__sec_train__, __sec_valid__ = common.read_conf(os.path.join(proj_dir, "etc/nomen_est_omen.ini"), "modes", ["training", "validation"])


class DNN (object):


  def __init__ (self, model):
    self.model = model


  def pltTrainingMetrics (self, outdir):

    # Get dictionary with loss valies
    loss = self.history.history
    # Dictionary for differnet curves and losses
    import ROOT
    graphs = {}
    for key in loss:
      graphs[key] = ROOT.TGraph(len(loss))
      graphs[key].SetTitle(key)
      for i in range(len(loss[key])):
        graphs[key].SetPoint(i, i+1, loss[key][i])

    from myplt.figure import Figure
    for key in ["loss", "acc"]:
      Figure(canv=(700,500)) \
        .setColorPalette(87, invert=True) \
        .setDrawOption("AL PMC PLC") \
        .addGraph([graphs[name] for name in graphs if key in name]) \
        .addLegend() \
        .addAxisTitle(title_x="Number of epochs", title_y=key) \
        .save("%s/%s" % (outdir, key)) \
        .close()


  def saveDiagram (self, outdir):

    fname = os.path.join(outdir, "%s.png" % self.model.name)
    tf.keras.utils.plot_model(self.model, to_file=fname, show_shapes=True)
    return self


  def saveModel (self, outdir):

    # Add paths to meta data
    self.meta["Path2Arch"] = os.path.join(outdir, "arch.json")
    self.meta["Path2Weights"] = os.path.join(outdir, "weights.h5")
    # Save the architrecture and model weights
    self.model.save_weights(self.meta["Path2Weights"])
    model_json = self.model.to_json()
    with open(self.meta["Path2Arch"], "w") as fJSON:
      fJSON.write(model_json)
    return self


  def saveHistory (self, outdir):

    # Save history of training
    with open(os.path.join(outdir, "history.json"), "wb") as fPICKLE:
      pickle.dump(self.history.history, fPICKLE)
    return self


  def saveMetadata (self, outdir):

    with open(os.path.join(outdir, "metadata.json"), "w") as fJSON:
      json.dump(self.meta, fJSON)
    return self


  def load (self, path2model, path2weight):

    with open(path2model, "r") as fJSON:
      self.model = tf.keras.models.model_from_json(fJSON.read())
      self.model.load_weights(path2weight)


  def train (self, datapipe, n_epochs=100, batch_size=10, n_train=100000, n_valid=100000, shuffle=True):

    # Get meta data from datapipe
    self.meta = datapipe.meta

    # Generator for training and validation
    gen_train = datapipe.gen("train", batch_size, n_train, shuffle=shuffle)
    gen_valid = datapipe.gen("valid", batch_size, n_points=n_valid, shuffle=shuffle)

    from myutils.profile import Profile
    with Profile("Start training"):
      # Train model
      self.history = self.model.fit_generator(
        gen_train(),
        steps_per_epoch=gen_train.n_batches,
        validation_data=gen_valid(),
        validation_steps=gen_valid.n_batches,
        epochs=n_epochs, verbose=1, shuffle=shuffle)
    return self

