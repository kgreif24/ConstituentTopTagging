#!/usr/bin/env python2

# System imports
import os, sys
import json
import ast

# Scientific imports
import numpy as np
import ROOT

# Custom imports
import common
import myplt


def _run ():

  args = common.get_args()
  entries = args.entries
  if len(entries) == 0: entries = range(len(args.inputs))

  # Dict for TGraphs
  graphs = {}
  for path2file, entry in zip(args.inputs, entries):
    with open(path2file) as fJSON:
      data = ast.literal_eval(json.load(fJSON))
      if not graphs: graphs = {key:[] for key in data}
      for key in data:
        loss = data[key]
        graphs[key].append(ROOT.TGraph(len(loss)))
        for i in range(len(loss)): graphs[key][-1].SetPoint(i, i+1, loss[i])
        graphs[key][-1].SetTitle("%s,,l" % entry)

  """
    Start plotting
  """

  # Some global plotting settings (this will be visible on all plots)
  import myplt.settings
  myplt.settings.file.extension = ["pdf", "png"]
  myplt.settings.textframe.dx = 0.1
  myplt.settings.common_head() \
    .addRow(text="#sqrt{s}=14 TeV, |#eta|<2.0, p_{T} #in [550, 650] (GeV), calo imgs 40#times40") \
    .setTextAlign("left")
  text = myplt.settings.common_text()
  for txt in args.text: text = text.addText(txt) 

  from myplt.figure import Figure
  for key in graphs:
    Figure(canv=(700,500)) \
      .setColorPalette(87, invert=True) \
      .setDrawOption("AL PMC PLC") \
      .addGraph(graphs[key]) \
      .setLogY() \
      .addAxisTitle(title_x="Number of epochs", title_y="Loss (%s)" % key, title_z=args.title_z) \
      .addLegend(xmin=0.8, ymin=0.12, xmax=1.13, ymax=0.925) \
      .setPadMargin(margin=0.2, side="right") \
      .save("%s/loss_family.%s" % (args.outdir, key)) \
      .close()


if __name__ == "__main__":

  _run()
