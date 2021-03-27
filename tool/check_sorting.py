#!/usr/bin/env python2

# System imports
import os, sys

# Scientific imports
import root_numpy


def _run_from_ROOT (args):

  import ROOT
  # Convert ROOT file too array
  dat = root_numpy.root2array(args.input, treename="train", branches=["fjet_sortClus_pt"], stop=100000)

  # Get arrays
  arr = dat["fjet_sortClus_pt"] / 1000.

  # Determine number of constituents
  n_constit = 100

  # Some global plotting settings
  import myplt.settings
  myplt.settings.file.extension = ["pdf", "png"]
  myplt.settings.common_head() \
    .addRow(text="#sqrt{s}=13 TeV, |#eta|<2.0, p^{truth}_{T} > 350 GeV, Contained (rel. 22)") \
    .setTextAlign("left")
  myplt.settings.common_text() \
    .addExperiment()  \
    .addText("Anti-k_{t} R=1.0 jets") \
    .addText("UFO SD(#beta=1.0, z_{cut}=0.1), CS+SK") \
  # Legend
  myplt.settings.legend.x1 = 0.6
  myplt.settings.legend.x2 = 0.9

  # Fill histogram
  th2 = ROOT.TH2F("nConstitVsPt", "", n_constit, 1, n_constit, 100, 1E1, 1000)
  [th2.Fill(i+1, pt[i]) for pt in arr for i in range(n_constit)]

  # Plot (hopefully) equal pT spectra
  from myplt.figure import Figure
  Figure(canv=(600,500)) \
    .setDrawOption("COLZ") \
    .setColorPalette(103) \
    .addHist(th2) \
    .setLogY().setLogZ() \
    .setPadMargin(margin=0.2, side="right") \
    .addAxisTitle(
      title_x="Position of constituent",
      title_y="Weighted transverse momentum p^{truth}_{T} [GeV]") \
    .save("plt/check_sorting").close()

  for j in range(n_constit-1):

    # Fill histogram
    th2 = ROOT.TH2F("corr_%s" % j, "", 100, 1E1, 1000, 100, 1E1, 1000)
    [th2.Fill(pt1, pt2) for pt1, pt2 in zip(arr[:,j], arr[:,j+1])]

    # Plot (hopefully) equal pT spectra
    from myplt.figure import Figure
    Figure(canv=(600,500)) \
      .setDrawOption("COLZ") \
      .setColorPalette(103) \
      .addHist(th2) \
      .setLogZ() \
      .setPadMargin(margin=0.2, side="right") \
      .addAxisTitle(
        title_x="Transverse momentum constituent p^{%s}_{T} [GeV]" % j,
        title_y="Transverse momentum constituent p^{%s}_{T} [GeV]" % j) \
      .save("plt/corr/pair_%s_%s" % (j,j+1)).close()


def _run_from_HDF5(args):
  import h5py
  pass


if __name__ == "__main__":

  import argparse
  # Get input file from command line
  parser = argparse.ArgumentParser(description="Check if input looks reasonable")
  parser.add_argument("--input", metavar="I", type=str, help="Input file [ROOT]")
  args = parser.parse_args()

  if args.input.endswith(".root"):
    _run_from_ROOT(args)
  elif args.input.endswith(".h5*"):
    _run_from_HDF5(args)


