#!/usr/bin/env python2

# System imports
import os, sys
import array

# Scientific imports
import root_numpy
import numpy as np
import ROOT


def _run_from_ROOT (args):

  # Set pT binning
  pt_slice = array.array("d", [350, 450, 550, 650, 750, 850, 950, 1050, 1200, 1350, 1500, 1650, 1800, 1950, 2100, 2300, 2500, 2700,2900, 3150, 3500])

  # Define histogram
  th1_nEventsInBin_sig = ROOT.TH1F("count_sig", "Signal,,l", len(pt_slice)-1, pt_slice)
  th1_nEventsInBin_bkg = ROOT.TH1F("count_bkg", "Background,,l", len(pt_slice)-1, pt_slice)

  from functools import partial
  root2array = partial(root_numpy.root2array, args.input, treename="train", branches=["fjet_training_weight_pt"], stop=-1)

  for i in range(len(pt_slice)-1):

    # Convert ROOT file too array
    selection = "(%s < fjet_truthJet_pt/1000.) && (fjet_truthJet_pt/1000. < %s)" % (pt_slice[i], pt_slice[i+1])
    dat_sig = root2array(selection="%s && (fjet_signal==1)" % selection)
    dat_bkg = root2array(selection="%s && (fjet_signal==0)" % selection)

    # Get arrays
    arr_sig = dat_sig["fjet_training_weight_pt"]
    arr_bkg = dat_bkg["fjet_training_weight_pt"]

    # Fill histo
    th1_nEventsInBin_sig.SetBinContent(i+1, sum(arr_sig))
    th1_nEventsInBin_bkg.SetBinContent(i+1, sum(arr_bkg))

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
    .addText("#bf{Training data set}")
  # Legend
  myplt.settings.legend.x1 = 0.6
  myplt.settings.legend.x2 = 0.9

  # Plot (hopefully) equal pT spectra
  from myplt.figure import Figure
  Figure(canv=(600,500)) \
    .setDrawOption("PMC PLC HIST") \
    .addHist([th1_nEventsInBin_sig, th1_nEventsInBin_bkg]) \
    .addLegend() \
    .addAxisTitle(
      title_x="Transverse momentum p^{truth}_{T} [GeV]",
      title_y="Weighted number of events #sum w_{#lower[-0.5]{train}}") \
    .save("plt/nEventsInPtBins").close()


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


