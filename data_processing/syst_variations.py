""" syst_variations.py - This script defines function which apply systematic
variations to the constituent level inputs. These functions can be passed to
the root converter class to produce data set with applied systematics.

Currently, the variations implemented are the EM scale cluster uncertainties.

For now, we hard code the names of constituent level branches in the jet
batches. (e.g. 'fjet_clus_E').

Author: Kevin Greif
Last updated 7/1/22
python3
"""

import awkward as ak
import numpy as np

def reco_efficiency(jets, uncert_map, constit_branches):
    """ reco_efficiency - This function applies the cluster reconstruction
    efficiency systematic variation to the constituent level inputs contained
    in the jet batch.

    Arguments:
    jets (dict): The jet batch, almost always as defined in the root converter
    class after cuts have been applied. See root_converter.py for details.
    uncert_map (TFile): The uncertaintiy map file object loaded using PyROOT
    constit_branches (list): The names of the constituent branches to apply 
    variation on.

    Returns:
    (dict): A dictionary containing the constituent level quantities with
    applies systematic variation. Keys are identical to those given as input.
    """

    # Get cluster scale histogram from uncert map
    cluster_scale = uncert_map.Get('Scale')

    # Convert energy to GeV
    en = jets['fjet_clus_E'] / 1000

    # To use PyROOT we need to write a loop over the individual jet constituents
    # This will require nested for loops.
    # We will write boolean values to an awkward array using ak.ArrayBuilder.
    # This array can then be used to index the constituent arrays and drop
    # the unwanted constituents
    n_jets = len(en)
    n_constits = ak.count(en, axis=1)

    # Initialize builder
    builder = ak.ArrayBuilder()

    # Jet loop
    for i in range(n_jets):

        # Add list to builder
        builder.begin_list()

        # Constituent loop
        for j in range(n_constits[i]):

            # Get constituents energy, eta, and taste
            cons_en = en[i,j]
            cons_eta = abs(jets['fjet_clus_eta'][i,j])
            cons_taste = jets['fjet_clus_taste'][i,j]

            # If constituent is not neutral (taste == 1), immediately write
            # True and skip to next
            if cons_taste != 1:
                builder.append(True)
                continue

            # Get energy and eta bins
            Ebin = cluster_scale.GetXaxis().FindBin(cons_en)
            ebin = cluster_scale.GetYaxis().FindBin(cons_eta)

            # Correct overflows
            if (Ebin > cluster_scale.GetNbinsX()):
                Ebin = cluster_scale.GetNbinsX()
            elif (Ebin < 1):
                Ebin = 1

            if (ebin > cluster_scale.GetNbinsY()):
                ebin = cluster_scale.GetNbinsY()
            elif (ebin < 1):
                ebin = 1

            # If we have bin content, divide cluster energy by scale
            if (cluster_scale.GetBinContent(Ebin, ebin) > 0):
                p = cons_en / cluster_scale.GetBinContent(Ebin, ebin)

            # Now find r, depending on the value of eta
            if (cons_eta < 0.6):
                r = (0.12*np.exp(-0.51*p) + 4.76*np.exp(-0.29*p*p)) / 100
            elif (cons_eta < 1.1):
                r = (0.17*np.exp(-1.31*p) + 4.33*np.exp(-0.23*p*p)) / 100
            elif (cons_eta < 1.4):
                r = (0.17*np.exp(-0.95*p) + 1.14*np.exp(-0.04*p*p)) / 100
            elif (cons_eta < 1.5):
                # This one is a crummy fit, see twiki
                r = (0.15*np.exp(-1.14*p) + 2768.98*np.exp(-4.2*p*p)) / 100
            elif (cons_eta < 1.8):
                r = (0.16*np.exp(-2.77*p) + 0.67*np.exp(-0.11*p*p)) / 100
            elif (cons_eta < 1.9):
                r = (0.16*np.exp(-1.47*p) + 0.86*np.exp(-0.12*p*p)) / 100
            else:
                r = (0.16*np.exp(-1.61*p) + 4.99*np.exp(-0.52*p*p)) / 100

            # Get random number
            rng = np.random.default_rng()
            flip = rng.uniform()

            # Accept or reject constituent
            print("\np:", p)
            print("r:", r)
            print("flip:", flip)
            if ((flip < r) and (cons_en / 1000 < 2.5)):
                builder.append(False)
                print("Dropped")
            else:
                builder.append(True)
                print("Kept")

        # End constituent loop
        # Close builder list
        builder.end_list()

    # End jet loop
    # Get awkward array from builder
    keep = builder.snapshot()

    # Index all constituent level branches, dropping the required constituents
    var_dict = {kw: jets[kw][keep] for kw in constit_branches}

    return var_dict
