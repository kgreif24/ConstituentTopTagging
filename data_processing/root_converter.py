""" root_converter.py - This file defines the RootConverter class, which is
the class responsible for looping over a list of .root dumper output files
and converting them into the h5 file format. A single dataset can
subsequently be built by shuffling the .h5 output files. See README of
data_processing submodule for details.

Author: Kevin Greif
Last updated 6/28/2022
python3
"""

import processing_utils as pu

class RootConverter:
    """ RootConverter -
    """

    def __init__(self, setup_dict):
        """ __init__ - The init function for the class takes in a dictionary
        of parameters for the conversion. See main program below for all of
        the necessary parameters to include
        """

        # First make dictionary an instance variable
        self.params = setup_dict

        # Now calculate derived parameters. Open files list to get names
        listfile = open(self.params['source_list'], "r")
        files = listfile.readlines()
        self.files = [f.rstrip() + self.params['tree_name'] for f in files]

        # Because files come from different pt slices, we need to pull a
        # representative sample of each file for our train/test data sets.
        # Find number of jets we expect to pull from each file
        cb = self.params['cut_branches']
        svb = self.params['svb_flag']
        raw_file_events = [pu.find_cut_len_new(name, cb, svb) for name in files]
        raw_events = np.sum(raw_file_events)
        print("We have", raw_events, "jets in total")
        print("We wish to keep", total_events, "of these jets")



if __name__ == '__main__':

    # Define convert_dict which is passed to RootConverter class
    convert_dict = {
        'svb_flag': True,
        'source_list': './dat/ZprimeTTSamples.list',
        'tree_name': ':FlatSubstructureJetTree',
        'rw_type': 'w',
        'max_constits': 200,
        'target_dir': './dataloc/intermediates_mc/',
        'n_targets': 36,
        'total_events': 22375114,
        's_constit_branches': [
            'fjet_clus_pt', 'fjet_clus_eta',
            'fjet_clus_phi', 'fjet_clus_E'
        ],
        'hl_branches': [
            'fjet_Tau1_wta', 'fjet_Tau2_wta', 'fjet_Tau3_wta',
            'fjet_Tau4_wta', 'fjet_Split12', 'fjet_Split23',
            'fjet_ECF1', 'fjet_ECF2', 'fjet_ECF3', 'fjet_C2',
            'fjet_D2', 'fjet_Qw', 'fjet_L2', 'fjet_L3',
            'fjet_ThrustMaj'
        ],
        't_constit_branches': [
            'fjet_clus_eta', 'fjet_clus_phi', 'fjet_clus_pt', 'fjet_clus_E'
        ],
        'jet_branches': ['fjet_pt', 'fjet_eta', 'fjet_phi', 'fjet_m'],
        'label_name': 'labels',
        'cut_branches': [
            'fatjet_truth_eta', 'fatjet_truth_pt', 'fatjet_numConstituents', 'fatjet_m',
            'fatjet_truth_dRmatched_particle_flavor', 'fatjet_truth_dRmatched_particle_dR',
            'fatjet_truth_dRmatched_particle_dR_top_W_matched', 'fatjet_ungroomed_truth_m',
            'fatjet_truth_ungroomedParent_GhostBHadronsFinalCount', 'fatjet_ungroomed_truth_Split23',
            'fatjet_ungroomed_truth_pt', 'fatjet_ungroomed_truth_ghostNTop',
            'ttbar_deltaR_rapidity'
        ]
    }

    # Build the class
    rc = RootConverter(convert_dict)
