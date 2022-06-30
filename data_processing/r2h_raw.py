""" r2h_raw.py - This script runs the r2h step of data processing, applying
the raw preprocessing. This is the minimal preprocessing used for public
data sets.

Author: Kevin Greif
python3
Last updated 6/30/22
"""

from root_converter import RootConverter
import preprocessing as pp

# Define convert_dict which is passed to RootConverter class
sig_dict = {
    'svb_flag': True,
    'trim': False,
    'source_list': './dat/sig_test.list',
    'rw_type': 'w',
    'cut_branches': [
        'fjet_truthJet_eta', 'fjet_truthJet_pt', 'fjet_numConstituents', 'fjet_m',
        'fjet_truth_dRmatched_particle_flavor', 'fjet_truth_dRmatched_particle_dR',
        'fjet_truthJet_dRmatched_particle_dR_top_W_matched', 'fjet_ungroomed_truthJet_m',
        'fjet_truthJet_ungroomedParent_GhostBHadronsFinalCount', 'fjet_ungroomed_truthJet_Split23',
        'fjet_ungroomed_truthJet_pt'
    ]
}

bkg_dict = {
    'svb_flag': False,
    'trim': True,
    'source_list': './dat/bkg_test.list',
    'rw_type': 'a',
    'cut_branches': [
        'fjet_truthJet_eta', 'fjet_truthJet_pt', 'fjet_numConstituents', 'fjet_m'
    ]
}

stem_dict = {
    'tree_name': ':FlatSubstructureJetTree',
    'max_constits': 200,
    'target_dir': './dataloc/intermediates_test/',
    'n_targets': 4,
    'total': 22375114,
    's_constit_branches': [
        'fjet_clus_pt', 'fjet_clus_eta',
        'fjet_clus_phi', 'fjet_clus_E'
    ],
    'pt_name': 'fjet_clus_pt',
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
}

sig_dict.update(stem_dict)
bkg_dict.update(stem_dict)

# Build the classes
sig_rc = RootConverter(sig_dict)
bkg_rc = RootConverter(bkg_dict)

# Run main programs
sig_rc.run(constit_func=pp.raw_preprocess)
bkg_rc.run(constit_func=pp.raw_preprocess)
