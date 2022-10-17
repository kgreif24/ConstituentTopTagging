""" r2h_raw.py - This script runs the r2h step of data processing, applying
the raw preprocessing. This is the minimal preprocessing used for public
data sets.

Author: Kevin Greif
python3
Last updated 7/5/22
"""

from root_converter import RootConverter
import processing_utils as pu
import preprocessing as pp
import syst_variations as syst

# Define convert_dict which is passed to RootConverter class
convert_dict = {
    'tree_name': ':FlatSubstructureJetTree',
    'trim': True,
    'flatten': False,
    'source_list': './dat/dijet_data.list',
    'max_constits': 200,
    'target_dir': './dataloc/int_dijet_s2_nominal/',
    'name_stem': 'dijet_',
    'n_targets': 50,
    'total': 20000000,
    'constit_func': pp.raw_preprocess,
    'logs': True,
    'syst_func': None,
    'syst_loc': 'DFS-L/DATA/whiteson/kgreif/ModTaggingData/cluster_uncert_map_EM.root',
    'cut_func': pu.common_cuts,
    'jet_func': None,
    's_constit_branches': [
        'fjet_clus_pt', 'fjet_clus_eta',
        'fjet_clus_phi', 'fjet_clus_E',
        'fjet_clus_taste'
    ],
    's_jet_branches': ['fjet_pt', 'fjet_eta', 'fjet_phi', 'fjet_m'],
    'pt_name': 'fjet_clus_pt',
    'hl_branches': [],
    'event_branches': [],
    't_constit_branches': [
        'fjet_clus_eta', 'fjet_clus_phi', 'fjet_clus_pt', 'fjet_clus_E'
    ],
    't_onehot_branches': ['fjet_clus_taste'],
    't_onehot_classes': [3],
    't_jet_branches': ['fjet_pt', 'fjet_eta', 'fjet_phi', 'fjet_m'],
    'cut_branches': [
        'fjet_truthJet_eta', 'fjet_truthJet_pt', 'fjet_numConstituents', 'fjet_m',
        'fjet_truth_dRmatched_particle_flavor', 'fjet_truth_dRmatched_particle_dR',
        'fjet_truthJet_dRmatched_particle_dR_top_W_matched', 'fjet_ungroomed_truthJet_m',
        'fjet_truthJet_ungroomedParent_GhostBHadronsFinalCount', 'fjet_ungroomed_truthJet_Split23',
        'fjet_ungroomed_truthJet_pt'
    ],
    'mask_lim': 0.1
}

# Build the class
rc = RootConverter(convert_dict)

# Run main programs
rc.run()

