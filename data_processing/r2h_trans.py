""" r2h_trans.py - This script runs the r2h step of data processing, applying
the raw preprocessing. This is particularly meant for converting Delphes
data .root files to the .h5 format.

Author: Kevin Greif
python3
Last updated 7/17/2022
"""

from root_converter import RootConverter
import processing_utils as pu
import preprocessing as pp
import syst_variations as syst

# Define convert_dict which is passed to RootConverter class
convert_dict = {
    'tree_name': ':SubstructureJetTree',
    'trim': True,
    'source_list': './dat/trans_zprime.list',
    'rw_type': 'w',
    'max_constits': 200,
    'target_dir': './dataloc/int_zprime_trans/',
    'n_targets': 10,
    'total': 5000000,
    # 'total': 100000,
    'constit_func': pp.raw_preprocess,
    'syst_func': None,
    'cut_func': pu.trans_cuts,
    's_constit_branches': [
        'fjet_clus_pt', 'fjet_clus_eta',
        'fjet_clus_phi', 'fjet_clus_E'
    ],
    'pt_name': 'fjet_clus_pt',
    'hl_branches': [],
    't_constit_branches': [
        'fjet_clus_eta', 'fjet_clus_phi', 'fjet_clus_pt', 'fjet_clus_E'
    ],
    'jet_branches': ['fjet_pt', 'fjet_eta', 'fjet_phi', 'fjet_m'],
    'cut_branches': [
        'fjet_eta', 'fjet_pt', 'fjet_numConstits', 'fjet_m'
    ]
}

# Build the class
rc = RootConverter(convert_dict)

# Run main programs
rc.run()

