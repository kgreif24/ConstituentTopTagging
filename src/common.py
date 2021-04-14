import os, sys
proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, proj_dir)


# Some global variables
__default_outdir__ = "OUTPUT"


def read_conf (path2file, section=None, name=None):

  import configparser
  conf = configparser.ConfigParser(allow_no_value=True)
  conf.read(path2file)

  if section is None and name is None:
    return conf.keys()

  if not isinstance(path2file, list):
    path2file = [path2file]
  if not isinstance(section, list) and not isinstance(name, list):
    section, name = [section], [name]
  elif not isinstance(section, list) and isinstance(name, list):
    section = [section] * len(name)
  elif isinstance(section, list) and not isinstance(name, list):
    name = [name] * len(section)

  # We return a tuple in the form (1,2,3, ...)
  return_tuple = []
  for s, n in zip(section, name):
    return_tuple.append(conf[s][n].encode("ascii","ignore"))

  if len(return_tuple) > 1:
    return tuple(return_tuple)
  return return_tuple[0]


# Help calls
help_truth_label = "Definition of the signal to be used (default: %(default)s)."
help_input_directory = "Directory where input files are stored"
help_input = "Path to project."
help_inputs = "Path to projects."
help_features = "Training features to the DNN."
help_input_sig = "Input directory from which to read input ROOT files with signal samples."
help_input_bkg = "Input directory from which to read input ROOT files with background samples."
help_n_events = "Total size of dataset"
help_n_train = "Size of training dataset"
help_n_test = "Size of testing dataset"
help_objective ="Which function to minimize"
help_suffix = "Suffix appended to final file name. (default: %(default)s)"
help_suffix_train = "Add a suffix to training section in model name"
help_suffix_data = "Add a suffix to data section in model name"
help_suffix_model = "Add a suffix to model section in model name"
help_batch_size = "Batch size: Set the number of jets to look before updating the weights of the NN (default: %(default)s)."
help_n_epoch = "Set the number of epochs to train over (default: %(default)s)."
help_seed_nr = "Seed initialization (default: %(default)s)."
help_learning_rate = "Learning rate used by the optimizer (default: %(default)s)."
help_adam_beta1 = "Beta1 used by the Adam optimizer (default: %(default)s)."
help_adam_beta2 = "Beta2 used by the Adam optimizer (default: %(default)s)."
help_optimizer = "optimizer for training (default: %(default)s)."
help_objective = "objective (or loss) function for training (default: %(default)s)."
help_preprocc_type = "Different pre-processing options (default: %(default)s)."
help_jssv_group = "Input group of JSSV %(default)"
help_clipnorm="Clipnorm for gradient clipping (default: %(default)s)."
help_node_config = "Architecture of the NN, options for different nodes per layer (default: %(default)s)."
help_sample_weight = "Use sample weights for training (default: %(default)s)?"
help_target = "Target the model is supposed to predict: %(default)s)?"
help_jc_config = "Configuration file with definition of different jet sub structure variables (default: %(default)s)."
help_r_l1 = "l1 weight regularization penalty (default: %(default)s)."
help_r_l2 = "l2 weight regularization penalty (default: %(default)s)."
help_model_config = "Architecture of the NN, options for different nodes per layer (default: %(default)s)."
help_outdir = "Output directory, to which to write output files (default: %(default)s)."
help_fout = "Name of output file (default: %(default)s)."
help_clip_sample_weight = "Clip upper value of sample weights (default: %(default)s)."
help_init_distr = "Weight initialization distribution determines how the initial weights are set (default: %(default)s)."
help_LRS = "Use Learning Rate Scheduler (False)"
help_activation_function = "Activation function (default: %(default)s)."
help_jssv_config = "Configuration file with definition of different jet sub structure variables (default: %(default)s)."
help_batch_norm = "Normalize data between two layers (default: %(default)s)"
help_architecture = "Which architecture to use for the model"
help_slim_config = "Configuration file with slimming conditions (default: %(default)s)."
help_start = "Start index of array slicing (default: %(default)s)."
help_step = "Step size in array slicing (default: %(default)s)."
help_stop = "Stop index in array slicing (default: %(default)s)."
help_flat_pt = "Pt variable that is suposed to be flat during training (default: %(default)s)."
help_max_processes = "Maximum number of concurrent processes to use (default: %(default)s)."
help_entries = "Entries in plt's legend (default: %(default)s)."
help_txt_top = "Text to be displayed on top of plots (default: %(default)s)."
help_txt_pad = "Text to be displayed in pad (default: %(default)s)."
help_poly_order = "Order of the polynomial (default: %(default)s)."
help_config = "Path to configuration file of the model."
help_working_point = "Working point (default: %(default)s)."
help_n_constit = "Number of constituents (default: %(default)s)."

# Defaults
def_truth_label = "rel22.0"
def_objective = "binary_crossentropy"
def_suffix_data = ""
def_suffix_train = ""
def_suffix_model = ""
def_learning_rate = 0.0005
def_optimizer = "adam"
def_preprocc_type = "standard"
def_adam_beta1 = 0.9
def_adam_beta2 = 0.999
def_n_epoch = 200
def_sed_nr = 12264
def_batch_size = 128
def_jssv_group = 11
def_clipnorm = 0
def_node_config = "nNodesWrtFeatures1"
def_sample_weight = "fjet_training_weight_pt"
def_target = "fjet_signal"
def_jc_config = os.path.join(proj_dir, "..", "etc/jc.ini")
def_r_l1 = 0.001
def_r_l2 = 0.001
def_model_config = "5Dense"
def_outdir = __default_outdir__
def_fout = "out"
def_clip_sample_weight = 0
def_init_distr = "glorot_uniform"
def_activation_function = "relu"
def_jssv_config = os.path.join(proj_dir, "..", "etc/jssv.ini")
def_batch_norm = 1
def_architecture = "PFN"
def_suffix = ""
def_slim_config = os.path.join(proj_dir, "..", "etc/trim_slim.ini")
def_start = 0
def_step = 3
def_stop = -1
def_flat_pt = "fjet_truthJet_pt"
def_max_processes = 5
def_entries = []
def_txt_top = []
def_txt_pad = []
def_poly_order = 7
def_working_point = 50
def_n_constit = 200

# Choices
choices_objective = ["binary_crossentropy" , "categorical_crossentropy", "mse"]
choices_optimizer = ["adam", "adamax", "rmsprop"]
choices_preprocc_type = ["min_max", "standard"]
choices_node_config = ["nNodesWrtFeatures1", "nNodesWrtFeatures2", "nNodesWrtFeatures3"]
choices_model_config = ["1Dense", "2Dense", "3Dense", "4Dense", "5Dense", "6Dense", "8Dense"]
choices_activation_function = ["relu", "tanh", "ELU", "PReLU", "SReLU"]
choices_batch_norm = [0, 1]
choices_architecture = ["PFN", "EFN", "DNN", "RNN"]
choices_truth_label = read_conf(os.path.join(proj_dir, "..", "etc/selection.ini"))
choices_flat_pt = ["fjet_truthJet_pt", "fjet_pt"]

# List of possible arguments
arguments = {
  "input": \
    dict(type=str, required=True,
    help=help_inputs),
  "inputs": \
    dict(type=str, nargs="+", required=True,
    help=help_input),
  "features": \
    dict(type=str, nargs="+", required=True,
    help=help_features),
  "input-sig": \
    dict(required=True,
    help=help_input_sig),
  "input-bkg": \
    dict(required=True,
    help=help_input_bkg),
  "outdir": \
    dict(type=str, default=def_outdir,
    help=help_outdir),
  "fout": \
    dict(type=str, default=def_fout,
    help=help_fout),
  "truth-label": \
    dict(type=str, default=def_truth_label,
    help=help_truth_label),
  "n-events": \
    dict(type=int, required=True,
    help=help_n_events),
  "n-train": \
    dict(type=int, required=True,
    help=help_n_train),
  "n-test": \
    dict(type=int, required=True,
    help=help_n_test),
  "objective": \
    dict(type=str, default=def_objective,
    choices=choices_objective,
    help=help_objective),
  "optimizer": \
    dict(type=str, default=def_optimizer,
    choices=choices_optimizer,
    help=help_optimizer),
  "suffix": \
    dict(type=str, default=def_suffix,
    help=help_suffix),
  "suffix-data": \
    dict(type=str, default=def_suffix_data,
    help=help_suffix_train),
  "suffix-model": \
    dict(type=str, default=def_suffix_model,
    help=help_suffix_model),
  "suffix-train": \
    dict(type=str, default=def_suffix_train,
    help=help_suffix_data),
  "learning-rate": \
    dict(type=float, default = def_learning_rate,
    help=help_learning_rate),
  "adam-beta1": \
    dict(type=float, default=def_adam_beta1,
    help=help_adam_beta1),
  "adam-beta2": \
    dict(type=float, default = def_adam_beta2,
    help=help_adam_beta2),
  "n-constit": \
    dict(type=int, default=def_n_constit,
    help=help_n_constit),
  "n-epoch": \
    dict(type=int, default=def_n_epoch,
    help=help_n_epoch),
  "seed-nr": \
    dict(type=int, default=def_sed_nr,
    help=help_seed_nr),
  "batch-size": \
    dict(type=int, default=def_batch_size,
    help=help_batch_size),
  "preprocc-type-option": \
    dict(type=str, default=def_preprocc_type,
    choices=choices_preprocc_type,
    help=help_preprocc_type),
  "jssv-group": \
    dict(type=int, default=def_jssv_group,
    help=help_jssv_group),
  "clipnorm": \
    dict(type=float, default=def_clipnorm,
    help=help_clipnorm),
  "node-config": \
    dict(type=str, default=def_node_config,
    choices=choices_node_config,
    help=help_node_config),
  "sample-weight": \
    dict(type=str, default=def_sample_weight,
    help=help_sample_weight),
  "target": \
    dict(type=str, default=def_target,
    help=help_target),
  "jc-config": \
    dict(type=str, default=def_jc_config,
    help=help_jc_config),
  "r-l1": \
    dict(type=float, default=def_r_l1,
    help=help_r_l1),
  "r-l2": \
    dict(type=float, default=def_r_l2,
    help=help_r_l2),
  "model-config": \
    dict(type=str, default=def_model_config,
    choices=choices_model_config,
    help=help_model_config),
  "slim-config": \
    dict(action="store", type=str, default=def_slim_config,
    help=help_slim_config),
  "clip-sample-weight": \
    dict(type=float, default=def_clip_sample_weight,
    help=def_clip_sample_weight),
  "init-dist": \
    dict(type=str, default=def_init_distr,
    help=help_init_distr),
  "LRS": \
    dict(action="store_true",
    help=help_LRS),
  "activation-function": \
    dict(type=str, default=def_activation_function, choices=choices_activation_function,
    help=help_activation_function),
  "jssv-config": \
    dict(type=str, default=def_jssv_config,
    help=help_jssv_config),
  "batch-norm": \
    dict(type=int, default=def_batch_norm,
    choices=choices_batch_norm,
    help=help_batch_norm),
  "architecture": \
    dict(type=str, default=def_architecture,
    choices=choices_architecture,
    help=help_architecture),
  "start": \
    dict(type=int, default=def_start,
    help=help_start),
  "step": \
    dict(type=int, default=def_step,
    help=help_step),
  "stop": \
    dict(type=int, default=def_stop,
    help=help_stop),
  "flat-pt": \
    dict(type=str, default=def_flat_pt, choices=choices_flat_pt,
    help=help_flat_pt),
  "max-processes": \
    dict(type=int, default=def_max_processes,
    help=help_max_processes),
  "entries": \
    dict(type=str, nargs="+", default=def_entries,
    help=help_entries),
  "txt-top": \
    dict(type=str, nargs="+", default=def_txt_top,
    help=help_txt_top),
  "txt-pad": \
    dict(type=str, nargs="+", default=def_txt_pad,
    help=help_txt_pad),
  "poly-order": \
    dict(type=int, default=def_poly_order,
    help=help_poly_order),
  "config": \
    dict(type=str, required=True,
    help=help_config),
  "working-point": \
    dict(type=int,
    help=help_working_point),
  "load": \
    dict(type=bool, default=False,
    help=""),
  "Phi-sizes": \
    dict(type=str, nargs="+", default=[100, 100],
    help=""),
  "F-sizes": \
    dict(type=str, nargs="+", default=[100, 100],
    help="")
}


def get_parser (**kwargs):

  import argparse

  # Validate
  kwargs = {k.replace("_","-"): v for (k,v) in kwargs.items()}
  for k in set(kwargs) - set(arguments):
    raise IOError("get_parser: [ERROR] Keyword %s is not supported." % k)

  # Construct parser
  parser = argparse.ArgumentParser(description="[Generic] Argiment parser.")
  for k in filter(lambda k: kwargs[k], kwargs):
    parser.add_argument("--" + k, **arguments[k])

  return parser
