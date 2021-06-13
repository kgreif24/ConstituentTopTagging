# Basic import(s)
import os

def mkdir (path):

  """Script to ensure that the directory at `path` exists.

  Arguments:
      path: String specifying path to directory to be created.
  """

  # Check mether  output directory exists
  if not os.path.exists(path):
    try:
       os.makedirs(path)
    except OSError:
      # Apparently, 'path' already exists.
      pass
    pass
  return


def dump_argparse(args, fname):

  import json
  with open(fname, "w") as f:
    json.dump(args.__dict__, f, indent=2)


def load_argparse(args, fname):

  import json
  with open(fname, "w") as f:
    args.__dict__ = json.load(f)

