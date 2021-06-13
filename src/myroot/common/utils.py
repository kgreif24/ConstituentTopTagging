# Basic import(s)
import os
import gc


def garbage_collect (f):
  """
  Function decorator to manually perform garbage collection after the call,
  so as to avoid unecessarily large memory consumption.
  """

  def wrapper(*args, **kwargs):
    ret = f(*args, **kwargs)
    gc.collect()
    return ret

  return wrapper


def mkdir (path):

  """Script to ensure that the directory at `path` exists.

  Arguments:
      path: String specifying path to directory to be created.
  """

  # Check mether  output directory exists
  if not os.path.exists(path):
    print "mkdir: Creating output directory:\n  {}".format(path)
    try:
       os.makedirs(path)
    except OSError:
      # Apparently, 'path' already exists.
      pass
    pass
  return


def purge (dir_, pattern):

  import re
  for f in os.listdir(dir_):
    if re.search(pattern, f):
      os.remove(os.path.join(dir_, f))


def dump_argparse(args, fname):

  import json
  with open(fname, "w") as f:
    json.dump(args.__dict__, f, indent=2)


def load_argparse(args, fname):

  import json
  with open(fname, "w") as f:
    args.__dict__ = json.load(f)
