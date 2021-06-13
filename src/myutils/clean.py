# Basic import(s)
import os


def garbage_collect (f):
  """
  Function decorator to manually perform garbage collection after the call,
  so as to avoid unecessarily large memory consumption.
  """

  import gc
  def wrapper (*args, **kwargs):
    ret = f(*args, **kwargs)
    gc.collect()
    return ret

  return wrapper


def purge (dir_, pattern):

  import re
  for f in os.listdir(dir_):
    if re.search(pattern, f):
      os.remove(os.path.join(dir_, f))

