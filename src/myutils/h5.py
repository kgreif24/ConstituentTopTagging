import os, sys

# Load some cool stuff
import clean
import root_numpy
import h5py
import orga


@clean.garbage_collect
def save_hdf5 (data, path, name="dataset", write_mode="w", gzip=True, metadata=None, verbose=True):
  """
  Save numpy recarray to HDF5 file.

  Arguments:
      data: Numpy recarray to be saved to file.
      path: Path to HDF5 save file.
      name: Name of dataset in which to store the data.
      gzip: Whether to apply gzip compression to HDF5 file.
  """

  # Add extension if neccessary
  if not path.endswith(".h5"):
    path += ".h5"

  # Ensure directory exists
  basedir = "/".join(path.split("/")[:-1])
  if basedir: orga.mkdir(basedir)

  # Save array to HDF5 file
  with h5py.File(path, write_mode) as hf:
    dset = hf.create_dataset(name, data=data, compression="gzip" if gzip else None)
    # Set metadata
    if metadata:
      if isinstance(metadata, dict):
        for key in metadata:
          dset.attrs[key] = metadata[key]
      else:
        dset.attrs = metadata
          
  if verbose:
    print("[INFO] Saved file: %s" % path)


@clean.garbage_collect
def load_hdf5 (path, name="dataset", metadata=False):
  """
  Load numpy recarray from HDF5 file.

  Arguments:
      path: Path to HDF5 from which to read array.
      name: Name of dataset in which data is stored.
  """

  # Load array from HDF5 file
  with h5py.File(path, "r") as hf:
    data = hf[name][:]
    meta = dict(hf[name].attrs)

  if not metadata:
    return data
  else:
    return data, meta


def load_metadata (path, name="dataset"):

  """
  Load get from HDF5 file.

  Arguments:
      path: Path to HDF5 from which to read array.
      name: Name of dataset in which data is stored.
  """

  # Load array from HDF5 file
  with h5py.File(path, "r") as hf:
    if hasattr(hf[name], "attrs"):
      meta = dict(hf[name].attrs)
      return meta
    else:
      return dict()


def is_dataset(path, name="dataset"):

  """
  Check if HDF5 file contains a dataset with name `name`.

  Arguments:
      path: Path to HDF5 from which to read array.
      name: Name of dataset in which data is stored.
  """

  # Load array from HDF5 file
  with h5py.File(path, "r") as hf:
    keys = hf.keys()

  if name in keys: return True
  else: return False


def rootlike_selection(arr, branches=None, selection=None, object_selection=None, start=None, stop=None, step=None, include_weight=False, weight_name="weight", cache_size=-1):

  # Convert array to ROOT.TTree to make use of selection
  t_tree_tmp = root_numpy.array2tree(arr, name="tree_tmp", tree=None)

  # Apply selection on ttree and convert back to array
  arr_selec = root_numpy.tree2array(t_tree_tmp, branches=branches, selection=selection, object_selection=object_selection, start=start, stop=stop, step=step, include_weight=include_weight, weight_name=weight_name)

  import h5py
  # Return array with selections applied
  return arr_selec


if __name__ == "__main__":

  import numpy as np
  # Cuts
  a = np.array([(1, 2, 3), (4, 5, 6)], 
    dtype=[("a", np.int32), ("b", np.float32), ("c", np.float64)])
  print("[INFO] Before selection:", a)
  print("[INFO] After selection:", rootlike_selection(a, selection="a>1"))
