import os, sys
import h5py


def batch2h5py (path2file, dataset, group, write_mode="a", metadata=None, verbose=False):

  # Check write mode
  if os.path.isfile(path2file) and write_mode == "w":
    write_mode = "a"
  # Create the output directory if it does not exist
  path2dir = os.path.dirname(path2file)
  if not os.path.exists(path2dir) and path2dir:
    os.makedirs(path2dir)

  if verbose:
    print("[\033[1mINFO\033[0m] Write data to batch.")

  with h5py.File(path2file, write_mode) as f:

    # Check if the group is already defined in the data set
    if group not in f.keys():
      grp = f.create_group(group)

    # Loop over all keys/fields in the data set
    for key in dataset.dtype.names:
      if key not in f[group].keys():
        shape = list(dataset[key].view().shape)
        shape[0] = None
        shape = tuple(shape)
        dset = f.create_dataset(os.path.join(group, key), dataset[key].view().shape, dtype=dataset[key].view().dtype, chunks=dataset[key].view().shape, maxshape=shape)
        dset[:] = dataset[key]
      else:
        dset = f[os.path.join(group, key)]
        dset.resize(dset.shape[0] + dataset[key].shape[0], axis=0)
        dset[-dataset[key].shape[0]:] = dataset[key].view()
      if verbose:
        print("[    ] Dataset: %s - group: %s - length: %s" % (group, key, dset.shape[0]))

    # Add some metadata to this data set by setting an attribute
    if metadata is not None:
      if isinstance(metadata, dict):
        for key in metadata:
          f[group].attrs[key] = metadata[key]
      else:
        f[group].attrs = metadata

