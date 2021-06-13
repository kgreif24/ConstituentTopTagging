import root_numpy
import numpy
import h5py
import myutils.h5


def _not_found(keys_available, keys_requested):

  if isinstance(keys_requested, list):
    if not all(key in keys_available  for key in keys_requested):
      print("[ERROR] Not all requested keys (%s) are in HDF5 file" % keys_requested)
      print("        Choose between: %s" % ", ".join(keys_available))
      sys.exit()
  elif isinstance(keys_requested, str):
    if keys_requested not in keys_available:
      print("[ERROR] Not all requested keys (%s) are in HDF5 file" % keys_requested)
      print("        Choose between: %s" % ", ".join(keys_available))
      sys.exit()


def root2h5py (filenames, treename=None, out_name="output.h5", branches=None, selection=None, object_selection=None, start=None, stop=None, step=None, include_weight=False, weight_name="weight", cache_size=-1, warn_missing_tree=False):

  # Each additional argument will be stored as metadata
  metadata = locals()

  # Convert trees in ROOT files into a numpy structured array
  out_array = root_numpy.root2array(filenames, treename, branches, selection, object_selection, start, stop, step, include_weight, weight_name, cache_size, warn_missing_tree) 

  # Create a h5py file
  myutils.h5.save_hdf5(out_array, out_name, name=treename, write_mode="w", metadata=metadata, gzip=True)
  

def root2array (path2file, treename="EventTree", branches=None, selection=None, object_selection=None, start=None, stop=None, step=None, include_weight=False, weight_name="weight", cache_size=-1, warn_missing_tree=False, to_array=False):

  # Split in the end?
  split = False

  # Check if branches is a list of lists
  _branches = list(branches)
  if any(isinstance(item, list) for item in _branches):
    for i in range(len(_branches)):
       if not isinstance(_branches[i], list):
        _branches[i] = [_branches[i]]
    split = True
    # Flatten
    _branches = [branch for item in _branches for branch in item]

  # Convert trees in ROOT files into a numpy structured array
  arr = root_numpy.root2array(path2file, treename, _branches, selection, object_selection, start, stop, step, include_weight, weight_name, cache_size, warn_missing_tree) 

  if not split:
    if to_array:
      arr = recarray2array(arr)
    return arr
  else:
    r_tuple = []
    for group in branches:
      arr_group = arr[group]
      if to_array:
        arr_group = recarray2array(arr_group)
      r_tuple.append(numpy.array(arr_group))
    return tuple(r_tuple)


def h5py2root (filename, treename="tree", mode="update"):

  # Open h5py and load data into memory
  array = myutils.h5.load_hdf5(filename, name=treename)

  # Dump to root file
  root_numpy.array2root(arr, filename.replace(".h5", ".root"), treename, mode)


def struct2array (strctarray):

  return numpy.array(strctarray.tolist())


def array2root (arr, branches, path2file="output.root", tree_name="EventTree", mode="update", scaler=None):

  # Convert array to structures arrayp
  arr_rec = numpy.core.records.fromrecords(arr, names=branches)
 
  # Undo preprocessing?
  if scaler != None:
    for key in inpt.dtype.names:
      inpt[key] = scaler.transform(inpt[key], key, inverse=True)

  # To ROOT file 
  root_numpy.array2root(arr_rec, path2file, treename=tree_name, mode=mode)


def recarray2root (arr, path2file="output.root", tree_name="EventTree", mode="update", scaler=None):

  # Undo preprocessing?
  if scaler != None:
    for key in arr.dtype.names:
      arr[key] = scaler.inverseTransform(arr[key], key)

  # To ROOT file 
  root_numpy.array2root(arr, path2file, treename=tree_name, mode=mode)
  print("[INFO] [myutils.conversion] Saved file: %s" % path2file)


def h5py2array (path2file, inputs, targets=[], ds_name=None, n=-1, selection=None, sample_weight=[], class_weight=[], scaler=None, shuffle=True, to_array=True, stop=-1):

  with h5py.File(path2file, "r") as f:
 
    if ds_name != None and ds_name in f.keys():
      ds = f[ds_name]
    else:
      ds = f

    # Make a preselection before loading everything into memory
    if not isinstance(inputs, list): inputs = [inputs]
    if not isinstance(targets, list): targets = [targets]
    if not isinstance(sample_weight, list): sample_weight = [sample_weight]
    if not isinstance(class_weight, list): class_weight = [class_weight]
    keys_all =  list(inputs)
    if targets:
      keys_all += targets
    if sample_weight:
      keys_all += sample_weight
    if class_weight:
      keys_all += class_weight
    # Also check for keys that are requested in selection
    if selection != None:
      for key in ds.dtype.names:
        if key in selection and key not in keys_all:
          keys_all.append(key)
    # Get data with fields
    ds = ds[tuple(keys_all)]

    # Load all data into memory
    if stop != -1:
      print("[WARNING] `stop` must be used with care")
    ds = ds[0:stop]

    # If requested, apply a selection on entries
    if selection != None:
      if not isinstance(selection, list):
        ds = myutils.h5.rootlike_selection(ds, selection=selection)

    # Tuple/list with the arrays to be returned
    r_tuple = [] 
    
    # Actual inputs
    _not_found(ds.dtype.names, inputs)
    inpt = ds[inputs]
    n_events = inpt.shape[0]
    # Preprocess the dataset?
    if scaler != None:
      # Fit scaler to dataset and transform data
      for key in inpt.dtype.names:
        # If the scaler has already been fitted to data, fit won't have any effect
        scaler.fit(inpt[key], key)
        # Apply preprocessing
        inpt[key] = scaler.transform(inpt[key], key)
    r_tuple.append(inpt)
    # Targets
    if targets:
      _not_found(ds.dtype.names, targets)
      r_tuple.append(ds[targets])
    # Sample weights
    if sample_weight:
      _not_found(ds.dtype.names, sample_weight)
      r_tuple.append(ds[sample_weight])
    # Classes
    if class_weight:
      _not_found(ds.dtype.names, class_weights)
      r_tuple.append(ds[class_weight])

    if to_array:
      # Convert to normal numpy arrays
      for i in range(len(r_tuple)):
        r_tuple[i] = numpy.array(r_tuple[i].tolist())

    if shuffle:
      # Get array of permuted indices
      idx = numpy.random.permutation(n_events)
      for i in range(len(r_tuple)):
        r_tuple[i] = r_tuple[i][idx]

    # Reduce number of events
    if n != -1 and n < n_events:
      for i in range(len(r_tuple)):
        r_tuple[i] = r_tuple[i][0:n]

    if len(r_tuple) > 1:
      return tuple(r_tuple)
    else:
      return r_tuple[0]


def recarray2array (arr):

  return numpy.array(arr.tolist())


def array2recarray (arr, keys):

  # Convert array to structures arrayp
  return numpy.core.records.fromrecords(arr, names=keys)


def h52hepmc (fname):

  # (Original author: Karl Nordstrom)
  
  input_filename = sys.argv[1]
  store = pandas.HDFStore(input_filename)
  
  
  writer_signal = open(sys.argv[2], "w")
  writer_background = open(sys.argv[3], "w")
  writer_signal.write('HepMC::Version 2.06.00\nHepMC::IO_GenEvent-START_EVENT_LISTING\n')
  writer_background.write('HepMC::Version 2.06.00\nHepMC::IO_GenEvent-START_EVENT_LISTING\n')
  event_number = 1
  
  events = store.select("table")
  for row in events.iterrows():
    if row[1]["is_signal_new"] == 1:
      writer_signal.write('E ' + str(event_number) + ' 0 ')
      writer_signal.write('1000. 0.12 0.01 ')
      writer_signal.write( '10 -1' + ' 1 ')
      writer_signal.write('1 2'+ ' 0 0 0\n')
      writer_signal.write('U GEV MM\n')
      writer_signal.write('C ' + '10' + ' ' + '0.1' + '\n')
      writer_signal.write('V -1' + ' ' + '100001' + ' ' + '0.' + ' ' + '0.' + ' ' + '0.' + ' ' + '0.' + ' ' + '2' + ' ' + '200' + ' 0\n')
      writer_signal.write('P ' + '1' + ' ' + '2212' + ' ' +'0.' + ' ' +'0.'+ ' ' + '500' + ' ' + '500' + ' ' + '0' + ' ' + '4' + ' ' + '0' + ' ' + '0' + ' ' + '0' + ' 0\n' )
      writer_signal.write('P ' + '2' + ' ' + '2212' + ' ' +'0.' + ' ' +'0.'+ ' ' + '-500' + ' ' + '500' + ' ' + '0' + ' ' + '4' + ' ' + '0' + ' ' + '0' + ' ' + '0' + ' 0\n' )
      for i in range(200):
  	       writer_signal.write('P ' + str(i+3) + ' ' + '2212' + ' ' + str(row[1][1+4*i]) + ' ' +str(row[1][2+4*i]) + ' ' + str(row[1][3+4*i]) + ' ' + str(row[1][0+4*i]) + ' ' + '0' + ' ' + '1' + ' ' +  '0' + ' ' +  '0'  + ' ' + '0'  + ' 0\n' )
    elif row[1]["is_signal_new"] == 0:
      writer_background.write('E ' + str(event_number) + ' 0 ')
      writer_background.write('1000. 0.12 0.01 ')
      writer_background.write( '10 -1' + ' 1 ')
      writer_background.write('1 2'+ ' 0 0 0\n')
      writer_background.write('U GEV MM\n')
      writer_background.write('C ' + '10' + ' ' + '0.1' + '\n')
      writer_background.write('V -1' + ' ' + '100001' + ' ' + '0.' + ' ' + '0.' + ' ' + '0.' + ' ' + '0.' + ' ' + '2' + ' ' + '200' + ' 0\n')
      writer_background.write('P ' + '1' + ' ' + '2212' + ' ' +'0.' + ' ' +'0.'+ ' ' + '500' + ' ' + '500' + ' ' + '0' + ' ' + '4' + ' ' + '0' + ' ' + '0' + ' ' + '0' + ' 0\n' )
      writer_background.write('P ' + '2' + ' ' + '2212' + ' ' +'0.' + ' ' +'0.'+ ' ' + '-500' + ' ' + '500' + ' ' + '0' + ' ' + '4' + ' ' + '0' + ' ' + '0' + ' ' + '0' + ' 0\n' )
      for i in range(200):
  	       writer_background.write('P ' + str(i+3) + ' ' + '2212' + ' ' + str(row[1][1+4*i]) + ' ' +str(row[1][2+4*i]) + ' ' + str(row[1][3+4*i]) + ' ' + str(row[1][0+4*i]) + ' ' + '0' + ' ' + '1' + ' ' +  '0' + ' ' +  '0'  + ' ' + '0'  + ' 0\n' )
    print("Processed event number: " + str(event_number))
    event_number = event_number + 1
