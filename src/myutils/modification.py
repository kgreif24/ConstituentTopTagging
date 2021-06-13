
def add_field(data_old, name, data_new=None, function=None, fill_value=-1, usemask=False):

  # If the branch already exists, do nothing
  if name in data_old.dtype.names: return data_old

  if function != None:
    data_new = function(data_old)

  import numpy as np
  from numpy.lib.recfunctions import append_fields
  data_old = append_fields(data_old, name, data_new, fill_value=fill_value, usemask=usemask)
  return data_old


def drop_field(data, name):

   # Some checks on input
   if not isinstance(name, list): name = [name]

   import numpy as np
   import numpy.lib.recfunctions

   for field in name:
     if field in data.dtype.names:
       data = np.lib.recfunctions.drop_fields(data, field)

   return data

