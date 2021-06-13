
def dict2json(arg, fname, indent=2, write_mode="w"):

  import json
  with open(fname, write_mode) as f:
    json.dump(arg, f, indent=indent)

