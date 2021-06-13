import ROOT

def conv2vec(inputs, type="string"):

  """
    ROOT's RDF needs an std::vector as input

      Parameters:
        fname (str/list of str): String or list of strings filled into the vector

      Returns:
        std_vector (ROOT.vector): Input as ROOT.vector
  """

  if not isinstance(inputs, list):
    inputs = [inputs]

  vector = ROOT.vector(type)()
  for item in inputs:
    vector.push_back(item)

  return vector
