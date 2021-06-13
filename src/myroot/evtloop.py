import ROOT

class EventLoop(ROOT.RDataFrame):

  def __init__(self, tree_name, samples):

    if not isinstance(samples, list): samples = [samples]
    ROOT.RDataFrame.__init__(self, tree_name, samples)

    pass


if __name__ == "__main__":

  el = EventLoop()
  print(el)

  pass
