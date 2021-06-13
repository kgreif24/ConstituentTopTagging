import os

class Walker:

  """
  Simple class to manage projects and its directories
  """

  def __init__(self, proj_name, write_mode="RECREATE"):

    dir_list = os.path.abspath(proj_name).split(os.sep)
    # Set root dir
    self.root_dir = "./"
    if len(dir_list) > 1:
      self.root_dir = os.path.join(os.sep, *dir_list[0:-1])

    # Set project name and write mode
    self.proj_name = dir_list[-1]
    self.write_mode = write_mode

    # Load project depending on arguments
    self.loadProject()

    # Create basic directory table as dictionary
    self.dirs = { "root" : self.root_dir, "home" : self.home_dir}


  def addDir(self, path_from_home):

    tmp_path = os.path.join(self.home_dir, path_from_home)
    # Check if directory already exists
    if tmp_path in self.dirs:
      return
    # If the given path defines subdirectories dir1/dir2/... create a dictionary entry dor each one
    for d in path_from_home.split("/"):
      # Else add it
      self.dirs[path_from_home] = tmp_path
      # And create it
      self.mkDir(tmp_path)
    return self


  def buildNewProject(self, suffix=""):

    dirname = os.path.join(self.root_dir, self.proj_name, suffix)
    # Create folder
    self.mkDir(dirname)
    # Update path variable
    self.home_dir = dirname
    print("INFO: Created new project: %s" % self.home_dir)


  def getHomeDir(self):

    if self.write_mode.lower() == "update":
      # Load the very last project (default)
      self.loadLastProject()
    else:
      # Create new project
      self.buildNewProject()


  def get(self, dir_, fname=None):

      if dir_ in self.dirs:
        if fname == None:
          return self.dirs[dir_]
        else:
          return os.path.join(self.dirs[dir_], fname)
      else:
        self.addDir(dir_)
        if fname == None:
          return self.get(dir_)
        else:
          return os.path.join(self.get(dir_), fname)


  def loadProject(self):

    if self.write_mode.lower() == "update":
      # Load the very last project (default)
      self.loadLastProject()
    else:
      # Create new project
      self.buildNewProject()


  def loadLastProject(self):

    # Check if there is a project
    if not os.path.isdir(self.root_dir) or not len(os.listdir(self.root_dir)):
      print("WARNING: the project '%s' does not exist/is empty! Creating a new one" % self.root_dir)
      self.buildNewProject()
      return
    # Get latest project in root dir and use this one
    dir_list = next(os.walk(self.root_dir))[1]
    # Check if directory already exists
    if not any(self.proj_name == d for d in dir_list):
      print("WARNING: There is no project with name '%s'. Building a new one." % self.proj_name)
      self.buildNewProject()
      return
    # Filter corresponding mode
    dir_list = [d for d in dir_list if self.proj_name in d]
    self.home_dir = os.path.join(self.root_dir, self.proj_name)
    print("INFO: Loaded pre-existing project: %s" % self.home_dir)


  def mkDir(self, dirname):

    if not os.path.exists(dirname):
      os.makedirs(dirname)
      print("INFO: Created directory: %s" % dirname)

