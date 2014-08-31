"""Module defining script-side constants"""

class AnalysisMode(object):
  """ ALL is the default mode. Right now it is the same as VERIFY, but in
  the future this mode will run verification and bug-finding in parallel.
  """
  ALL = 0
  FINDBUGS = 1
  VERIFY = 2

class SourceLanguage(object):
  Unknown = 0
  OpenCL = 1
  CUDA = 2
