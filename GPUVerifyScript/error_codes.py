"""Module defining the error codes used by GPUVerify and gvtester."""

class ErrorCodes(object):
  SUCCESS = 0
  COMMAND_LINE_ERROR = 1
  CLANG_ERROR = 2
  OPT_ERROR = 3
  BUGLE_ERROR = 4
  GPUVERIFYVCGEN_ERROR = 5
  BOOGIE_ERROR = 6
  TIMEOUT = 7
  CTRL_C = 8
  CONFIGURATION_ERROR = 9
  REGEX_MISMATCH_ERROR = 100 # Only used by gvtester
