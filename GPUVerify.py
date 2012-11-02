#!/usr/bin/env python

"""
Generate a prefix sum for a given array length [N]
"""

import getopt
import sys
import os

import subprocess
import time

""" We support three analysis modes """
class AnalysisMode(object):
  """ This is the default mode, in which both verification and bug-finding will be run in parallel """
  ALL=0
  """ This is bug-finding only mode """
  FINDBUGS=1
  """ This is verify only mode """
  VERIFY=2

""" We support OpenCL and CUDA """
class SourceLanguage(object):
  Unknown=0
  OpenCL=1
  CUDA=2

""" Options for the tool """
class CommandLineOptions(object):
  SL = SourceLanguage.Unknown
  sourceFiles = []
  mode = AnalysisMode.ALL
  inference = True
  verbose = False
  groupSize = []
  numGroups = []

class Timeout(Exception):
    pass

def run(command):
  if CommandLineOptions.verbose:
    print " ".join(command)
  proc = subprocess.Popen(command, bufsize=0, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = proc.communicate()
  return stdout, stderr, proc.returncode

class ErrorCodes(object):
  SUCCESS = 0
  COMMAND_LINE_ERROR = 1
  CLANG_ERROR = 2
  OPT_ERROR = 3
  BUGLE_ERROR = 4
  GPUVERFYVCGEN_ERROR = 5
  BOOGIE_ERROR = 6


bugleDir = "C:\\prog\\bugle"
libclcDir = "C:\\prog\\libclc"
llvmBinDir = "C:\\prog\\llvm-build\\bin\\Release"
bugleBinDir = "C:\\prog\\BugleBuild\Debug"

""" Base class to handle the invocation of the various tools of which GPUVerify is comprised """
class ToolOptions(object):
  def __init__(self, toolname):
    self.toolname = toolname
    self.options = []
  def makeCommand(self):
    return [ self.toolname ] + self.options

""" Options to be passed to 'clang' when processing a kernel """

clangCoreOptions = [ "-ccc-host-triple", "nvptx--bugle",
                     "-g",
                     "-emit-llvm",
                     "-c" ]
clangCoreIncludes = [ bugleDir + "/include-blang" ]
clangCoreDefines = []

clangOpenCLOptions = [ "-Xclang", "-cl-std=CL1.2",
                       "-O0",
                       "-Xclang", "-mlink-bitcode-file",
                       "-Xclang", libclcDir + "/nvptx--bugle/lib/builtins.bc",
                       "-include", "opencl.h"
                     ]
clangOpenCLIncludes = [ libclcDir + "/generic/include" ]
clangOpenCLDefines = [ "cl_khr_fp64",
                       "cl_clang_storage_class_specifiers",
                       "__OPENCL_VERSION__"
                     ]

clangCUDAOptions = []
clangCUDAIncludes = []
clangCUDADefines = []

class ClangOptions(ToolOptions):
  def __init__(self):
    super(ClangOptions, self).__init__(llvmBinDir + "/clang")
    self.options = clangCoreOptions
    self.includes = clangCoreIncludes
    self.defines = clangCoreDefines
  def makeCommand(self):
    return (super(ClangOptions, self).makeCommand() + 
              [("-I" + str(o)) for o in self.includes] +
              [("-D" + str(o)) for o in self.defines])


def showHelpAndExit():
  print "OVERVIEW: GPUVerify driver"
  print ""
  print "USAGE: GPUVerify.py [options] <inputs>"
  print ""
  print "GENERAL OPTIONS:"
  print "  -h, --help              Display this message"
  print "  -I <value>              Add directory to include search path"
  print "  -D <value>              Define symbol"
  print "  --findbugs              Run tool in bug-finding mode"
  print "  --noinfer               Turn off invariant inference"
  print "  --verify                Run tool in verification mode"
  print "  --verbose               Show commands to run and use verbose output"
  print ""
  print "OPENCL OPTIONS:"
  print "  --local_size=X          Specify whether work-group is 1D, 2D"         
  print "              =(X,Y)      or 3D and specify size for each"
  print "              =(X,Y,Z)    dimension"
  print "  --num_groups=X          Specify whether grid of work-groups is"         
  print "              =(X,Y)      1D, 2D or 3D and specify size for each"
  print "              =(X,Y,Z)    dimension"
  print ""
  print "CUDA OPTIONS"
  print "  --blockDim=X            Specify whether thread block is 1D, 2D"         
  print "              =(X,Y)      or 3D and specify size for each"
  print "              =(X,Y,Z)    dimension"
  print "  --gridDim=X             Specify whether grid of thread blocks is"         
  print "              =(X,Y)      1D, 2D or 3D and specify size for each"
  print "              =(X,Y,Z)    dimension"
  exit(0)

def processVector(vector):
  vector = vector.strip()
  if vector[0] == '(' and vector[len(vector)-1] == ')':
    return map(int, vector[1:len(vector)-1].split(","))
  else:
    return [ int(vector) ]


def GPUVerifyError(msg, code):
  print "GPUVerify: error: " + msg
  exit(code)

def Verbose(msg):
  if(CommandLineOptions.verbose):
    print msg

def getSourceFiles(args):
  if len(args) == 0:
    GPUVerifyError("no .cl or .cu files supplied", ErrorCodes.COMMAND_LINE_ERROR)
  for a in args:
    filename, ext = os.path.splitext(a)
    if ext == ".cl":
      if CommandLineOptions.SL == SourceLanguage.CUDA:
        GPUVerifyError("illegal to pass both .cl and .cu files simultaneoulsy", ErrorCodes.COMMAND_LINE_ERROR)
      CommandLineOptions.SL = SourceLanguage.OpenCL
    elif ext == ".cu":
      if CommandLineOptions.SL == SourceLanguage.OpenCL:
        GPUVerifyError("illegal to pass both .cl and .cu files simultaneoulsy", ErrorCodes.COMMAND_LINE_ERROR)
      CommandLineOptions.SL = SourceLanguage.CUDA
    else:
      GPUVerifyError("unknown file extension " + ext + ", supported file extensions are .cl (OpenCL) and .cu (CUDA)", ErrorCodes.COMMAND_LINE_ERROR)
    CommandLineOptions.sourceFiles.append(a)

def showHelpIfRequested(opts):
  for o, a in opts:
    if o == "--help" or o == "-h":
      showHelpAndExit()

def processGeneralOptions(opts, args):
  for o, a in opts:
    if o == "-D":
      clangOptions.defines.append(a)
    if o == "-I":
      clangOptions.includes.append(a)
    if o == "--findbugs":
      CommandLineOptions.mode = AnalysisMode.FINDBUGS
    if o == "--verify":
      CommandLineOptions.mode = AnalysisMode.VERIFY
    if o == "--noinfer":
      CommandLineOptions.inference = False
    if o == "--verbose":
      CommandLineOptions.verbose = True

def processOpenCLOptions(opts, args):
  for o, a in opts:
    if o == "--local_size":
      if CommandLineOptions.groupSize != []:
        raise Exception("Must not define local_size twice")
      CommandLineOptions.groupSize = processVector(a)
    if o == "--num_groups":
      if CommandLineOptions.numGroups != []:
        raise Exception("Must not define num_groups twice")
      CommandLineOptions.numGroups = processVector(a)

def processCUDAOptions(opts, args):
  return

def main(argv=None):
  if argv is None:
    argv = sys.argv
  progname = argv[0]

  clangOptions = ClangOptions();

  try:
    opts, args = getopt.getopt(argv[1:],'D:I:h', 
             ['help', 'findbugs', 'verify', 'noinfer', 'verbose',
              'local_size=', 'num_groups=',
              'blockDim=', 'gridDim='])
  except getopt.GetoptError as getoptError:
    GPUVerifyError(getoptError.msg + ".  Try --help for list of options", ErrorCodes.COMMAND_LINE_ERROR)

  showHelpIfRequested(opts)
  getSourceFiles(args)
  processGeneralOptions(opts, args)
  if CommandLineOptions.SL == SourceLanguage.OpenCL:
    processOpenCLOptions(opts, args)
  if CommandLineOptions.SL == SourceLanguage.CUDA:
    processCUDAOptions(opts, args)

  filename, ext = os.path.splitext(args[0])

  if ext == ".cl":
    clangOptions.options += clangOpenCLOptions
    clangOptions.includes += clangOpenCLIncludes
    clangOptions.defines += clangOpenCLDefines
    clangOptions.defines.append("__" + str(len(CommandLineOptions.groupSize)) + "D_WORK_GROUP")
    clangOptions.defines.append("__" + str(len(CommandLineOptions.numGroups)) + "D_GRID")
    clangOptions.defines += [ "__LOCAL_SIZE_" + str(i) + "=" + str(CommandLineOptions.groupSize[i]) for i in range(0, len(CommandLineOptions.groupSize))]
    clangOptions.defines += [ "__NUM_GROUPS_" + str(i) + "=" + str(CommandLineOptions.numGroups[i]) for i in range(0, len(CommandLineOptions.numGroups))]

  else:
    assert(ext == ".cu")
    GPUVerifyError("support for CUDA not yet complete; please contact the developers", ErrorCodes.COMMAND_LINE_ERROR)

  clangOptions.options.append("-o")
  clangOptions.options.append(filename + ".bc")
  clangOptions.options.append(filename + ext)
  Verbose("Running clang")
  clangStdout, clangStderr, clangReturn = run(clangOptions.makeCommand())
  if clangReturn != 0:
    print clangStderr
    exit(ErrorCodes.CLANG_ERROR)

  optOptions = ToolOptions(llvmBinDir + "/opt");
  optOptions.options += [ "-mem2reg", "-globaldce", "-o", filename + ".opt.bc", filename + ".bc" ]
  Verbose("Running opt")
  optStdout, optStderr, optReturn = run(optOptions.makeCommand())
  if optReturn != 0:
    raise Exception(optStderr)

  bugleOptions = ToolOptions(bugleBinDir + "/bugle");
  bugleOptions.options += [ "-l", "cl" if ext == ".cl" else "cu", "-o", filename + ".gbpl", filename + ".opt.bc"]
  Verbose("Running bugle")
  bugleStdout, bugleStderr, bugleReturn = run(bugleOptions.makeCommand())
  if bugleReturn != 0:
    raise Exception(bugleStderr)

  gpuVerifyVCGenOptions = ToolOptions("GPUVerifyVCGen")
  gpuVerifyVCGenOptions.options += [ "/print:" + filename, filename + ".gbpl" ]

  if CommandLineOptions.mode == AnalysisMode.FINDBUGS or (not CommandLineOptions.inference):
    gpuVerifyVCGenOptions.options += [ "/noInfer" ]

  Verbose("Running gpuverifyvcgen")
  gpuVerifyVCGenStdout, gpuVerifyVCGenStderr, gpuVerifyVCGenReturn = run(gpuVerifyVCGenOptions.makeCommand())
  if gpuVerifyVCGenReturn != 0:
    raise Exception(gpuVerifyVCGenStderr)

  gpuVerifyBoogieDriverOptions = ToolOptions("GPUVerifyBoogieDriver")
  gpuVerifyBoogieDriverOptions.options += [ "/nologo",
                                            "/typeEncoding:m", 
                                            "/doModSetAnalysis", 
                                            "/proverOpt:OPTIMIZE_FOR_BV=true", 
                                            "/useArrayTheory", 
                                            "/z3opt:RELEVANCY=0", 
                                            "/z3opt:SOLVER=true", 
                                            "/doNotUseLabels", 
                                            "/noinfer", 
                                            "/enhancedErrorMessages:1",
                                            "/errorLimit:20",
                                            filename + ".bpl" ]
  if CommandLineOptions.mode == AnalysisMode.FINDBUGS:
    gpuVerifyBoogieDriverOptions.options += [ "/loopUnroll:2" ]
  elif not CommandLineOptions.inference:
    gpuVerifyBoogieDriverOptions.options += [ "/contractInfer" ]

  Verbose("Running gpuverifyboogiedriver")
  gpuVerifyBoogieDriverStdout, gpuVerifyBoogieDriverStderr, gpuVerifyBoogieDriverReturn = run(gpuVerifyBoogieDriverOptions.makeCommand())
  if gpuVerifyBoogieDriverReturn != 0:
    raise Exception(gpuVerifyBoogieDriverStderr)

  print gpuVerifyBoogieDriverStdout + gpuVerifyBoogieDriverStderr



if __name__ == '__main__':
  sys.exit(main())
