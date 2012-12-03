#!/usr/bin/env python

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
  adversarialAbstraction = False
  equalityAbstraction = False
  loopUnwindDepth = 2
  noBenign = False
  onlyDivergence = False
  onlyIntraGroup = False
  noLoopPredicateInvariants = False
  noSmartPredication = False
  noSourceLocInfer = False
  noUniformityAnalysis = False
  vcgenExtraOptions = []
  

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


bugleDir = sys.path[0] + "/bugle"
libclcDir = sys.path[0] + "/libclc"
llvmBinDir = sys.path[0] + "/bin"
bugleBinDir = sys.path[0] + "/bin"
gpuVerifyVCGenBinDir = sys.path[0] + "/bin"
gpuVerifyBoogieDriverBinDir = sys.path[0] + "/bin"
z3BinDir = sys.path[0] + "/bin"


""" Base class to handle the invocation of the various tools of which GPUVerify is comprised """
class ToolOptions(object):
  def __init__(self, toolname):
    self.toolname = toolname
    self.options = []
  def makeCommand(self):
    return [ self.toolname ] + self.options

""" Options to be passed to 'clang' when processing a kernel """

clangCoreOptions = [ "-target", "nvptx--bugle",
                     "-g",
                     "-gcolumn-info",
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

clangCUDAOptions = [ "-Xclang", "-fcuda-is-device" ]
clangCUDAIncludes = []
clangCUDADefines = [ "__CUDA_ARCH__" ]

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
  print "  --loop-unwind=X         Explore traces that pass through at most X loop heads"
  print "  --no-benign             Do not tolerate benign data races"
  print "  --no-infer              Turn off invariant inference"
  print "  --only-divergence       Only check for barrier divergence, not for races"
  print "  --only-intra-group      Do not check for inter-group races"
  print "  --verify                Run tool in verification mode"
  print "  --verbose               Show commands to run and use verbose output"
  print ""
  print "ADVANCED OPTIONS:"
  print "  --adversarial-abstraction  Completely abstract shared state, so that reads are"
  print "                          nondeterministic"
  print "  --array-equalities      Generate equality candidate invariants for array variables"
  print "  --boogie-opt=...        Specify option to be passed to Boogie"
  print "  --clang-opt=...         Specify option to be passed to CLANG"
  print "  --equality-abstraction  Make shared arrays nondeterministic, but consistent between"
  print "                          threads, at barriers"
  print "  --no-loop-predicate-invariants  Turn off automatic generation of loop invariants"
  print "                          related to predicates, which can be incorrect"
  print "  --no-smart-predication  Turn off smart predication"
  print "  --no-source-loc-infer   Turn off inference of source location information"
  print "  --no-uniformity-analysis  Turn off uniformity analysis"
  print "  --vcgen-opt=...         Specify option to be passed to be passed to VC generation"
  print "                          engine"
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
      GPUVerifyError("'" + a + "' has unknown file extension, supported file extensions are .cl (OpenCL) and .cu (CUDA)", ErrorCodes.COMMAND_LINE_ERROR)
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
    if o == "--loop-unwind":
      CommandLineOptions.mode = AnalysisMode.FINDBUGS
      try:
        if int(a) < 0:
          GPUVerifyError("negative value " + a + " provided as argument to --loop-unwind", ErrorCodes.COMMAND_LINE_ERROR) 
        CommandLineOptions.loopUnwindDepth = int(a)
      except ValueError:
        GPUVerifyError("non integer value '" + a + "' provided as argument to --loop-unwind", ErrorCodes.COMMAND_LINE_ERROR) 
    if o == "--no-benign":
      CommandLineOptions.noBenign = True
    if o == "--only-divergence":
      CommandLineOptions.onlyDivergence = True
    if o == "--only-intra-group":
      CommandLineOptions.onlyIntraGroup = True
    if o == "--adversarial-abstraction":
      if CommandLineOptions.equalityAbstraction:
        GPUVerifyError("illegal to specify both adversarial and equality abstractions", ErrorCodes.COMMAND_LINE_ERROR)
      CommandLineOptions.adversarialAbstraction = True
    if o == "--equality-abstraction":
      if CommandLineOptions.adversarialAbstraction:
        GPUVerifyError("illegal to specify both adversarial and equality abstractions", ErrorCodes.COMMAND_LINE_ERROR)
      CommandLineOptions.equalityAbstraction = True
    if o == "--no-loop-predicate-invariants":
      CommandLineOptions.noLoopPredicateInvariants = True
    if o == "--no-smart-predication":
      CommandLineOptions.noSmartPredication = True
    if o == "--no-source-loc-infer":
      CommandLineOptions.noSourceLocInfer = True
    if o == "--no-uniformity-analysis":
      CommandLineOptions.noUniformityAnalysis = True
    if o == "--clang-opt":
      raise(Exception)
    if o == "--vcgen-opt":
      CommandLineOptions.vcgenExtraOptions += str(a).split(" ")
    if o == "--boogie-opt":
      raise(Exception)


def processOpenCLOptions(opts, args):
  for o, a in opts:
    if o == "--local_size":
      if CommandLineOptions.groupSize != []:
        GPUVerifyError("illegal to define local_size multiple times", ErrorCodes.COMMAND_LINE_ERROR)
      CommandLineOptions.groupSize = processVector(a)
      for i in range(0, len(CommandLineOptions.groupSize)):
        if CommandLineOptions.groupSize[i] <= 0:
          GPUVerifyError("values specified for local_size dimensions must be positive", ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--num_groups":
      if CommandLineOptions.numGroups != []:
        raise Exception("illegal to define num_groups multiple times", ErrorCodes.COMMAND_LINE_ERROR)
      CommandLineOptions.numGroups = processVector(a)
      for i in range(0, len(CommandLineOptions.numGroups)):
        if CommandLineOptions.numGroups[i] <= 0:
          GPUVerifyError("values specified for num_groups dimensions must be positive", ErrorCodes.COMMAND_LINE_ERROR)

  if CommandLineOptions.groupSize == []:
    GPUVerifyError("work group size must be specified via --local_size=...", ErrorCodes.COMMAND_LINE_ERROR)
  if CommandLineOptions.numGroups == []:
    GPUVerifyError("number of work groups must be specified via --num_groups=...", ErrorCodes.COMMAND_LINE_ERROR)

def processCUDAOptions(opts, args):
  for o, a in opts:
    if o == "--blockDim":
      if CommandLineOptions.groupSize != []:
        GPUVerifyError("illegal to define blockDim multiple times", ErrorCodes.COMMAND_LINE_ERROR)
      CommandLineOptions.groupSize = processVector(a)
      for i in range(0, len(CommandLineOptions.groupSize)):
        if CommandLineOptions.groupSize[i] <= 0:
          GPUVerifyError("values specified for blockDim must be positive", ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--gridDim":
      if CommandLineOptions.numGroups != []:
        raise Exception("illegal to define gridDim multiple times", ErrorCodes.COMMAND_LINE_ERROR)
      CommandLineOptions.numGroups = processVector(a)
      for i in range(0, len(CommandLineOptions.numGroups)):
        if CommandLineOptions.numGroups[i] <= 0:
          GPUVerifyError("values specified for gridDim must be positive", ErrorCodes.COMMAND_LINE_ERROR)

  if CommandLineOptions.groupSize == []:
    GPUVerifyError("thread block size must be specified via --blockDim=...", ErrorCodes.COMMAND_LINE_ERROR)
  if CommandLineOptions.numGroups == []:
    GPUVerifyError("grid size must be specified via --gridDim=...", ErrorCodes.COMMAND_LINE_ERROR)

def main(argv=None):
  if argv is None:
    argv = sys.argv
  progname = argv[0]

  clangOptions = ClangOptions();

  try:
    opts, args = getopt.getopt(argv[1:],'D:I:h', 
             ['help', 'findbugs', 'verify', 'noinfer', 'verbose',
              'loop-unwind=', 'no-benign', 'only-divergence', 'only-intra-group', 
              'adversarial-abstraction', 'equality-abstraction', 'no-loop-predicate-invariants',
              'no-smart-predication', 'no-source-loc-infer', 'no-uniformity-analysis', 'clang-opt=', 
              'vcgen-opt=', 'boogie-opt=',
              'local_size=', 'num_groups=',
              'blockDim=', 'gridDim='
             ])
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
    clangOptions.options += clangCUDAOptions
    clangOptions.includes += clangCUDAIncludes
    clangOptions.defines += clangCUDADefines
    clangOptions.defines.append("__" + str(len(CommandLineOptions.groupSize)) + "D_THREAD_BLOCK")
    clangOptions.defines.append("__" + str(len(CommandLineOptions.numGroups)) + "D_GRID")
    clangOptions.defines += [ "__BLOCK_DIM_" + str(i) + "=" + str(CommandLineOptions.groupSize[i]) for i in range(0, len(CommandLineOptions.groupSize))]
    clangOptions.defines += [ "__GRID_DIM_" + str(i) + "=" + str(CommandLineOptions.numGroups[i]) for i in range(0, len(CommandLineOptions.numGroups))]

  clangOptions.options.append("-o")
  clangOptions.options.append(filename + ".bc")
  clangOptions.options.append(filename + ext)
  Verbose("Running clang")
  clangStdout, clangStderr, clangReturn = run(clangOptions.makeCommand())
  if clangReturn != 0:
    print clangStderr
    exit(ErrorCodes.CLANG_ERROR)
  if CommandLineOptions.verbose:
    print clangStdout

  optOptions = ToolOptions(llvmBinDir + "/opt");
  optOptions.options += [ "-mem2reg", "-globaldce", "-o", filename + ".opt.bc", filename + ".bc" ]
  Verbose("Running opt")
  optStdout, optStderr, optReturn = run(optOptions.makeCommand())
  if optReturn != 0:
    raise Exception(optStderr)
  if CommandLineOptions.verbose:
    print optStdout

  bugleOptions = ToolOptions(bugleBinDir + "/bugle");

  bugleOptions.options += [ "-l", "cl" if ext == ".cl" else "cu", "-o", filename + ".gbpl", filename + ".opt.bc"]

  Verbose("Running bugle")
  bugleStdout, bugleStderr, bugleReturn = run(bugleOptions.makeCommand())
  if bugleReturn != 0:
    raise Exception(bugleStderr)
  if CommandLineOptions.verbose:
    print bugleStdout

  gpuVerifyVCGenOptions = ToolOptions(gpuVerifyVCGenBinDir + "/GPUVerifyVCGen")
  gpuVerifyVCGenOptions.options += [ "/print:" + filename, filename + ".gbpl" ]
  if CommandLineOptions.adversarialAbstraction:
    gpuVerifyVCGenOptions.options += [ "/adversarialAbstraction" ]
  if CommandLineOptions.equalityAbstraction:
    gpuVerifyVCGenOptions.options += [ "/equalityAbstraction" ]
  if CommandLineOptions.noBenign:
    gpuVerifyVCGenOptions.options += [ "/noBenign" ]
  if CommandLineOptions.onlyDivergence:
    gpuVerifyVCGenOptions.options += [ "/onlyDivergence" ]
  if CommandLineOptions.onlyIntraGroup:
    gpuVerifyVCGenOptions.options += [ "/onlyIntraGroupRaceChecking" ]
  if CommandLineOptions.mode == AnalysisMode.FINDBUGS or (not CommandLineOptions.inference):
    gpuVerifyVCGenOptions.options += [ "/noInfer" ]
  if CommandLineOptions.noLoopPredicateInvariants:
    gpuVerifyVCGenOptions.options += [ "/noLoopPredicateInvariants" ]
  if CommandLineOptions.noSmartPredication:
    gpuVerifyVCGenOptions.options += [ "/noSmartPredication" ]
  if CommandLineOptions.noSourceLocInfer:
    gpuVerifyVCGenOptions.options += [ "/noSourceLocInfer" ]
  if CommandLineOptions.noUniformityAnalysis:
    gpuVerifyVCGenOptions.options += [ "/noUniformityAnalysis" ]
  gpuVerifyVCGenOptions.options += CommandLineOptions.vcgenExtraOptions

  Verbose("Running gpuverifyvcgen")
  gpuVerifyVCGenStdout, gpuVerifyVCGenStderr, gpuVerifyVCGenReturn = run(gpuVerifyVCGenOptions.makeCommand())
  if gpuVerifyVCGenReturn != 0:
    print gpuVerifyVCGenStdout + gpuVerifyVCGenStderr
    exit(ErrorCodes.GPUVERFYVCGEN_ERROR)
  if CommandLineOptions.verbose:
    print gpuVerifyVCGenStdout

  gpuVerifyBoogieDriverOptions = ToolOptions(gpuVerifyBoogieDriverBinDir + "/GPUVerifyBoogieDriver")
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
                                            "/z3exe:" + z3BinDir + "/z3",
                                            filename + ".bpl" ]
  if CommandLineOptions.mode == AnalysisMode.FINDBUGS:
    gpuVerifyBoogieDriverOptions.options += [ "/loopUnroll:" + str(CommandLineOptions.loopUnwindDepth) ]
  elif CommandLineOptions.inference:
    gpuVerifyBoogieDriverOptions.options += [ "/contractInfer" ]

  Verbose("Running gpuverifyboogiedriver")
  gpuVerifyBoogieDriverStdout, gpuVerifyBoogieDriverStderr, gpuVerifyBoogieDriverReturn = run(gpuVerifyBoogieDriverOptions.makeCommand())
  if gpuVerifyBoogieDriverReturn != 0:
    print gpuVerifyBoogieDriverStderr
    exit(ErrorCodes.BOOGIE_ERROR)

  if CommandLineOptions.verbose:
    print gpuVerifyBoogieDriverStdout
    print gpuVerifyBoogieDriverStderr

  if CommandLineOptions.mode == AnalysisMode.FINDBUGS:
    print "No defects were found while analysing: " + ", ".join(CommandLineOptions.sourceFiles)
    print "Notes:"
    print "- use --loop-unwind=N with N > " + str(CommandLineOptions.loopUnwindDepth) + " to search for deeper bugs"
    print "- re-run in verification mode to try to prove absence of defects"
  else:
    print "Verified: " + ", ".join(CommandLineOptions.sourceFiles)
    if not CommandLineOptions.onlyDivergence:
      print "- no data races within " + ("work groups" if CommandLineOptions.SL == SourceLanguage.OpenCL else "thread blocks")
      if not CommandLineOptions.onlyIntraGroup:
        print "- no data races between " + ("work groups" if CommandLineOptions.SL == SourceLanguage.OpenCL else "thread blocks")
    print "- no barrier divergence"
    print "- no assertion failures"
    print "(but absolutely no warranty provided)"

if __name__ == '__main__':
  sys.exit(main())
