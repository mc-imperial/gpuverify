#!/usr/bin/env python

import atexit
import getopt
import os
import subprocess
import sys
import time
import threading

bugleDir = sys.path[0] + "/bugle"
libclcDir = sys.path[0] + "/libclc"
llvmBinDir = sys.path[0] + "/bin"
bugleBinDir = sys.path[0] + "/bin"
gpuVerifyVCGenBinDir = sys.path[0] + "/bin"
gpuVerifyBoogieDriverBinDir = sys.path[0] + "/bin"
z3BinDir = sys.path[0] + "/bin"

""" Timing for the toolchain pipeline """
Timing = []

"""Horrible hack: WindowsError is not defined on UNIX systems, this works around that"""
try:
   WindowsError
except NameError:
   WindowsError = None


""" We support three analysis modes """
class AnalysisMode(object):
  """ This is the default mode.  Right now it is the same as VERIFY, 
      but in future this mode will run verification and bug-finding in parallel 
  """
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

clangCoreIncludes = [ bugleDir + "/include-blang" ]

clangCoreDefines = []

clangCoreOptions = [ "-target", "nvptx--bugle",
                     "-g",
                     "-gcolumn-info",
                     "-emit-llvm",
                     "-c" ]

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

""" Options for the tool """
class CommandLineOptions(object):
  SL = SourceLanguage.Unknown
  sourceFiles = []
  includes = clangCoreIncludes
  defines = clangCoreDefines
  clangOptions = clangCoreOptions
  optOptions = [ "-mem2reg", "-globaldce" ]
  gpuVerifyVCGenOptions = []
  gpuVerifyBoogieDriverOptions = [ "/nologo",
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
                                   "/z3exe:" + z3BinDir + "/z3"
                                 ]
  bugleOptions = []
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
  stopAtGbpl = False
  stopAtBpl = False
  time = False
  boogieTimeout=0
  keepTemps = False
  noThread2Asserts = False
  generateSmt2 = False
  noBarrierAccessChecks = False
  testsuite = False
  skip = { "clang": False,
           "opt": False,
           "bugle": False,
           "vcgen": False, }
  bugleLanguage = None

def SplitFilenameExt(f):
  filename, ext = os.path.splitext(f)
  if filename.endswith(".opt") and ext == ".bc":
    filename, unused_ext_ = os.path.splitext(filename)
    ext = ".opt.bc"
  return filename, ext

class Timeout(Exception):
    pass

class ToolWatcher(object):
  """ This class is used by run() to implement a timeout for tools.
  It uses threading.Timer to implement the timeout and provides
  a method for checking if the timeout occurred. It also provides a 
  method for cancelling the timeout.

  The class is reentrant
  """
   
  def __handleTimeOut(self):
    if self.popenObject.poll() == None :
      #Program is still running, let's kill it
      self.__killed=True
      self.popenObject.kill()

  """ Create a ToolWatcher instance with an existing "subprocess.Popen" instance
      and timeout.
  """
  def __init__(self,popenObject,timeout):
    """ Create ToolWatcher. This will start the timeout.
    """
    self.timeout=timeout
    self.popenObject=popenObject
    self.__killed=False

    self.timer=threading.Timer(self.timeout, self.__handleTimeOut)
    self.timer.start()
  
  """ Returns True if the timeout occurred """
  def timeOutOccured(self):
    return self.__killed
  
  """ Cancel the timeout. You must call this if your program wishes
      to exit else exit() will block waiting for this class's Thread
      (threading.Timer) to finish.
  """
  def cancelTimeout(self):
    self.timer.cancel()

def run(command,**kargs):
  """ Run a command. Additional keywords are supported after the
      positional arguments.
    
    timeout=X             : Set timeout in seconds
    timeoutErrorCode=CODE : The error code to return if a timeout occurs
  """

  
  popenargs={}
  if CommandLineOptions.verbose:
    print " ".join(command)

  else:
    popenargs['bufsize']=0
    popenargs['stdout']=subprocess.PIPE
    popenargs['stderr']=subprocess.PIPE
    
  killer=None
  proc = subprocess.Popen(command,**popenargs)
  if 'timeout' in kargs and 'timeoutErrorCode' in kargs:
    killer=ToolWatcher(proc,kargs['timeout'])
    
  try:
    stdout, stderr = proc.communicate()
    if type(killer) == ToolWatcher and killer.timeOutOccured():
      raise Timeout
          
  except KeyboardInterrupt:
    #Need to kill the timer if it exists else exit() will block until the timer finishes
    if killer != None:
      killer.cancelTimeout()
    exit(1)
    
  return stdout, stderr, proc.returncode

def RunTool(ToolName, Command, ErrorCode,**kargs):
  """ Run a tool. Additional keywords are supported after the
    positional arguments.
    
    timeout=X             : Set timeout in seconds
    timeoutErrorCode=CODE : The error code to return if a timeout occurs
  """
  Verbose("Running " + ToolName)
  try:
    start = time.time()
    stdout, stderr, returnCode = run(Command, **kargs)
    end = time.time()
  except Timeout:
    GPUVerifyError("Boogie timed out.", ErrorCodes.BOOGIE_TIMEOUT)
  except OSError as osError:
    print "Error while invoking " + ToolName + ": " + str(osError)
    exit(ErrorCode)
  except WindowsError as windowsError:
    print "Error while invoking " + ToolName + ": " + str(windowsError)
    exit(ErrorCode)
  if returnCode != 0:
    if stderr: print stderr
    exit(ErrorCode)
  if CommandLineOptions: Timing.append((ToolName, end-start))

class ErrorCodes(object):
  SUCCESS = 0
  COMMAND_LINE_ERROR = 1
  CLANG_ERROR = 2
  OPT_ERROR = 3
  BUGLE_ERROR = 4
  GPUVERIFYVCGEN_ERROR = 5
  BOOGIE_ERROR = 6
  BOOGIE_TIMEOUT=7

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
  print "  --time                  Show timing information"
  print "  --timeout=X             Allow Boogie component to run for X seconds before giving up."
  print "                          A timeout of 0 disables the timeout, this is the default."
  print ""
  print "ADVANCED OPTIONS:"
  print "  --adversarial-abstraction  Completely abstract shared state, so that reads are"
  print "                          nondeterministic"
  print "  --array-equalities      Generate equality candidate invariants for array variables"
  print "  --boogie-opt=...        Specify option to be passed to Boogie"
  print "  --bugle-lang=[cl|cu]    Bitcode language if passing in a bitcode file"
  print "  --clang-opt=...         Specify option to be passed to CLANG"
  print "  --equality-abstraction  Make shared arrays nondeterministic, but consistent between"
  print "                          threads, at barriers"
  print "  --gen-smt2              Generate smt2 file"
  print "  --keep-temps            Keep intermediate bc, gbpl and bpl files"
  print "  --no-barrier-access-checks      Turn off access checks for barrier invariants"
  print "  --no-loop-predicate-invariants  Turn off automatic generation of loop invariants"
  print "                          related to predicates, which can be incorrect"
  print "  --no-smart-predication  Turn off smart predication"
  print "  --no-source-loc-infer   Turn off inference of source location information"
  print "  --no-thread2-asserts    Turn off assertions for second thread"
  print "  --no-uniformity-analysis  Turn off uniformity analysis"
  print "  --stop-at-gbpl          Stop after generating gbpl"
  print "  --stop-at-bpl           Stop after generating bpl"
  print "  --testsuite             Testing testsuite program"
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

def GPUVerifyWarn(msg):
  print "GPUVerify: warning: " + msg

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
    filename, ext = SplitFilenameExt(a)
    if ext == ".cl":
      if CommandLineOptions.SL == SourceLanguage.CUDA:
        GPUVerifyError("illegal to pass both .cl and .cu files simultaneously", ErrorCodes.COMMAND_LINE_ERROR)
      CommandLineOptions.SL = SourceLanguage.OpenCL
    elif ext == ".cu":
      if CommandLineOptions.SL == SourceLanguage.OpenCL:
        GPUVerifyError("illegal to pass both .cl and .cu files simultaneously", ErrorCodes.COMMAND_LINE_ERROR)
      CommandLineOptions.SL = SourceLanguage.CUDA
    elif ext in [ ".bc", ".opt.bc", ".gbpl", ".bpl" ]:
      CommandLineOptions.skip["clang"] = True
      if ext in [        ".opt.bc", ".gbpl", ".bpl" ]:
        CommandLineOptions.skip["opt"] = True
      if ext in [                   ".gbpl", ".bpl" ]:
        CommandLineOptions.skip["bugle"] = True
      if ext in [                            ".bpl" ]:
        CommandLineOptions.skip["vcgen"] = True
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
      CommandLineOptions.defines.append(a)
    if o == "-I":
      CommandLineOptions.includes.append(a)
    if o == "--findbugs":
      CommandLineOptions.mode = AnalysisMode.FINDBUGS
    if o == "--verify":
      CommandLineOptions.mode = AnalysisMode.VERIFY
    if o in ("--noinfer", "--no-infer"):
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
    if o == "--keep-temps":
      CommandLineOptions.keepTemps = True
    if o == "--no-barrier-access-checks":
      CommandLineOptions.noBarrierAccessChecks = True
    if o == "--no-loop-predicate-invariants":
      CommandLineOptions.noLoopPredicateInvariants = True
    if o == "--no-smart-predication":
      CommandLineOptions.noSmartPredication = True
    if o == "--no-source-loc-infer":
      CommandLineOptions.noSourceLocInfer = True
    if o == "--no-uniformity-analysis":
      CommandLineOptions.noUniformityAnalysis = True
    if o == "--clang-opt":
      CommandLineOptions.clangOptions += str(a).split(" ")
    if o == "--vcgen-opt":
      CommandLineOptions.gpuVerifyVCGenOptions += str(a).split(" ")
    if o == "--boogie-opt":
      CommandLineOptions.gpuVerifyBoogieDriverOptions += str(a).split(" ")
    if o == "--stop-at-gbpl":
      CommandLineOptions.stopAtGbpl = True
    if o == "--stop-at-bpl":
      CommandLineOptions.stopAtBpl = True
    if o == "--time":
      CommandLineOptions.time = True
    if o == "--no-thread2-asserts":
      CommandLineOptions.noThread2Asserts = True
      CommandLineOptions.onlyDivergence = True
    if o == "--gen-smt2":
      CommandLineOptions.generateSmt2 = True
    if o == "--testsuite":
      CommandLineOptions.testsuite = True
    if o == "--bugle-lang":
      if a.lower() in ("cl", "cu"):
        CommandLineOptions.bugleLanguage = a.lower()
      else:
        GPUVerifyError("argument to --bugle-lang must be 'cl' or 'cu'", ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--timeout":
      try:
        CommandLineOptions.boogieTimeout = int(a)
        if CommandLineOptions.boogieTimeout < 0:
          raise ValueError
      except ValueError as e:
          GPUVerifyError("Invalid timeout \"" + a + "\"", ErrorCodes.COMMAND_LINE_ERROR)

def processOpenCLOptions(opts, args):
  for o, a in opts:
    if o == "--local_size":
      if CommandLineOptions.groupSize != []:
        GPUVerifyError("illegal to define local_size multiple times", ErrorCodes.COMMAND_LINE_ERROR)
      try:
        CommandLineOptions.groupSize = processVector(a)
      except ValueError:
        GPUVerifyError("argument to --local_size must be a (vector of) positive integer(s), found '" + a + "'", ErrorCodes.COMMAND_LINE_ERROR) 
      for i in range(0, len(CommandLineOptions.groupSize)):
        if CommandLineOptions.groupSize[i] <= 0:
          GPUVerifyError("values specified for local_size dimensions must be positive", ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--num_groups":
      if CommandLineOptions.numGroups != []:
        GPUVerifyError("illegal to define num_groups multiple times", ErrorCodes.COMMAND_LINE_ERROR)
      try:
        CommandLineOptions.numGroups = processVector(a)
      except ValueError:
        GPUVerifyError("argument to --num_groups must be a (vector of) positive integer(s), found '" + a + "'", ErrorCodes.COMMAND_LINE_ERROR) 
      for i in range(0, len(CommandLineOptions.numGroups)):
        if CommandLineOptions.numGroups[i] <= 0:
          GPUVerifyError("values specified for num_groups dimensions must be positive", ErrorCodes.COMMAND_LINE_ERROR)

  if CommandLineOptions.testsuite:
    if CommandLineOptions.groupSize or CommandLineOptions.numGroups:
      GPUVerifyWarn("local_size and num_groups flags ignored when using --testsuite")
    return

  if CommandLineOptions.groupSize == []:
    GPUVerifyError("work group size must be specified via --local_size=...", ErrorCodes.COMMAND_LINE_ERROR)
  if CommandLineOptions.numGroups == []:
    GPUVerifyError("number of work groups must be specified via --num_groups=...", ErrorCodes.COMMAND_LINE_ERROR)

def processCUDAOptions(opts, args):
  for o, a in opts:
    if o == "--blockDim":
      if CommandLineOptions.groupSize != []:
        GPUVerifyError("illegal to define blockDim multiple times", ErrorCodes.COMMAND_LINE_ERROR)
      try:
        CommandLineOptions.groupSize = processVector(a)
      except ValueError:
        GPUVerifyError("argument to --blockDim must be a (vector of) positive integer(s), found '" + a + "'", ErrorCodes.COMMAND_LINE_ERROR) 
      for i in range(0, len(CommandLineOptions.groupSize)):
        if CommandLineOptions.groupSize[i] <= 0:
          GPUVerifyError("values specified for blockDim must be positive", ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--gridDim":
      if CommandLineOptions.numGroups != []:
        GPUVerifyError("illegal to define gridDim multiple times", ErrorCodes.COMMAND_LINE_ERROR)
      try:
        CommandLineOptions.numGroups = processVector(a)
      except ValueError:
        GPUVerifyError("argument to --gridDim must be a (vector of) positive integer(s), found '" + a + "'", ErrorCodes.COMMAND_LINE_ERROR) 
      for i in range(0, len(CommandLineOptions.numGroups)):
        if CommandLineOptions.numGroups[i] <= 0:
          GPUVerifyError("values specified for gridDim must be positive", ErrorCodes.COMMAND_LINE_ERROR)

  if CommandLineOptions.testsuite:
    if CommandLineOptions.groupSize or CommandLineOptions.numGroups:
      GPUVerifyWarn("blockDim and gridDim flags ignored when using --testsuite")
    return

  if CommandLineOptions.groupSize == []:
    GPUVerifyError("thread block size must be specified via --blockDim=...", ErrorCodes.COMMAND_LINE_ERROR)
  if CommandLineOptions.numGroups == []:
    GPUVerifyError("grid size must be specified via --gridDim=...", ErrorCodes.COMMAND_LINE_ERROR)

def main(argv=None):
  if argv is None:
    argv = sys.argv
  progname = argv[0]

  try:
    opts, args = getopt.getopt(argv[1:],'D:I:h', 
             ['help', 'findbugs', 'verify', 'noinfer', 'no-infer', 'verbose',
              'loop-unwind=', 'no-benign', 'only-divergence', 'only-intra-group', 
              'adversarial-abstraction', 'equality-abstraction', 
              'no-barrier-access-checks', 'no-loop-predicate-invariants',
              'no-smart-predication', 'no-source-loc-infer', 'no-uniformity-analysis', 'clang-opt=', 
              'vcgen-opt=', 'boogie-opt=',
              'local_size=', 'num_groups=',
              'blockDim=', 'gridDim=',
              'stop-at-gbpl', 'stop-at-bpl', 'time', 'keep-temps',
              'no-thread2-asserts', 'gen-smt2', 'testsuite', 'bugle-lang=','timeout='
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

  filename, ext = SplitFilenameExt(args[0])

  if ext == ".cl":
    CommandLineOptions.clangOptions += clangOpenCLOptions
    if CommandLineOptions.testsuite:
      rmFlags = [ "-include", "opencl.h" ]
      CommandLineOptions.clangOptions = [ i for i in CommandLineOptions.clangOptions if i not in rmFlags ]
    CommandLineOptions.includes += clangOpenCLIncludes
    CommandLineOptions.defines += clangOpenCLDefines
    if not CommandLineOptions.testsuite:
      CommandLineOptions.defines.append("__" + str(len(CommandLineOptions.groupSize)) + "D_WORK_GROUP")
      CommandLineOptions.defines.append("__" + str(len(CommandLineOptions.numGroups)) + "D_GRID")
      CommandLineOptions.defines += [ "__LOCAL_SIZE_" + str(i) + "=" + str(CommandLineOptions.groupSize[i]) for i in range(0, len(CommandLineOptions.groupSize))]
      CommandLineOptions.defines += [ "__NUM_GROUPS_" + str(i) + "=" + str(CommandLineOptions.numGroups[i]) for i in range(0, len(CommandLineOptions.numGroups))]

  elif ext == ".cu":
    CommandLineOptions.clangOptions += clangCUDAOptions
    CommandLineOptions.includes += clangCUDAIncludes
    CommandLineOptions.defines += clangCUDADefines
    if not CommandLineOptions.testsuite:
      CommandLineOptions.defines.append("__" + str(len(CommandLineOptions.groupSize)) + "D_THREAD_BLOCK")
      CommandLineOptions.defines.append("__" + str(len(CommandLineOptions.numGroups)) + "D_GRID")
      CommandLineOptions.defines += [ "__BLOCK_DIM_" + str(i) + "=" + str(CommandLineOptions.groupSize[i]) for i in range(0, len(CommandLineOptions.groupSize))]
      CommandLineOptions.defines += [ "__GRID_DIM_" + str(i) + "=" + str(CommandLineOptions.numGroups[i]) for i in range(0, len(CommandLineOptions.numGroups))]

  # Intermediate filenames
  bcFilename = filename + '.bc'
  optFilename = filename + '.opt.bc'
  gbplFilename = filename + '.gbpl'
  bplFilename = filename + '.bpl'
  locFilename = filename + '.loc'
  smt2Filename = filename + '.smt2'
  if not CommandLineOptions.keepTemps:
    inputFilename = filename + ext
    def DeleteFile(filename):
      """ Delete the filename if it exists; but don't delete the original input """
      if filename == inputFilename: return
      try: os.remove(filename)
      except OSError: pass
    atexit.register(DeleteFile, bcFilename)
    atexit.register(DeleteFile, optFilename)
    if not CommandLineOptions.stopAtGbpl: atexit.register(DeleteFile, gbplFilename)
    if not CommandLineOptions.stopAtBpl: atexit.register(DeleteFile, bplFilename)
    if not CommandLineOptions.stopAtBpl: atexit.register(DeleteFile, locFilename)

  CommandLineOptions.clangOptions.append("-o")
  CommandLineOptions.clangOptions.append(bcFilename)
  CommandLineOptions.clangOptions.append(filename + ext)

  CommandLineOptions.optOptions += [ "-o", optFilename, bcFilename ]

  if ext in [ ".cl", ".cu" ]:
    CommandLineOptions.bugleOptions += [ "-l", "cl" if ext == ".cl" else "cu", "-o", gbplFilename, optFilename ]
  elif not CommandLineOptions.skip['bugle']:
    lang = CommandLineOptions.bugleLanguage
    if not lang: # try to infer
      try:
        proc = subprocess.Popen([ llvmBinDir + "/llvm-nm", filename + ext ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if "get_local_size" in stdout: lang = 'cl'
        if "blockDim" in stdout: lang = 'cu'
      except: pass
    if not lang:
      GPUVerifyError("must specify --bugle-lang=[cl|cu] when given a bitcode .bc file", ErrorCodes.COMMAND_LINE_ERROR)
    assert lang in [ "cl", "cu" ]
    CommandLineOptions.bugleOptions += [ "-l", lang, "-o", gbplFilename, optFilename ]

  if CommandLineOptions.adversarialAbstraction:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/adversarialAbstraction" ]
  if CommandLineOptions.equalityAbstraction:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/equalityAbstraction" ]
  if CommandLineOptions.noBenign:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/noBenign" ]
  if CommandLineOptions.onlyDivergence:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/onlyDivergence" ]
  if CommandLineOptions.onlyIntraGroup:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/onlyIntraGroupRaceChecking" ]
  if CommandLineOptions.mode == AnalysisMode.FINDBUGS or (not CommandLineOptions.inference):
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/noInfer" ]
  if CommandLineOptions.noBarrierAccessChecks:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/noBarrierAccessChecks" ]
  if CommandLineOptions.noLoopPredicateInvariants:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/noLoopPredicateInvariants" ]
  if CommandLineOptions.noSmartPredication:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/noSmartPredication" ]
  if CommandLineOptions.noSourceLocInfer:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/noSourceLocInfer" ]
  if CommandLineOptions.noUniformityAnalysis:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/noUniformityAnalysis" ]
  CommandLineOptions.gpuVerifyVCGenOptions += [ "/print:" + filename, gbplFilename ] #< ignore .bpl suffix

  if CommandLineOptions.mode == AnalysisMode.FINDBUGS:
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/loopUnroll:" + str(CommandLineOptions.loopUnwindDepth) ]
  elif CommandLineOptions.inference:
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/contractInfer" ]

  if CommandLineOptions.generateSmt2:
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/proverLog:" + smt2Filename ]
  CommandLineOptions.gpuVerifyBoogieDriverOptions += [ bplFilename ]

  """ RUN CLANG """
  if not CommandLineOptions.skip["clang"]:
    RunTool("clang", 
             [llvmBinDir + "/clang"] + 
             CommandLineOptions.clangOptions + 
             [("-I" + str(o)) for o in CommandLineOptions.includes] + 
             [("-D" + str(o)) for o in CommandLineOptions.defines],
             ErrorCodes.CLANG_ERROR)

  """ RUN OPT """
  if not CommandLineOptions.skip["opt"]:
    RunTool("opt",
            [llvmBinDir + "/opt"] + 
            CommandLineOptions.optOptions,
            ErrorCodes.OPT_ERROR)

  """ RUN BUGLE """
  if not CommandLineOptions.skip["bugle"]:
    RunTool("bugle",
            [bugleBinDir + "/bugle"] +
            CommandLineOptions.bugleOptions,
            ErrorCodes.BUGLE_ERROR)

  if CommandLineOptions.stopAtGbpl: return 0

  """ RUN GPUVERIFYVCGEN """
  if not CommandLineOptions.skip["vcgen"]:
    RunTool("gpuverifyvcgen",
            (["mono"] if os.name == "posix" else []) +
            [gpuVerifyVCGenBinDir + "/GPUVerifyVCGen.exe"] +
            CommandLineOptions.gpuVerifyVCGenOptions,
            ErrorCodes.GPUVERIFYVCGEN_ERROR)

    if CommandLineOptions.noThread2Asserts:
      f = open(bplFilename)
      lines = f.readlines()
      f.close()
      f = open(bplFilename, 'w')
      for line in lines:
        if 'thread 2' not in line: f.write(line)
      f.close()

  if CommandLineOptions.stopAtBpl: return 0

  """ RUN GPUVERIFYBOOGIEDRIVER """
  timeoutArguments={}
  if CommandLineOptions.boogieTimeout > 0:
    timeoutArguments['timeout']= CommandLineOptions.boogieTimeout
    timeoutArguments['timeoutErrorCode']=ErrorCodes.BOOGIE_TIMEOUT
    
  RunTool("gpuverifyboogiedriver",
          (["mono"] if os.name == "posix" else []) +
          [gpuVerifyBoogieDriverBinDir + "/GPUVerifyBoogieDriver.exe"] +
          CommandLineOptions.gpuVerifyBoogieDriverOptions,
          ErrorCodes.BOOGIE_ERROR,
          **timeoutArguments)

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
  return 0

if __name__ == '__main__':
  returnCode = main()
  if CommandLineOptions.time:
    print "Timing information:"
    pad = max([ len(tool) for tool,t in Timing ])
    for (tool, t) in Timing:
      print "- %s : %.03f secs" % (tool.ljust(pad), t)
  sys.exit(returnCode)
