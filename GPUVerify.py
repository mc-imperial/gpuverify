#!/usr/bin/env python2.7

import atexit
import getopt
import os
import signal
import subprocess
import sys
import timeit
import threading
import multiprocessing # Only for determining number of CPU cores available
import getversion
import pprint

"""
    Print an error message to stderr and exit with status code.

    msg : A String describing the error
    code : An error code (integer) from ErrorCodes class
"""
def GPUVerifyError(msg, code):
  # Determine string for error code
  codeString = None
  for cs in [ x for x in dir(ErrorCodes) if not x.startswith('_') ]:
    if getattr(ErrorCodes, cs) == code:
      codeString = cs

  if codeString == None:
    codeString = 'UNKNOWN'

  sys.stderr.write('GPUVerify: {} error: {}\n'.format(codeString, msg))
  sys.exit(code)

class ErrorCodes(object):
  SUCCESS = 0
  COMMAND_LINE_ERROR = 1
  CLANG_ERROR = 2
  OPT_ERROR = 3
  BUGLE_ERROR = 4
  GPUVERIFYVCGEN_ERROR = 5
  BOOGIE_ERROR = 6
  BOOGIE_TIMEOUT = 7
  CTRL_C = 8
  GPUVERIFYVCGEN_TIMEOUT = 9
  CONFIGURATION_ERROR = 10

# Try to import the paths need for GPUVerify's tools
try:
  import gvfindtools
  # Initialise the paths (only needed for deployment version of gvfindtools.py)
  gvfindtools.init(sys.path[0])
except ImportError:
  GPUVerifyError('Cannot find \'gvfindtools.py\'.'
                 ' Did you forget to create it from a template?',
                 ErrorCodes.CONFIGURATION_ERROR)

""" Timing for the toolchain pipeline """
Timing = []

""" WindowsError is not defined on UNIX systems, this works around that """
try:
   WindowsError
except NameError:
   WindowsError = None

""" Horrible hack: Patch sys.exit() so we can get the exitcode in atexit callbacks """
class ExitHook(object):
  def __init__(self):
    self.code = None

  def hook(self):
    self.realExit = sys.exit
    sys.exit = self.exit

  def exit(self, code=0):
    self.code = code
    self.realExit(code)

exitHook = ExitHook()
exitHook.hook()

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

clangCoreIncludes = [ gvfindtools.bugleSrcDir + "/include-blang" ]

clangCoreDefines = []

clangCoreOptions = [ "-target", "nvptx--bugle",
                     "-Wall",
                     "-g",
                     "-gcolumn-info",
                     "-emit-llvm",
                     "-c"
                   ]

if os.name == "posix":

  if os.path.isfile(gvfindtools.bugleBinDir \
                    + "/libbugleInlineCheckPlugin.so"):
    bugleInlineCheckPlugin = gvfindtools.bugleBinDir \
                             + "/libbugleInlineCheckPlugin.so"
  elif os.path.isfile(gvfindtools.bugleBinDir \
                      + "/libbugleInlineCheckPlugin.dylib"):
    bugleInlineCheckPlugin = gvfindtools.bugleBinDir \
                             + "/libbugleInlineCheckPlugin.dylib"
  else:
    GPUVerifyError('Could not find Bugle Inline Check plugin', ErrorCodes.CONFIGURATION_ERROR)

  clangInlineOptions = [ "-Xclang", "-load",
                         "-Xclang", bugleInlineCheckPlugin,
                         "-Xclang", "-add-plugin",
                         "-Xclang", "inline-check"
                       ]
else:
  clangInlineOptions = []

clangOpenCLOptions = [ "-Xclang", "-cl-std=CL1.2",
                       "-O0",
                       "-Xclang", "-mlink-bitcode-file",
                       "-Xclang", gvfindtools.libclcInstallDir + "/lib/clc/nvptx--bugle.bc",
                       "-include", "opencl.h"
                     ]
clangOpenCLIncludes = [ gvfindtools.libclcInstallDir + "/include" ]
clangOpenCLDefines = [ "cl_khr_fp64",
                       "cl_clang_storage_class_specifiers",
                       "__OPENCL_VERSION__"
                     ]

clangCUDAOptions = [ "-Xclang", "-fcuda-is-device",
                       "-include", "cuda.h"
                   ]
clangCUDAIncludes = [ gvfindtools.libclcInstallDir + "/include" ]
clangCUDADefines = [ "__CUDA_ARCH__" ]

""" Options for the tool """
class CommandLineOptions(object):
  SL = SourceLanguage.Unknown
  sourceFiles = [] # The OpenCL or CUDA files to be processed
  includes = clangCoreIncludes
  defines = clangCoreDefines
  clangOptions = clangCoreOptions
  optOptions = [ "-mem2reg", "-globaldce" ]
  defaultOptions = [ "/nologo", "/typeEncoding:m", "/mv:-",
                     "/doModSetAnalysis", "/useArrayTheory",
                     "/doNotUseLabels", "/enhancedErrorMessages:1"
                   ]
  gpuVerifyVCGenOptions = []
  gpuVerifyCruncherOptions = []
  gpuVerifyBoogieDriverOptions = []
  bugleOptions = []
  mode = AnalysisMode.ALL
  debugging = False
  verbose = False
  silent = False
  groupSize = []
  numGroups = []
  adversarialAbstraction = False
  equalityAbstraction = False
  loopUnwindDepth = 2
  noBenign = False
  onlyDivergence = False
  onlyIntraGroup = False
  onlyLog = False
  noLoopPredicateInvariants = False
  noSmartPredication = False
  noSourceLocInfer = False
  noUniformityAnalysis = False
  inference = True
  invInferConfigFile = "inference.cfg"
  stagedInference = False
  parallelInference = False
  dynamicAnalysis = False
  scheduling = "all-together"
  inferInfo = False
  debuggingHoudini = False
  stopAtOpt = False
  stopAtGbpl = False
  stopAtBpl = False
  stopAtInv = False
  time = False
  timeCSVLabel = None
  boogieMemout=0
  vcgenTimeout=0
  cruncherTimeout=300
  boogieTimeout=300
  keepTemps = False
  mathInt = False
  asymmetricAsserts = False
  generateSmt2 = False
  noBarrierAccessChecks = False
  noConstantWriteChecks = False
  callSiteAnalysis = False
  testsuite = False
  warpSync = False
  warpSize = 32
  atomic = "rw"
  noRefinedAtomics = False
  solver = "z3"
  logic = "QF_ALL_SUPPORTED"
  skip = { "clang": False,
           "opt": False,
           "bugle": False,
           "vcgen": False, 
           "cruncher": False }
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
      # Program is still running, let's kill it
      self.__killed=True
      self.popenObject.terminate()

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

def run(command,timeout=0):
  """ Run a command with an optional timeout. A timeout of zero
      implies no timeout.
  """
  popenargs={}
  if CommandLineOptions.verbose:
    print " ".join(command)
  else:
    popenargs['bufsize']=0
    popenargs['stdout']=subprocess.PIPE
    popenargs['stderr']=subprocess.STDOUT

  killer=None
  def cleanupKiller():
    if killer!=None:
      killer.cancelTimeout()

  proc = subprocess.Popen(command,**popenargs)
  if timeout > 0:
    killer=ToolWatcher(proc,timeout)
  try:
    stdout, stderr = proc.communicate()
    if killer != None and killer.timeOutOccured():
      raise Timeout
  except KeyboardInterrupt:
    cleanupKiller()
    proc.wait()
    sys.exit(ErrorCodes.CTRL_C)
  finally:
    #Need to kill the timer if it exists else exit() will block until the timer finishes
    cleanupKiller()

  return stdout, stderr, proc.returncode

def RunTool(ToolName, Command, ErrorCode,timeout=0,timeoutErrorCode=None):
  """ Run a tool.
      If the timeout is set to 0 then there will be no timeout.
      If the timeout is > 0 then timeoutErrorCode MUST be set!
  """
  Verbose("Running " + ToolName)
  try:
    start = timeit.default_timer()
    stdout, stderr, returnCode = run(Command, timeout)
    end = timeit.default_timer()
  except Timeout:
    if CommandLineOptions.time:
      Timing.append((ToolName, timeout))
    GPUVerifyError(ToolName + " timed out.  Use --timeout=N with N > " + str(timeout) + " to increase timeout, or --timeout=0 to disable timeout.", timeoutErrorCode)
  except (OSError,WindowsError) as e:
    GPUVerifyError("While invoking " + ToolName + ": " + str(e) + "\nWith command line args:\n" + pprint.pformat(Command),ErrorCode)

  if CommandLineOptions.time:
    Timing.append((ToolName, end-start))
  if returnCode != 0:
    if stdout: print >> sys.stderr, stdout
    if stderr: print >> sys.stderr, stderr
    sys.exit(ErrorCode)

def showVersionAndExit():
  """ This will check if using gpuverify from development directory.
      If so this will invoke Mercurial to find out version information.
      If this is a deployed version we will try to read the version from
      a file instead
  """

  print(getversion.getVersionString())
  sys.exit(ErrorCodes.SUCCESS)

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
  print "  --memout=X              Give Boogie a hard memory limit of X megabytes."
  print "                          A memout of 0 disables the memout. The default is " + str(CommandLineOptions.boogieMemout) + " megabytes."
  print "  --no-benign             Do not tolerate benign data races"
  print "  --only-divergence       Only check for barrier divergence, not for races"
  print "  --only-intra-group      Do not check for inter-group races"
  print "  --time                  Show timing information"
  print "  --timeout=X             Allow Boogie to run for X seconds before giving up."
  print "                          A timeout of 0 disables the timeout. The default is " + str(CommandLineOptions.boogieTimeout) + " seconds."
  print "  --verify                Run tool in verification mode"
  print "  --verbose               Show commands to run and use verbose output"
  print "  --version               Show version information."
  print ""
  print "ADVANCED OPTIONS:"
  print "  --adversarial-abstraction  Completely abstract shared state, so that reads are"
  print "                          nondeterministic"
  print "  --array-equalities      Generate equality candidate invariants for array variables"
  print "  --asymmetric-asserts    Emit assertions only for first thread.  Sound, and may lead"
  print "                          to faster verification, but can yield false positives"
  print "  --boogie-file=X.bpl     Specify a supporting .bpl file to be used during verification"
  print "  --boogie-opt=...        Specify option to be passed to Boogie"
  print "  --bugle-lang=[cl|cu]    Bitcode language if passing in a bitcode file"
  print "  --bugle-opt=...         Specify option to be passed to Bugle"
  print "  --call-site-analysis    Turn on call site analysis"
  print "  --clang-opt=...         Specify option to be passed to CLANG"
  print "  --debug                 Enable debugging of GPUVerify components: exceptions will"
  print "                          not be suppressed"
  print "  --equality-abstraction  Make shared arrays nondeterministic, but consistent between"
  print "                          threads, at barriers"
  print "  --gen-smt2              Generate smt2 file"
  print "  --keep-temps            Keep intermediate bc, gbpl and bpl files"
  print "  --math-int              Represent integer types using mathematical integers"
  print "                          instead of bit-vectors"
  print "  --no-annotations        Ignore all source-level annotations"
  print "  --only-requires         Ignore all source-level annotations except for requires"
  print "  --no-barrier-access-checks      Turn off access checks for barrier invariants"
  print "  --no-constant-write-checks      Turn off access checks for writes to constant space"
  print "  --no-loop-predicate-invariants  Turn off automatic generation of loop invariants"
  print "                          related to predicates, which can be incorrect"
  print "  --no-smart-predication  Turn off smart predication"
  print "  --no-source-loc-infer   Turn off inference of source location information"
  print "  --no-uniformity-analysis  Turn off uniformity analysis"
  print "  --only-log              Log accesses to arrays, but do not check for races.  This"
  print "                          can be useful for determining access pattern invariants"
  print "  --silent                Silent on success; only show errors/timing"
  print "  --stop-at-opt           Stop after LLVM optimization pass"
  print "  --stop-at-gbpl          Stop after generating gbpl"
  print "  --stop-at-bpl           Stop after generating bpl"
  print "  --stop-at-inv           Stop after generating an annotated with invariants bpl"
  print "  --time-as-csv=label     Print timing as CSV row with label"
  print "  --testsuite             Testing testsuite program"
  print "  --vcgen-timeout=X       Allow VCGen to run for X seconds."
  print "  --vcgen-opt=...         Specify option to be passed to be passed to VC generation"
  print "                          engine"
  print "  --warp-sync=X           Synchronize threads within warps, sized X, defaulting to 32"
  print "  --atomic=X              Check atomics as racy against reads (r), writes(w), both(rw), or none(none)"
  print "                          (default is --atomic=rw)"
  print "  --no-refined-atomics    Don't do abstraction refinement on the return values from atomics"
  print "  --solver=X              Choose which SMT Theorem Prover to use in the backend."
  print "                          Available options: 'Z3' or 'cvc4' (default is 'Z3')"
  print "  --logic=X               Define the logic to be used by the CVC4 SMT solver backend"
  print "                          (default is QF_ALL_SUPPORTED)"
  print ""
  print "INVARIANT INFERENCE OPTIONS:"
  print "  --no-infer              Turn off invariant inference"
  print "  --infer-timeout=X       Allow GPUVerifyCruncher to run for X seconds."
  print "  --staged-inference      Perform invariant inference in stages; this can boost"
  print "                          performance for complex kernels (but this is not guaranteed)"
  print "  --parallel-inference    Use multiple solver instances in parallel to accelerate invariant"
  print "                          inference (but this is not guaranteed)"
  print "  --dynamic-analysis      Use dynamic analysis to falsify invariants."
  print "  --scheduling=X          Choose a parallel scheduling strategy from the following: 'all-together',"
  print "                          'unsound-first', 'dynamic-first' or 'phased'. The 'all-together' strategy"
  print "                          executes all refutation engines together. The 'unsound-first' strategy"
  print "                          executes any unsound engines (either static or dynamic) before the sound"
  print "                          engines. The 'dynamic-first' strategy executes any dynamic engines before"
  print "                          the static engines. The 'phased' strategy executes first any dynamic engines,"
  print "                          then any unsound static engines and then the sound static engines. The default"
  print "                          scheduling is 'all-together'."
  print "  --infer-config-file=X.cfg       Specify a custom configuration file to be used"
  print "                          during invariant inference"
  print "  --infer-info            Prints information about the inference process."
  print ""
  print "OPENCL OPTIONS:"
  print "  --local_size=X          Specify whether work-group is 1D, 2D"
  print "              =[X,Y]      or 3D and specify size for each"
  print "              =[X,Y,Z]    dimension"
  print "  --num_groups=X          Specify whether grid of work-groups is"
  print "              =[X,Y]      1D, 2D or 3D and specify size for each"
  print "              =[X,Y,Z]    dimension"
  print ""
  print "CUDA OPTIONS"
  print "  --blockDim=X            Specify whether thread block is 1D, 2D"
  print "              =[X,Y]      or 3D and specify size for each"
  print "              =[X,Y,Z]    dimension"
  print "  --gridDim=X             Specify whether grid of thread blocks is"
  print "              =[X,Y]      1D, 2D or 3D and specify size for each"
  print "              =[X,Y,Z]    dimension"
  sys.exit(0)

def processVector(vector):
  vector = vector.strip()
  if vector[0] == '[' and vector[-1] == ']':
    return map(int, vector[1:-1].split(","))
  else:
    return map(int, vector.split(","))

def GPUVerifyWarn(msg):
  print "GPUVerify: warning: " + msg

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
    elif ext in [ ".bc", ".opt.bc", ".gbpl", ".bpl", ".inv.bpl" ]:
      CommandLineOptions.skip["clang"] = True
      if ext in [        ".opt.bc", ".gbpl", ".bpl", ".inv.bpl" ]:
        CommandLineOptions.skip["opt"] = True
      if ext in [                   ".gbpl", ".bpl", ".inv.bpl" ]:
        CommandLineOptions.skip["bugle"] = True
      if ext in [                            ".bpl", ".inv.bpl" ]:
        CommandLineOptions.skip["vcgen"] = True
      if ext in [                                    ".inv.bpl" ]:
        CommandLineOptions.skip["cruncher"] = True
    else:
      GPUVerifyError("'" + a + "' has unknown file extension, supported file extensions are .cl (OpenCL) and .cu (CUDA)", ErrorCodes.COMMAND_LINE_ERROR)
    CommandLineOptions.sourceFiles.append(a)

def showHelpIfRequested(opts):
  for o, a in opts:
    if o == "--help" or o == "-h":
      showHelpAndExit()

def showVersionIfRequested(opts):
  for o, a in opts:
    if o == "--version":
      showVersionAndExit()

def processGeneralOptions(opts, args):
  # All options that can be processed without resulting in an error go
  # in this loop. Some of these we want to handle even when some other
  # option results in an error, e.g., the time related options.
  for o, a in opts:
    if o == "-D":
      CommandLineOptions.defines.append(a)
    if o == "-I":
      CommandLineOptions.includes.append(a)
    if o == "--debug":
      CommandLineOptions.debugging = True
    if o == "--findbugs":
      CommandLineOptions.mode = AnalysisMode.FINDBUGS
    if o == "--verify":
      CommandLineOptions.mode = AnalysisMode.VERIFY
    if o in ("--noinfer", "--no-infer"):
      CommandLineOptions.inference = False
    if o == "--verbose":
      CommandLineOptions.verbose = True
    if o == "--silent":
      CommandLineOptions.silent = True
    if o == "--no-benign":
      CommandLineOptions.noBenign = True
    if o == "--only-divergence":
      CommandLineOptions.onlyDivergence = True
    if o == "--only-intra-group":
      CommandLineOptions.onlyIntraGroup = True
    if o == "--only-log":
      CommandLineOptions.onlyLog = True
    if o == "--keep-temps":
      CommandLineOptions.keepTemps = True
    if o == "--math-int":
      CommandLineOptions.mathInt = True
    if o in ("--no-annotations", "--only-requires"):
      # Must be added after include of opencl or cuda header
      noAnnotationsHeader = [ "-include", "annotations/no_annotations.h" ]
      clangOpenCLOptions.extend(noAnnotationsHeader)
      clangCUDAOptions.extend(noAnnotationsHeader)
      if o == "--only-requires":
        clangOpenCLDefines.append("ONLY_REQUIRES")
        clangCUDADefines.append("ONLY_REQUIRES")
    if o == "--no-barrier-access-checks":
      CommandLineOptions.noBarrierAccessChecks = True
    if o == "--no-constant-write-checks":
      CommandLineOptions.noConstantWriteChecks = True
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
      CommandLineOptions.gpuVerifyCruncherOptions += str(a).split(" ")
      CommandLineOptions.gpuVerifyBoogieDriverOptions += str(a).split(" ")
    if o == "--bugle-opt":
      CommandLineOptions.bugleOptions += str(a).split(" ")
    if o == "--staged-inference":
      CommandLineOptions.stagedInference = True
    if o == "--parallel-inference":
      CommandLineOptions.parallelInference = True
    if o == "--dynamic-analysis":
      CommandLineOptions.dynamicAnalysis = True
    if o == "--infer-info":
      CommandLineOptions.inferInfo = True
    if o == "--debug-houdini":
      CommandLineOptions.debuggingHoudini = True
    if o == "--stop-at-opt":
      CommandLineOptions.stopAtOpt = True
    if o == "--stop-at-gbpl":
      CommandLineOptions.stopAtGbpl = True
    if o == "--stop-at-bpl":
      CommandLineOptions.stopAtBpl = True
    if o == "--stop-at-inv":
      CommandLineOptions.stopAtInv = True
    if o == "--time":
      CommandLineOptions.time = True
    if o == "--time-as-csv":
      CommandLineOptions.time = True
      CommandLineOptions.timeCSVLabel = a
    if o == "--asymmetric-asserts":
      CommandLineOptions.asymmetricAsserts = True
    if o == "--gen-smt2":
      CommandLineOptions.generateSmt2 = True
    if o == "--testsuite":
      CommandLineOptions.testsuite = True
    if o == "--no-refined-atomics":
      CommandLineOptions.noRefinedAtomics = True
    if o == "--call-site-analysis":
      CommandLineOptions.callSiteAnalysis = True

  # All options whose processing can result in an error go in this loop.
  # See also the comment above the previous loop.
  for o, a in opts:
    if o == "--loop-unwind":
      CommandLineOptions.mode = AnalysisMode.FINDBUGS
      try:
        if int(a) < 0:
          GPUVerifyError("negative value " + a + " provided as argument to --loop-unwind", ErrorCodes.COMMAND_LINE_ERROR)
        CommandLineOptions.loopUnwindDepth = int(a)
      except ValueError:
        GPUVerifyError("non integer value '" + a + "' provided as argument to --loop-unwind", ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--memout":
      try:
        CommandLineOptions.boogieMemout = int(a)
        if CommandLineOptions.boogieMemout < 0:
          raise ValueError
      except ValueError as e:
          GPUVerifyError("Invalid memout \"" + a + "\"", ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--adversarial-abstraction":
      if CommandLineOptions.equalityAbstraction:
        GPUVerifyError("illegal to specify both adversarial and equality abstractions", ErrorCodes.COMMAND_LINE_ERROR)
      CommandLineOptions.adversarialAbstraction = True
    if o == "--equality-abstraction":
      if CommandLineOptions.adversarialAbstraction:
        GPUVerifyError("illegal to specify both adversarial and equality abstractions", ErrorCodes.COMMAND_LINE_ERROR)
      CommandLineOptions.equalityAbstraction = True
    if o == "--warp-sync":
      CommandLineOptions.warpSync = True
      try:
        if int(a) < 0 :
          GPUVerifyError("negative value " + a + " provided as argument to --warp-sync", ErrorCodes.COMMAND_LINE_ERROR)
        CommandLineOptions.warpSize = int(a)
      except ValueError:
        GPUVerifyError("non integer value '" + a + "' provided as argument to --warp-sync",ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--atomic":
      if a.lower() in ("r","w","rw","none"):
        CommandLineOptions.atomic = a.lower()
      else:
        GPUVerifyError("argument to --atomic must be 'r','w','rw', or 'none'", ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--solver":
      if a.lower() in ("z3","cvc4"):
        CommandLineOptions.solver = a.lower()
      else:
        GPUVerifyError("argument to --solver must be 'Z3' or 'CVC4'", ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--scheduling":
      if a.lower() in ("all-together","unsound-first","dynamic-first","phased"):
        CommandLineOptions.scheduling = a.lower()
      else:
        GPUVerifyError("argument to --scheduling must be 'all-together', 'unsound-first', 'dynamic-first' or'phased'", ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--logic":
      if a.upper() in ("ALL_SUPPORTED","QF_ALL_SUPPORTED"):
        CommandLineOptions.logic = a.upper()
      else:
        GPUVerifyError("argument to --logic must be 'ALL_SUPPORTED' or 'QF_ALL_SUPPORTED'", ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--bugle-lang":
      if a.lower() in ("cl", "cu"):
        CommandLineOptions.bugleLanguage = a.lower()
      else:
        GPUVerifyError("argument to --bugle-lang must be 'cl' or 'cu'", ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--vcgen-timeout":
      try:
        CommandLineOptions.vcgenTimeout = int(a)
        if CommandLineOptions.vcgenTimeout < 0:
          raise ValueError
      except ValueError as e:
          GPUVerifyError("Invalid VCGen timeout \"" + a + "\"", ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--infer-timeout":
      try:
        CommandLineOptions.cruncherTimeout = int(a)
        if CommandLineOptions.cruncherTimeout < 0:
          raise ValueError
      except ValueError as e:
          GPUVerifyError("Invalid Cruncher timeout \"" + a + "\"", ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--timeout":
      try:
        CommandLineOptions.boogieTimeout = int(a)
        if CommandLineOptions.boogieTimeout < 0:
          raise ValueError
      except ValueError as e:
          GPUVerifyError("Invalid timeout \"" + a + "\"", ErrorCodes.COMMAND_LINE_ERROR)
    if o == "--boogie-file":
      filename, ext = SplitFilenameExt(a)
      if ext != ".bpl":
        GPUVerifyError("'" + a + "' specified via --boogie-file should have extension .bpl", ErrorCodes.COMMAND_LINE_ERROR)
      CommandLineOptions.gpuVerifyCruncherOptions += [ a ]
    if o == "--infer-config-file":
      filename, ext = SplitFilenameExt(a)
      if ext != ".cfg":
        GPUVerifyError("'" + a + "' specified via --infer-config-file should have extension .cfg", ErrorCodes.COMMAND_LINE_ERROR)
      CommandLineOptions.invInferConfigFile = a

def processOpenCLOptions(opts, args):
  for o, a in opts:
    if o == "--local_size":
      if CommandLineOptions.groupSize != []:
        GPUVerifyError("illegal to define local_size multiple times", ErrorCodes.COMMAND_LINE_ERROR)
      try:
        CommandLineOptions.groupSize = processVector(a)
      except ValueError:
        GPUVerifyError("argument to --local_size must be a (vector of) positive integer(s), found '" + a + "'", ErrorCodes.COMMAND_LINE_ERROR)
      CommandLineOptions.gpuVerifyCruncherOptions += [ "/blockHighestDim:" + str(len(CommandLineOptions.groupSize) - 1) ]
      CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/blockHighestDim:" + str(len(CommandLineOptions.groupSize) - 1) ]
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
      CommandLineOptions.gpuVerifyCruncherOptions += [ "/gridHighestDim:" + str(len(CommandLineOptions.numGroups) - 1) ]
      CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/gridHighestDim:" + str(len(CommandLineOptions.numGroups) - 1) ]
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
      CommandLineOptions.gpuVerifyCruncherOptions += [ "/blockHighestDim:" + str(len(CommandLineOptions.groupSize) - 1) ]
      CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/blockHighestDim:" + str(len(CommandLineOptions.groupSize) - 1) ]
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
      CommandLineOptions.gpuVerifyCruncherOptions += [ "/gridHighestDim:" + str(len(CommandLineOptions.numGroups) - 1) ]
      CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/gridHighestDim:" + str(len(CommandLineOptions.numGroups) - 1) ]
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
    opts, args = getopt.gnu_getopt(argv[1:],'D:I:h', 
             ['help', 'version', 'debug', 'findbugs', 'verify', 'noinfer', 'no-infer', 'verbose', 'silent',
              'loop-unwind=', 'memout=', 'no-benign', 'only-divergence', 'only-intra-group', 
              'only-log', 'adversarial-abstraction', 'equality-abstraction', 
              'no-annotations', 'only-requires', 'no-barrier-access-checks', 'no-constant-write-checks',
              'no-loop-predicate-invariants', 'no-smart-predication', 'no-source-loc-infer',
              'no-uniformity-analysis', 'call-site-analysis', 'clang-opt=',
              'vcgen-opt=', 'vcgen-timeout=', 'boogie-opt=', 'bugle-opt=',
              'local_size=', 'num_groups=', 'blockDim=', 'gridDim=', 'math-int',
              'stop-at-opt', 'stop-at-gbpl', 'stop-at-bpl', 'stop-at-inv',
              'time', 'time-as-csv=', 'keep-temps',
              'asymmetric-asserts', 'gen-smt2', 'testsuite', 'bugle-lang=','timeout=',
              'boogie-file=', 'infer-config-file=',
              'infer-timeout=', 'staged-inference', 'parallel-inference',
              'dynamic-analysis', 'scheduling=', 'infer-info', 'debug-houdini',
              'warp-sync=', 'atomic=', 'no-refined-atomics',
              'solver=', 'logic='
             ])
  except getopt.GetoptError as getoptError:
    GPUVerifyError(getoptError.msg + ".  Try --help for list of options", ErrorCodes.COMMAND_LINE_ERROR)

  showHelpIfRequested(opts)
  showVersionIfRequested(opts)
  getSourceFiles(args)
  processGeneralOptions(opts, args)
  if CommandLineOptions.SL == SourceLanguage.OpenCL:
    processOpenCLOptions(opts, args)
  if CommandLineOptions.SL == SourceLanguage.CUDA:
    processCUDAOptions(opts, args)

  filename, ext = SplitFilenameExt(args[0])

  if ext == ".cl":
    CommandLineOptions.clangOptions += clangOpenCLOptions
    CommandLineOptions.clangOptions += clangInlineOptions
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
  ibplFilename = filename + '.inv.bpl'
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
    if not CommandLineOptions.stopAtOpt: atexit.register(DeleteFile, optFilename)
    if not CommandLineOptions.stopAtGbpl: atexit.register(DeleteFile, gbplFilename)
    if not CommandLineOptions.stopAtBpl: atexit.register(DeleteFile, bplFilename)
    if not CommandLineOptions.stopAtBpl: atexit.register(DeleteFile, locFilename)
    if not CommandLineOptions.stopAtInv: atexit.register(DeleteFile, ibplFilename)

  CommandLineOptions.clangOptions.append("-o")
  CommandLineOptions.clangOptions.append(bcFilename)
  CommandLineOptions.clangOptions.append(filename + ext)

  CommandLineOptions.optOptions += [ "-o", optFilename, bcFilename ]

  if ext in [ ".cl", ".cu" ]:
    CommandLineOptions.bugleOptions += [ "-l", "cl" if ext == ".cl" else "cu", "-o", gbplFilename, optFilename ]
    if CommandLineOptions.mathInt:
      CommandLineOptions.bugleOptions += [ "-i", "math" ]
  elif not CommandLineOptions.skip['bugle']:
    lang = CommandLineOptions.bugleLanguage
    if not lang: # try to infer
      try:
        proc = subprocess.Popen([ gvfindtools.llvmBinDir + "/llvm-nm", filename + ext ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if "get_local_size" in stdout: lang = 'cl'
        if "blockDim" in stdout: lang = 'cu'
      except: pass
    if not lang:
      GPUVerifyError("must specify --bugle-lang=[cl|cu] when given a bitcode .bc file", ErrorCodes.COMMAND_LINE_ERROR)
    assert lang in [ "cl", "cu" ]
    CommandLineOptions.bugleOptions += [ "-l", lang, "-o", gbplFilename, optFilename ]

  CommandLineOptions.gpuVerifyVCGenOptions += [ "/atomics:" + CommandLineOptions.atomic ]
  if CommandLineOptions.warpSync:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/doWarpSync:" + str(CommandLineOptions.warpSize) ]
  if CommandLineOptions.noRefinedAtomics:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/noRefinedAtomics" ]
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
    CommandLineOptions.gpuVerifyCruncherOptions += [ "/onlyIntraGroupRaceChecking" ]
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/onlyIntraGroupRaceChecking" ]
  if CommandLineOptions.onlyLog:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/onlyLog" ]
  if CommandLineOptions.mode == AnalysisMode.FINDBUGS or (not CommandLineOptions.inference):
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/noInfer" ]
  if CommandLineOptions.noBarrierAccessChecks:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/noBarrierAccessChecks" ]
  if CommandLineOptions.noConstantWriteChecks:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/noConstantWriteChecks" ]
  if CommandLineOptions.noLoopPredicateInvariants:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/noLoopPredicateInvariants" ]
  if CommandLineOptions.noSmartPredication:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/noSmartPredication" ]
  if CommandLineOptions.noSourceLocInfer:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/noSourceLocInfer" ]
    CommandLineOptions.gpuVerifyCruncherOptions += [ "/noSourceLocInfer" ]
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/noSourceLocInfer" ]
  if CommandLineOptions.noUniformityAnalysis:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/noUniformityAnalysis" ]
  if CommandLineOptions.asymmetricAsserts:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/asymmetricAsserts" ]
  if CommandLineOptions.stagedInference:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/stagedInference" ]
    CommandLineOptions.gpuVerifyCruncherOptions += [ "/stagedInference" ]
  if CommandLineOptions.mathInt:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/mathInt" ]
  if CommandLineOptions.callSiteAnalysis:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/callSiteAnalysis" ]

  CommandLineOptions.gpuVerifyVCGenOptions += [ "/print:" + filename, gbplFilename ] #< ignore .bpl suffix

  if CommandLineOptions.mode == AnalysisMode.FINDBUGS:
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/loopUnroll:" + str(CommandLineOptions.loopUnwindDepth) ]
  
  if CommandLineOptions.boogieMemout > 0:
    CommandLineOptions.gpuVerifyCruncherOptions.append("/z3opt:-memory:" + str(CommandLineOptions.boogieMemout))
    CommandLineOptions.gpuVerifyBoogieDriverOptions.append("/z3opt:-memory:" + str(CommandLineOptions.boogieMemout))
    
  CommandLineOptions.gpuVerifyCruncherOptions += [ "/noinfer" ]
  CommandLineOptions.gpuVerifyCruncherOptions += [ "/contractInfer" ]
  CommandLineOptions.gpuVerifyCruncherOptions += [ "/concurrentHoudini" ]
  if CommandLineOptions.inferInfo:
    CommandLineOptions.gpuVerifyCruncherOptions += [ "/inferInfo" ]
    CommandLineOptions.gpuVerifyCruncherOptions += [ "/trace" ]
  if CommandLineOptions.debuggingHoudini:
    CommandLineOptions.gpuVerifyCruncherOptions += [ "/debugConcurrentHoudini" ]
  if CommandLineOptions.parallelInference:
    CommandLineOptions.gpuVerifyCruncherOptions += [ "/parallelInference" ]
    CommandLineOptions.gpuVerifyCruncherOptions += [ "/parallelInferenceScheduling:" + CommandLineOptions.scheduling ]
  elif CommandLineOptions.solver == "cvc4":
      CommandLineOptions.gpuVerifyCruncherOptions += [ "/proverOpt:SOLVER=cvc4" ]
  if CommandLineOptions.dynamicAnalysis:
    CommandLineOptions.gpuVerifyCruncherOptions += [ "/dynamicAnalysis" ]
  CommandLineOptions.gpuVerifyCruncherOptions += [ "/z3exe:" + gvfindtools.z3BinDir + os.sep + "z3.exe" ]
  CommandLineOptions.gpuVerifyCruncherOptions += [ "/cvc4exe:" + gvfindtools.cvc4BinDir + os.sep + "cvc4.exe" ]
  CommandLineOptions.gpuVerifyCruncherOptions += [ "/proverOpt:LOGIC=" + CommandLineOptions.logic ]
  
  if CommandLineOptions.solver == "cvc4":
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/proverOpt:SOLVER=cvc4" ]
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/cvc4exe:" + gvfindtools.cvc4BinDir + os.sep + "cvc4.exe" ]
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/proverOpt:LOGIC=" + CommandLineOptions.logic ]
  else:
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/z3exe:" + gvfindtools.z3BinDir + os.sep + "z3.exe" ]

  if CommandLineOptions.generateSmt2:
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/proverLog:" + smt2Filename ]
  if CommandLineOptions.debugging:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/debugGPUVerify" ]
    CommandLineOptions.gpuVerifyCruncherOptions += [ "/debugGPUVerify" ]
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/debugGPUVerify" ]
  if not CommandLineOptions.mathInt:
    # CommandLineOptions.gpuVerifyCruncherOptions += [ "/proverOpt:OPTIMIZE_FOR_BV=true" ]
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/proverOpt:OPTIMIZE_FOR_BV=true" ]
    if CommandLineOptions.solver == "z3":
      CommandLineOptions.gpuVerifyCruncherOptions += [ "/z3opt:RELEVANCY=0", "/z3opt:SOLVER=true" ]
      CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/z3opt:RELEVANCY=0", "/z3opt:SOLVER=true" ]
  
  CommandLineOptions.gpuVerifyCruncherOptions += CommandLineOptions.defaultOptions
  CommandLineOptions.gpuVerifyBoogieDriverOptions += CommandLineOptions.defaultOptions
  CommandLineOptions.gpuVerifyCruncherOptions += [ "/invInferConfigFile:" + os.path.dirname(os.path.abspath(__file__)) + os.sep + CommandLineOptions.invInferConfigFile ]
  CommandLineOptions.gpuVerifyCruncherOptions += [ bplFilename ]
  CommandLineOptions.gpuVerifyBoogieDriverOptions += [ bplFilename ]

  """ RUN CLANG """
  if not CommandLineOptions.skip["clang"]:
    RunTool("clang",
             [gvfindtools.llvmBinDir + "/clang"] +
             CommandLineOptions.clangOptions +
             [("-I" + str(o)) for o in CommandLineOptions.includes] +
             [("-D" + str(o)) for o in CommandLineOptions.defines],
             ErrorCodes.CLANG_ERROR)

  """ RUN OPT """
  if not CommandLineOptions.skip["opt"]:
    RunTool("opt",
            [gvfindtools.llvmBinDir + "/opt"] +
            CommandLineOptions.optOptions,
            ErrorCodes.OPT_ERROR)

  if CommandLineOptions.stopAtOpt: return 0

  """ RUN BUGLE """
  if not CommandLineOptions.skip["bugle"]:
    RunTool("bugle",
            [gvfindtools.bugleBinDir + "/bugle"] +
            CommandLineOptions.bugleOptions,
            ErrorCodes.BUGLE_ERROR)

  if CommandLineOptions.stopAtGbpl: return 0

  """ RUN GPUVERIFYVCGEN """
  timeoutArguments={}
  if CommandLineOptions.vcgenTimeout > 0:
    timeoutArguments['timeout']= CommandLineOptions.vcgenTimeout
    timeoutArguments['timeoutErrorCode']=ErrorCodes.GPUVERIFYVCGEN_TIMEOUT
  if not CommandLineOptions.skip["vcgen"]:
    RunTool("gpuverifyvcgen",
            (["mono"] if os.name == "posix" else []) +
            [gvfindtools.gpuVerifyVCGenBinDir + "/GPUVerifyVCGen.exe"] +
            CommandLineOptions.gpuVerifyVCGenOptions,
            ErrorCodes.GPUVERIFYVCGEN_ERROR,
            **timeoutArguments)

  if CommandLineOptions.stopAtBpl: return 0

  if CommandLineOptions.inference and (not CommandLineOptions.mode == AnalysisMode.FINDBUGS):
    """ RUN GPUVERIFYCRUNCHER """
    timeoutArguments={}
    if CommandLineOptions.cruncherTimeout > 0:
      timeoutArguments['timeout']= CommandLineOptions.cruncherTimeout
      timeoutArguments['timeoutErrorCode']=ErrorCodes.BOOGIE_TIMEOUT
    if not CommandLineOptions.skip["cruncher"]:
      RunTool("gpuverifycruncher",
              (["mono"] if os.name == "posix" else []) +
              [gvfindtools.gpuVerifyCruncherBinDir + os.sep + "GPUVerifyCruncher.exe"] +
              CommandLineOptions.gpuVerifyCruncherOptions,
              ErrorCodes.BOOGIE_ERROR,
              **timeoutArguments)
              
    if CommandLineOptions.stopAtInv: return 0

  """ RUN GPUVERIFYBOOGIEDRIVER """
  timeoutArguments={}
  if CommandLineOptions.boogieTimeout > 0:
    timeoutArguments['timeout']= CommandLineOptions.boogieTimeout
    timeoutArguments['timeoutErrorCode']=ErrorCodes.BOOGIE_TIMEOUT

  RunTool("gpuverifyboogiedriver",
          (["mono"] if os.name == "posix" else []) +
          [gvfindtools.gpuVerifyBoogieDriverBinDir + "/GPUVerifyBoogieDriver.exe"] +
          CommandLineOptions.gpuVerifyBoogieDriverOptions,
          ErrorCodes.BOOGIE_ERROR,
          **timeoutArguments)

  """ SUCCESS - REPORT STATUS """
  if CommandLineOptions.silent:
    return 0

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

def showTiming():
  if Timing:
    tools, times = map(list, zip(*Timing))
    total = sum(times)
  else:
    tools, times = [], []
    total = 0.0

  if CommandLineOptions.timeCSVLabel is not None:
    label = CommandLineOptions.timeCSVLabel
    times.append(total)
    row = [ '%.3f' % t for t in times ]
    if len(label) > 0: row.insert(0, label)
    if exitHook.code is ErrorCodes.SUCCESS:
      row.insert(1,'PASS')
      print ', '.join(row)
    else:
      row.insert(1,'FAIL(' + str(exitHook.code) + ')')
      print >> sys.stderr, ', '.join(row)
  else:
    print "Timing information (%.2f secs):" % total
    if tools:
      padTool = max([ len(tool) for tool in tools ])
      padTime = max([ len('%.3f secs' % t) for t in times ])
      for (tool, t) in Timing:
        print "- %s : %s" % (tool.ljust(padTool), ('%.3f secs' % t).rjust(padTime))
    else:
      print "- no tools ran"

def killChildrenPosix():
  def handler(signal,frame):
    return

  signal.signal(signal.SIGINT, handler)
  os.killpg(0,signal.SIGINT)

def exitHandler():
  if CommandLineOptions.time:
    showTiming()

  sys.stderr.flush()
  sys.stdout.flush()

  # Kill child processes that might not have been killed, e.g., Z3
  if os.name == 'posix':
    killChildrenPosix()

if __name__ == '__main__':
  atexit.register(exitHandler)
  sys.exit(main())
