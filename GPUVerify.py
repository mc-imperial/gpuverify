#!/usr/bin/env python
# vim: set shiftwidth=2 tabstop=2 expandtab softtabstop=2:

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

class GPUVerifyException(Exception):
  """
    These exceptions are used as a replacement
    for using sys.exit()
  """

  def __init__(self, code, msg=None, stdout=None, stderr=None):
    """
      code : Should be a member of the ErrorCodes class
      msg  : An optional string
      stdout : An optional string showing stdout message of a tool
      stderr : An optional string showing stderr message of a tool
    """
    self.code = code
    self.msg = msg
    self.stdout= stdout
    self.stderr = stderr
  
  def getExitCode(self):
    return self.code

  def __str__(self):
    """
      Provide a human readable form of the Exception
    """
    # Determine string for error code
    codeString = None
    for cs in [ x for x in dir(ErrorCodes) if not x.startswith('_') ]:
      if getattr(ErrorCodes, cs) == self.code:
        codeString = cs

    if codeString == None:
      codeString = 'UNKNOWN'

    retStr = 'GPUVerify: {} error ({}):'.format(codeString, self.code)

    if self.msg:
      retStr = retStr + ' ' + self.msg

    if self.stdout:
      retStr = retStr + '\n\nStandard output:\n' + str(self.stdout.decode())

    if self.stderr:
      retStr = retStr + '\n\nStandard error:\n' + str(self.stderr.decode())

    return retStr

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
  raise GPUVerifyException('Cannot find \'gvfindtools.py\'.'
                           ' Did you forget to create it from a template?',
                           ErrorCodes.CONFIGURATION_ERROR)

class BatchCaller(object):
  """
  This class allows functions to be registered (similar to atexit )
  and later called using the call() method
  """

  def __init__(self, verbose=False):
    from collections import namedtuple
    self.calls = [ ]
    self.verbose = verbose

    # The type we will use to represent function calls
    self.fcallType = namedtuple('FCall',['function', 'nargs', 'kargs'])

  def setVerbose(self, v=True):
    self.verbose = v

  def register(self, function, *nargs, **kargs):
    """
    Register function.

    function : The function to call

    The remaining arguments can be positional or keyword arguments
    to pass to the function.
    """
    call = self.fcallType(function, nargs, kargs)
    self.calls.append(call)

  def call(self, inReverse=False):
    """ Call registered functions
    """
    if inReverse:
      self.calls.reverse()

    for call in self.calls:
      if self.verbose:
        print("Clean up handler Calling " + str(call.function.__name__) + '(' + 
              str(call.nargs) + ', ' + str(call.kargs) + ')' )
      call.function(*(call.nargs), **(call.kargs))

  def clear(self):
    """
      Remove all registered calls
    """
    self.calls = [ ]

    assert len(self.calls) == 0

cleanUpHandler = BatchCaller()

""" Timing for the toolchain pipeline """
Timing = []

""" WindowsError is not defined on UNIX systems, this works around that """
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
    raise GPUVerifyException(ErrorCodes.CONFIGURATION_ERROR, 'Could not find Bugle Inline Check plugin')

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
class DefaultCmdLineOptions(object):
  """
  This class defines all the default options for the tool
  """
  def __init__(self):
    self.SL = SourceLanguage.Unknown
    self.sourceFiles = [] # The OpenCL or CUDA files to be processed
    self.includes = clangCoreIncludes
    self.defines = clangCoreDefines
    self.clangOptions = list(clangCoreOptions) # Make sure we make a copy so we don't change the global list
    self.optOptions = [ "-mem2reg", "-globaldce" ]
    self.defaultOptions = [ "/nologo", "/typeEncoding:m", "/mv:-",
                       "/doModSetAnalysis", "/useArrayTheory",
                       "/doNotUseLabels", "/enhancedErrorMessages:1"
                     ]
    self.gpuVerifyVCGenOptions = []
    self.gpuVerifyCruncherOptions = []
    self.gpuVerifyBoogieDriverOptions = []
    self.bugleOptions = []
    self.mode = AnalysisMode.ALL
    self.debugging = False
    self.verbose = False
    self.silent = False
    self.groupSize = []
    self.numGroups = []
    self.adversarialAbstraction = False
    self.equalityAbstraction = False
    self.loopUnwindDepth = 2
    self.noBenign = False
    self.onlyDivergence = False
    self.onlyIntraGroup = False
    self.onlyLog = False
    self.noLoopPredicateInvariants = False
    self.noSmartPredication = False
    self.noSourceLocInfer = False
    self.noUniformityAnalysis = False
    self.inference = True
    self.invInferConfigFile = "inference.cfg"
    self.stagedInference = False
    self.parallelInference = False
    self.dynamicAnalysis = False
    self.scheduling = "default"
    self.inferInfo = False
    self.debuggingHoudini = False
    self.stopAtOpt = False
    self.stopAtGbpl = False
    self.stopAtBpl = False
    self.stopAtCbpl = False
    self.time = False
    self.timeCSVLabel = None
    self.boogieMemout=0
    self.vcgenTimeout=0
    self.cruncherTimeout=300
    self.boogieTimeout=300
    self.keepTemps = False
    self.mathInt = False
    self.asymmetricAsserts = False
    self.generateSmt2 = False
    self.noBarrierAccessChecks = False
    self.noConstantWriteChecks = False
    self.noInline = False
    self.callSiteAnalysis = False
    self.warpSync = False
    self.warpSize = 32
    self.atomic = "rw"
    self.noRefinedAtomics = False
    self.solver = "z3"
    self.logic = "QF_ALL_SUPPORTED"
    self.skip = { "clang": False,
             "opt": False,
             "bugle": False,
             "vcgen": False,
             "cruncher": False }
    self.bugleLanguage = None

# Use instance of class so we can later reset it
CommandLineOptions = DefaultCmdLineOptions()

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
    print(" ".join(command))
  else:
    popenargs['bufsize']=0
    if __name__ != '__main__':
      # We don't want messages to go to stdout if being used as module
      popenargs['stdout']=subprocess.PIPE
      popenargs['stderr']=subprocess.PIPE
    else:
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
    raise GPUVerifyException(ErrorCodes.CTRL_C)
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
    raise GPUVerifyException( timeoutErrorCode, ToolName + " timed out.  Use --timeout=N with N > " + str(timeout) + " to increase timeout, or --timeout=0 to disable timeout.")
  except (OSError,WindowsError) as e:
    raise GPUVerifyException(ErrorCode, "While invoking " + ToolName + ": " + str(e) + "\nWith command line args:\n" + pprint.pformat(Command))

  if CommandLineOptions.time:
    Timing.append((ToolName, end-start))
  if returnCode != ErrorCodes.SUCCESS:
    raise GPUVerifyException(ErrorCode,"", stdout, stderr)

def showVersionAndExit():
  """ This will check if using gpuverify from development directory.
      If so this will invoke Mercurial to find out version information.
      If this is a deployed version we will try to read the version from
      a file instead
  """

  print(getversion.getVersionString())
  raise GPUVerifyException(ErrorCodes.SUCCESS)

def showHelpAndExit():
  stringReplacements = {
    'boogieMemout': CommandLineOptions.boogieMemout,
    'boogieTimeout': CommandLineOptions.boogieTimeout
  }

  print("""OVERVIEW: GPUVerify driver

  USAGE: GPUVerify.py [options] <inputs>

  GENERAL OPTIONS:
    -h, --help              Display this message
    -I <value>              Add directory to include search path
    -D <value>              Define symbol
    --findbugs              Run tool in bug-finding mode
    --loop-unwind=X         Explore traces that pass through at most X loop heads
    --memout=X              Give Boogie a hard memory limit of X megabytes.
                            A memout of 0 disables the memout. The default is {boogieMemout} megabytes.
    --no-benign             Do not tolerate benign data races
    --only-divergence       Only check for barrier divergence, not for races
    --only-intra-group      Do not check for inter-group races
    --time                  Show timing information
    --timeout=X             Allow Boogie to run for X seconds before giving up.
                            A timeout of 0 disables the timeout. The default is {boogieTimeout} seconds.
    --verify                Run tool in verification mode
    --verbose               Show commands to run and use verbose output
    --version               Show version information.

  ADVANCED OPTIONS:
    --adversarial-abstraction  Completely abstract shared state, so that reads are
                            nondeterministic
    --array-equalities      Generate equality candidate invariants for array variables
    --asymmetric-asserts    Emit assertions only for first thread.  Sound, and may lead
                            to faster verification, but can yield false positives
    --boogie-file=X.bpl     Specify a supporting .bpl file to be used during verification
    --boogie-opt=...        Specify option to be passed to Boogie
    --bugle-lang=[cl|cu]    Bitcode language if passing in a bitcode file
    --bugle-opt=...         Specify option to be passed to Bugle
    --call-site-analysis    Turn on call site analysis
    --clang-opt=...         Specify option to be passed to CLANG
    --debug                 Enable debugging of GPUVerify components: exceptions will
                            not be suppressed
    --equality-abstraction  Make shared arrays nondeterministic, but consistent between
                            threads, at barriers
    --gen-smt2              Generate smt2 file
    --keep-temps            Keep intermediate bc, gbpl, bpl and cbpl files
    --math-int              Represent integer types using mathematical integers
                            instead of bit-vectors
    --no-annotations        Ignore all source-level annotations
    --only-requires         Ignore all source-level annotations except for requires
    --no-barrier-access-checks      Turn off access checks for barrier invariants
    --no-constant-write-checks      Turn off access checks for writes to constant space
    --no-inline             Turn off automatic inlining by Bugle
    --no-loop-predicate-invariants  Turn off automatic generation of loop invariants
                            related to predicates, which can be incorrect
    --no-smart-predication  Turn off smart predication
    --no-source-loc-infer   Turn off inference of source location information
    --no-uniformity-analysis  Turn off uniformity analysis
    --only-log              Log accesses to arrays, but do not check for races.  This
                            can be useful for determining access pattern invariants
    --silent                Silent on success; only show errors/timing
    --stop-at-opt           Stop after LLVM optimization pass
    --stop-at-gbpl          Stop after generating gbpl
    --stop-at-cbpl          Stop after generating an annotated bpl
    --stop-at-bpl           Stop after generating bpl
    --time-as-csv=label     Print timing as CSV row with label
    --vcgen-timeout=X       Allow VCGen to run for X seconds.
    --vcgen-opt=...         Specify option to be passed to be passed to VC generation
                            engine
    --warp-sync=X           Synchronize threads within warps, sized X, defaulting to 32
    --atomic=X              Check atomics as racy against reads (r), writes(w), both(rw), or none(none)
                            (default is --atomic=rw)
    --no-refined-atomics    Don't do abstraction refinement on the return values from atomics
    --solver=X              Choose which SMT Theorem Prover to use in the backend.
                            Available options: 'Z3' or 'cvc4' (default is 'Z3')
    --logic=X               Define the logic to be used by the CVC4 SMT solver backend
                            (default is QF_ALL_SUPPORTED)

  INVARIANT INFERENCE OPTIONS:
    --no-infer              Turn off invariant inference
    --infer-timeout=X       Allow GPUVerifyCruncher to run for X seconds.
    --staged-inference      Perform invariant inference in stages; this can boost
                            performance for complex kernels (but this is not guaranteed)
    --parallel-inference    Use multiple solver instances in parallel to accelerate invariant
                            inference (but this is not guaranteed)
    --dynamic-analysis      Use dynamic analysis to falsify invariants.
    --scheduling=X          Choose a parallel scheduling strategy from the following: 'default',
                            'unsound-first' or 'brute-force'. The 'default' strategy executes
                            first any dynamic engines, then any unsound static engines and then
                            the sound static engines. The 'unsound-first' strategy executes any
                            unsound engines (either static or dynamic) together before the sound
                            engines. The 'brute-force' strategy executes all engines together but
                            performance is highly non-deterministic.
    --infer-config-file=X.cfg       Specify a custom configuration file to be used
                            during invariant inference
    --infer-info            Prints information about the inference process.

  OPENCL OPTIONS:
    --local_size=X          Specify whether work-group is 1D, 2D
                =[X,Y]      or 3D and specify size for each
                =[X,Y,Z]    dimension
    --num_groups=X          Specify whether grid of work-groups is
                =[X,Y]      1D, 2D or 3D and specify size for each
                =[X,Y,Z]    dimension

  CUDA OPTIONS
    --blockDim=X            Specify whether thread block is 1D, 2D
                =[X,Y]      or 3D and specify size for each
                =[X,Y,Z]    dimension
    --gridDim=X             Specify whether grid of thread blocks is
                =[X,Y]      1D, 2D or 3D and specify size for each
                =[X,Y,Z]    dimension
  """.format(**stringReplacements))
  raise GPUVerifyException(ErrorCodes.SUCCESS)

def processVector(vector):
  vector = vector.strip()
  if vector[0] == '[' and vector[-1] == ']':
    return list(map(int, vector[1:-1].split(",")))
  else:
    return list(map(int, vector.split(",")))

def GPUVerifyWarn(msg):
  print("GPUVerify: warning: " + msg)

def Verbose(msg):
  if(CommandLineOptions.verbose):
    print(msg)

def getSourceFiles(args):
  if len(args) == 0:
    raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "no .cl or .cu files supplied")
  for a in args:
    filename, ext = SplitFilenameExt(a)
    if ext == ".cl":
      if CommandLineOptions.SL == SourceLanguage.CUDA:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "illegal to pass both .cl and .cu files simultaneously")
      CommandLineOptions.SL = SourceLanguage.OpenCL
    elif ext == ".cu":
      if CommandLineOptions.SL == SourceLanguage.OpenCL:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "illegal to pass both .cl and .cu files simultaneously")
      CommandLineOptions.SL = SourceLanguage.CUDA
    elif ext in [ ".bc", ".opt.bc", ".gbpl", ".bpl", ".cbpl" ]:
      CommandLineOptions.skip["clang"] = True
      if ext in [        ".opt.bc", ".gbpl", ".bpl", ".cbpl" ]:
        CommandLineOptions.skip["opt"] = True
      if ext in [                   ".gbpl", ".bpl", ".cbpl" ]:
        CommandLineOptions.skip["bugle"] = True
      if ext in [                            ".bpl", ".cbpl" ]:
        CommandLineOptions.skip["vcgen"] = True
      if ext in [                                    ".cbpl" ]:
        CommandLineOptions.skip["cruncher"] = True
    else:
      raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "'" + a + "' has unknown file extension, supported file extensions are .cl (OpenCL) and .cu (CUDA)")
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
    if o == "--no-inline":
      CommandLineOptions.noInline = True
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
    if o == "--stop-at-cbpl":
      CommandLineOptions.stopAtCbpl = True
    if o == "--stop-at-bpl":
      CommandLineOptions.stopAtBpl = True
    if o == "--time":
      CommandLineOptions.time = True
    if o == "--time-as-csv":
      CommandLineOptions.time = True
      CommandLineOptions.timeCSVLabel = a
    if o == "--asymmetric-asserts":
      CommandLineOptions.asymmetricAsserts = True
    if o == "--gen-smt2":
      CommandLineOptions.generateSmt2 = True
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
          raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "negative value " + a + " provided as argument to --loop-unwind")
        CommandLineOptions.loopUnwindDepth = int(a)
      except ValueError:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "non integer value '" + a + "' provided as argument to --loop-unwind")
    if o == "--memout":
      try:
        CommandLineOptions.boogieMemout = int(a)
        if CommandLineOptions.boogieMemout < 0:
          raise ValueError
      except ValueError as e:
          raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "Invalid memout \"" + a + "\"")
    if o == "--adversarial-abstraction":
      if CommandLineOptions.equalityAbstraction:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "illegal to specify both adversarial and equality abstractions")
      CommandLineOptions.adversarialAbstraction = True
    if o == "--equality-abstraction":
      if CommandLineOptions.adversarialAbstraction:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "illegal to specify both adversarial and equality abstractions")
      CommandLineOptions.equalityAbstraction = True
    if o == "--warp-sync":
      CommandLineOptions.warpSync = True
      try:
        if int(a) < 0 :
          raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "negative value " + a + " provided as argument to --warp-sync")
        CommandLineOptions.warpSize = int(a)
      except ValueError:
        raise GPUVerifyException(ErrorCodes.COMMAND_LINE_ERROR, "non integer value '" + a + "' provided as argument to --warp-sync")
    if o == "--atomic":
      if a.lower() in ("r","w","rw","none"):
        CommandLineOptions.atomic = a.lower()
      else:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "argument to --atomic must be 'r','w','rw', or 'none'")
    if o == "--solver":
      if a.lower() in ("z3","cvc4"):
        CommandLineOptions.solver = a.lower()
      else:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "argument to --solver must be 'Z3' or 'CVC4'")
    if o == "--scheduling":
      if a.lower() in ("all-together","unsound-first","dynamic-first","phased"):
        CommandLineOptions.scheduling = a.lower()
      else:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "argument to --scheduling must be 'all-together', 'unsound-first', 'dynamic-first' or'phased'")
    if o == "--logic":
      if a.upper() in ("ALL_SUPPORTED","QF_ALL_SUPPORTED"):
        CommandLineOptions.logic = a.upper()
      else:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "argument to --logic must be 'ALL_SUPPORTED' or 'QF_ALL_SUPPORTED'")
    if o == "--bugle-lang":
      if a.lower() in ("cl", "cu"):
        CommandLineOptions.bugleLanguage = a.lower()
      else:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "argument to --bugle-lang must be 'cl' or 'cu'")
    if o == "--vcgen-timeout":
      try:
        CommandLineOptions.vcgenTimeout = int(a)
        if CommandLineOptions.vcgenTimeout < 0:
          raise ValueError
      except ValueError as e:
          raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "Invalid VCGen timeout \"" + a + "\"")
    if o == "--infer-timeout":
      try:
        CommandLineOptions.cruncherTimeout = int(a)
        if CommandLineOptions.cruncherTimeout < 0:
          raise ValueError
      except ValueError as e:
          raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "Invalid Cruncher timeout \"" + a + "\"")
    if o == "--timeout":
      try:
        CommandLineOptions.boogieTimeout = int(a)
        if CommandLineOptions.boogieTimeout < 0:
          raise ValueError
      except ValueError as e:
          raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "Invalid timeout \"" + a + "\"")
    if o == "--boogie-file":
      filename, ext = SplitFilenameExt(a)
      if ext != ".bpl":
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "'" + a + "' specified via --boogie-file should have extension .bpl")
      CommandLineOptions.gpuVerifyCruncherOptions += [ a ]
    if o == "--infer-config-file":
      filename, ext = SplitFilenameExt(a)
      if ext != ".cfg":
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "'" + a + "' specified via --infer-config-file should have extension .cfg")
      CommandLineOptions.invInferConfigFile = a

def processOpenCLOptions(opts, args):
  for o, a in opts:
    if o == "--local_size":
      if CommandLineOptions.groupSize != []:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "illegal to define local_size multiple times")
      try:
        CommandLineOptions.groupSize = processVector(a)
      except ValueError:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "argument to --local_size must be a (vector of) positive integer(s), found '" + a + "'")
      CommandLineOptions.gpuVerifyCruncherOptions += [ "/blockHighestDim:" + str(len(CommandLineOptions.groupSize) - 1) ]
      CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/blockHighestDim:" + str(len(CommandLineOptions.groupSize) - 1) ]
      for i in range(0, len(CommandLineOptions.groupSize)):
        if CommandLineOptions.groupSize[i] <= 0:
          raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "values specified for local_size dimensions must be positive")
    if o == "--num_groups":
      if CommandLineOptions.numGroups != []:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "illegal to define num_groups multiple times")
      try:
        CommandLineOptions.numGroups = processVector(a)
      except ValueError:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "argument to --num_groups must be a (vector of) positive integer(s), found '" + a + "'")
      CommandLineOptions.gpuVerifyCruncherOptions += [ "/gridHighestDim:" + str(len(CommandLineOptions.numGroups) - 1) ]
      CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/gridHighestDim:" + str(len(CommandLineOptions.numGroups) - 1) ]
      for i in range(0, len(CommandLineOptions.numGroups)):
        if CommandLineOptions.numGroups[i] <= 0:
          raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "values specified for num_groups dimensions must be positive")

  if CommandLineOptions.groupSize == []:
    raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "work group size must be specified via --local_size=...")
  if CommandLineOptions.numGroups == []:
    raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "number of work groups must be specified via --num_groups=...")

def processCUDAOptions(opts, args):
  for o, a in opts:
    if o == "--blockDim":
      if CommandLineOptions.groupSize != []:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "illegal to define blockDim multiple times")
      try:
        CommandLineOptions.groupSize = processVector(a)
      except ValueError:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "argument to --blockDim must be a (vector of) positive integer(s), found '" + a + "'")
      CommandLineOptions.gpuVerifyCruncherOptions += [ "/blockHighestDim:" + str(len(CommandLineOptions.groupSize) - 1) ]
      CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/blockHighestDim:" + str(len(CommandLineOptions.groupSize) - 1) ]
      for i in range(0, len(CommandLineOptions.groupSize)):
        if CommandLineOptions.groupSize[i] <= 0:
          raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "values specified for blockDim must be positive")
    if o == "--gridDim":
      if CommandLineOptions.numGroups != []:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "illegal to define gridDim multiple times")
      try:
        CommandLineOptions.numGroups = processVector(a)
      except ValueError:
        raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "argument to --gridDim must be a (vector of) positive integer(s), found '" + a + "'")
      CommandLineOptions.gpuVerifyCruncherOptions += [ "/gridHighestDim:" + str(len(CommandLineOptions.numGroups) - 1) ]
      CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/gridHighestDim:" + str(len(CommandLineOptions.numGroups) - 1) ]
      for i in range(0, len(CommandLineOptions.numGroups)):
        if CommandLineOptions.numGroups[i] <= 0:
          raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "values specified for gridDim must be positive")

  if CommandLineOptions.groupSize == []:
    raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "thread block size must be specified via --blockDim=...")
  if CommandLineOptions.numGroups == []:
    raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "grid size must be specified via --gridDim=...")

def _main(argv):
  """
   This function should NOT be called directly instead call main()
   It is assumed that argv has had sys.argv[0] removed
  """
  progname = __name__
  if progname.endswith('.py'):
    progname = progname[:-3]

  try:
    opts, args = getopt.gnu_getopt(argv,'D:I:h',
             ['help', 'version', 'debug', 'findbugs', 'verify', 'noinfer', 'no-infer', 'verbose', 'silent',
              'loop-unwind=', 'memout=', 'no-benign', 'only-divergence', 'only-intra-group',
              'only-log', 'adversarial-abstraction', 'equality-abstraction',
              'no-annotations', 'only-requires', 'no-barrier-access-checks', 'no-constant-write-checks',
              'no-inline', 'no-loop-predicate-invariants', 'no-smart-predication', 'no-source-loc-infer',
              'no-uniformity-analysis', 'call-site-analysis', 'clang-opt=',
              'vcgen-opt=', 'vcgen-timeout=', 'boogie-opt=', 'bugle-opt=',
              'local_size=', 'num_groups=', 'blockDim=', 'gridDim=', 'math-int',
              'stop-at-opt', 'stop-at-gbpl', 'stop-at-cbpl', 'stop-at-bpl',
              'time', 'time-as-csv=', 'keep-temps',
              'asymmetric-asserts', 'gen-smt2', 'bugle-lang=','timeout=',
              'boogie-file=', 'infer-config-file=',
              'infer-timeout=', 'staged-inference', 'parallel-inference',
              'dynamic-analysis', 'scheduling=', 'infer-info', 'debug-houdini',
              'warp-sync=', 'atomic=', 'no-refined-atomics',
              'solver=', 'logic='
             ])
  except getopt.GetoptError as getoptError:
    raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, getoptError.msg + ".  Try --help for list of options")

  showHelpIfRequested(opts)
  showVersionIfRequested(opts)
  getSourceFiles(args)
  processGeneralOptions(opts, args)
  if CommandLineOptions.SL == SourceLanguage.OpenCL:
    processOpenCLOptions(opts, args)
  if CommandLineOptions.SL == SourceLanguage.CUDA:
    processCUDAOptions(opts, args)

  cleanUpHandler.setVerbose(CommandLineOptions.verbose)

  filename, ext = SplitFilenameExt(args[0])

  if ext == ".cl":
    CommandLineOptions.clangOptions += clangOpenCLOptions
    CommandLineOptions.clangOptions += clangInlineOptions
    CommandLineOptions.includes += clangOpenCLIncludes
    CommandLineOptions.defines += clangOpenCLDefines
    CommandLineOptions.defines.append("__" + str(len(CommandLineOptions.groupSize)) + "D_WORK_GROUP")
    CommandLineOptions.defines.append("__" + str(len(CommandLineOptions.numGroups)) + "D_GRID")
    CommandLineOptions.defines += [ "__LOCAL_SIZE_" + str(i) + "=" + str(CommandLineOptions.groupSize[i]) for i in range(0, len(CommandLineOptions.groupSize))]
    CommandLineOptions.defines += [ "__NUM_GROUPS_" + str(i) + "=" + str(CommandLineOptions.numGroups[i]) for i in range(0, len(CommandLineOptions.numGroups))]

  elif ext == ".cu":
    CommandLineOptions.clangOptions += clangCUDAOptions
    CommandLineOptions.includes += clangCUDAIncludes
    CommandLineOptions.defines += clangCUDADefines
    CommandLineOptions.defines.append("__" + str(len(CommandLineOptions.groupSize)) + "D_THREAD_BLOCK")
    CommandLineOptions.defines.append("__" + str(len(CommandLineOptions.numGroups)) + "D_GRID")
    CommandLineOptions.defines += [ "__BLOCK_DIM_" + str(i) + "=" + str(CommandLineOptions.groupSize[i]) for i in range(0, len(CommandLineOptions.groupSize))]
    CommandLineOptions.defines += [ "__GRID_DIM_" + str(i) + "=" + str(CommandLineOptions.numGroups[i]) for i in range(0, len(CommandLineOptions.numGroups))]

  # Intermediate filenames
  bcFilename = filename + '.bc'
  optFilename = filename + '.opt.bc'
  gbplFilename = filename + '.gbpl'
  cbplFilename = filename + '.cbpl'
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
    cleanUpHandler.register(DeleteFile, bcFilename)
    if not CommandLineOptions.stopAtOpt: cleanUpHandler.register(DeleteFile, optFilename)
    if not CommandLineOptions.stopAtGbpl: cleanUpHandler.register(DeleteFile, gbplFilename)
    if not CommandLineOptions.stopAtCbpl: cleanUpHandler.register(DeleteFile, cbplFilename)
    if not CommandLineOptions.stopAtBpl: cleanUpHandler.register(DeleteFile, bplFilename)
    if not CommandLineOptions.stopAtBpl: cleanUpHandler.register(DeleteFile, locFilename)

  CommandLineOptions.clangOptions.append("-o")
  CommandLineOptions.clangOptions.append(bcFilename)
  CommandLineOptions.clangOptions.append(filename + ext)

  CommandLineOptions.optOptions += [ "-o", optFilename, bcFilename ]

  if ext in [ ".cl", ".cu" ]:
    CommandLineOptions.bugleOptions += [ "-l", "cl" if ext == ".cl" else "cu", "-o", gbplFilename, optFilename ]
    if CommandLineOptions.mathInt:
      CommandLineOptions.bugleOptions += [ "-i", "math" ]
    if not CommandLineOptions.noInline:
      CommandLineOptions.bugleOptions += [ "-inline" ]
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
      raise GPUVerifyException( ErrorCodes.COMMAND_LINE_ERROR, "must specify --bugle-lang=[cl|cu] when given a bitcode .bc file")
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
    CommandLineOptions.gpuVerifyCruncherOptions += [ "/proverLog:" + smt2Filename ]
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/proverLog:" + smt2Filename ]
  if CommandLineOptions.debugging:
    CommandLineOptions.gpuVerifyVCGenOptions += [ "/debugGPUVerify" ]
    CommandLineOptions.gpuVerifyCruncherOptions += [ "/debugGPUVerify" ]
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/debugGPUVerify" ]
  if not CommandLineOptions.mathInt:
    CommandLineOptions.gpuVerifyCruncherOptions += [ "/proverOpt:OPTIMIZE_FOR_BV=true" ]
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/proverOpt:OPTIMIZE_FOR_BV=true" ]
    if CommandLineOptions.solver == "z3":
      CommandLineOptions.gpuVerifyCruncherOptions += [ "/z3opt:RELEVANCY=0", "/z3opt:SOLVER=true" ]
      CommandLineOptions.gpuVerifyBoogieDriverOptions += [ "/z3opt:RELEVANCY=0", "/z3opt:SOLVER=true" ]
  
  CommandLineOptions.gpuVerifyCruncherOptions += CommandLineOptions.defaultOptions
  CommandLineOptions.gpuVerifyBoogieDriverOptions += CommandLineOptions.defaultOptions
  CommandLineOptions.gpuVerifyCruncherOptions += [ "/invInferConfigFile:" + os.path.dirname(os.path.abspath(__file__)) + os.sep + CommandLineOptions.invInferConfigFile ]
  CommandLineOptions.gpuVerifyCruncherOptions += [ bplFilename ]
  
  if CommandLineOptions.inference and (not CommandLineOptions.mode == AnalysisMode.FINDBUGS):
    CommandLineOptions.gpuVerifyBoogieDriverOptions += [ cbplFilename ]
  else:
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

    if CommandLineOptions.stopAtCbpl: return 0

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
    print("No defects were found while analysing: " + ", ".join(CommandLineOptions.sourceFiles))
    print("Notes:")
    print("- use --loop-unwind=N with N > " + str(CommandLineOptions.loopUnwindDepth) + " to search for deeper bugs")
    print("- re-run in verification mode to try to prove absence of defects")
  else:
    print("Verified: " + ", ".join(CommandLineOptions.sourceFiles))
    if not CommandLineOptions.onlyDivergence:
      print("- no data races within " + ("work groups" if CommandLineOptions.SL == SourceLanguage.OpenCL else "thread blocks"))
      if not CommandLineOptions.onlyIntraGroup:
        print("- no data races between " + ("work groups" if CommandLineOptions.SL == SourceLanguage.OpenCL else "thread blocks"))
    print("- no barrier divergence")
    print("- no assertion failures")
    print("(but absolutely no warranty provided)")

  return 0

def showTiming(exitCode):
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
    if exitCode is ErrorCodes.SUCCESS:
      row.insert(1,'PASS')
      print(', '.join(row))
    else:
      row.insert(1,'FAIL(' + str(exitCode) + ')')
      print >> sys.stderr, ', '.join(row)
  else:
    print("Timing information (%.2f secs):" % total)
    if tools:
      padTool = max([ len(tool) for tool in tools ])
      padTime = max([ len('%.3f secs' % t) for t in times ])
      for (tool, t) in Timing:
        print("- %s : %s" % (tool.ljust(padTool), ('%.3f secs' % t).rjust(padTime)))
    else:
      print("- no tools ran")

def killChildrenPosix():
  # Kill child processes that might not have been killed, e.g., Z3
  if os.name != 'posix':
    return

  def handler(signal,frame):
    return

  signal.signal(signal.SIGINT, handler)
  os.killpg(0,signal.SIGINT)

def handleTiming(exitCode):
  if CommandLineOptions.time:
    showTiming(exitCode)

  sys.stderr.flush()
  sys.stdout.flush()

def _cleanUpGlobals():
  """
  In order to make the tool importable and usable
  as a python module we need to clean up the
  global variables.
  """
  global CommandLineOptions

  # Reset options
  CommandLineOptions = DefaultCmdLineOptions()

def main(argv):
  """ This wraps GPUVerify's real main function so
      that we can handle exceptions and trigger our own exit
      commands.

      This is the entry point that should be used if you want
      to use this file as a module rather than as a script.

      If verification fails in any way then a GPUVerifyException
      will be raised. If verification was successful ErrorCodes.SUCCESS
      will be returned.

      Example:

      import GPUVerify
      try:
        GPUVerify.main(['--local_size=32','--num_groups=2','your_kernel.cl'])
      except GPUVerifyVerification as e:
        # Handle error
  """
  def doCleanUp(timing, exitCode=ErrorCodes.SUCCESS):
    if timing:
      # We must call this before cleaning up globals
      # because it depends on them
      cleanUpHandler.register(handleTiming, exitCode)

    # Clean up globals so main() can be re-executed in
    # the context of an interactive python console
    if __name__ != '__main__':
      cleanUpHandler.register(_cleanUpGlobals)

    # We should call this last.
    cleanUpHandler.register(killChildrenPosix)

    cleanUpHandler.call()

    cleanUpHandler.clear() # Clean up for next use

  try:
    _main(argv)
  except GPUVerifyException as e:
    doCleanUp(timing=True, exitCode=e.getExitCode() )
    raise
  except Exception:
    # Something went very wrong
    doCleanUp(timing=False, exitCode=0 ) # It doesn't matter what the exitCode is
    raise

  doCleanUp(timing=True) # Do this outside try block so we don't call twice!
  return ErrorCodes.SUCCESS
  
if __name__ == '__main__':
  try:
    main(sys.argv[1:])
  except GPUVerifyException as e:
    # We assume that globals are not cleaned up when running as a script so it 
    # is safe to read CommandLineOptions
    if e.getExitCode() != ErrorCodes.SUCCESS and CommandLineOptions.debugging:
      print(str(e))
    sys.exit(e.getExitCode())

  sys.exit(ErrorCodes.SUCCESS)
