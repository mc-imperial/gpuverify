#!/usr/bin/env python
# vim: set shiftwidth=2 tabstop=2 expandtab softtabstop=2:
from __future__ import print_function
import argparse
import os
import signal
import subprocess
import sys
import timeit
import threading
import multiprocessing # Only for determining number of CPU cores available
import getversion
import pprint
from collections import defaultdict
import copy

# To properly kill child processes cross platform
try:
  import psutil
  psutilPresent = True
except ImportError:
  psutilPresent = False

class GPUVerifyException(Exception):
  """
    These exceptions are used as a replacement
    for using sys.exit()
  """

  def __init__(self, code, msg=None):
    """
      code : Should be a member of the ErrorCodes class
      msg  : An optional string
      stdout : An optional string showing stdout message of a tool
      stderr : An optional string showing stderr message of a tool
    """
    self.code = code
    self.msg = msg

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

    retStr = 'GPUVerify: {} error ({})'.format(codeString, self.code)

    if self.msg:
      retStr = retStr + ': ' + self.msg

    return retStr

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

# Try to import the paths need for GPUVerify's tools
try:
  import gvfindtools
  # Initialise the paths (only needed for deployment version of gvfindtools.py)
  gvfindtools.init(sys.path[0])
except ImportError:
  raise GPUVerifyException(ErrorCodes.CONFIGURATION_ERROR,
                           'Cannot find \'gvfindtools.py\'.'
                           ' Did you forget to create it from a template?')

class BatchCaller(object):
  """
  This class allows functions to be registered (similar to atexit)
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
        print("Clean up handler Calling " + str(call.function.__name__) + '(' + \
              str(call.nargs) + ', ' + str(call.kargs) + ')')
      call.function(*(call.nargs), **(call.kargs))

  def clear(self):
    """
      Remove all registered calls
    """
    self.calls = [ ]

    assert len(self.calls) == 0

cleanUpHandler = BatchCaller()

""" Timing for the toolchain pipeline """
Tools = ["clang", "opt", "bugle", "gpuverifyvcgen", "gpuverifycruncher", "gpuverifyboogiedriver"]
Extensions = { 'clang': ".bc", 'opt': ".opt.bc", 'bugle': ".gbpl", 'gpuverifyvcgen': ".bpl", 'gpuverifycruncher': ".cbpl" }

""" WindowsError is not defined on UNIX systems, this works around that """
try:
  WindowsError
except NameError:
  class WindowsError(Exception):
    pass


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

clangCoreDefines = [ ]

clangCoreOptions = [ "-Wall",
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
                       "-fno-builtin",
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
    self.sourceFiles = [] # The OpenCL or CUDA files to be processed
    self.includes = clangCoreIncludes
    self.defines = clangCoreDefines
    self.clangOptions = list(clangCoreOptions) # Make sure we make a copy so we don't change the global list
    self.optOptions = [ "-mem2reg", "-globaldce" ]
    self.defaultOptions = [ "/nologo", "/typeEncoding:m", "/mv:-",
                       "/doModSetAnalysis", "/useArrayTheory",
                       "/doNotUseLabels", "/enhancedErrorMessages:1"
                     ]
    self.vcgenOptions = [ "/noPruneInfeasibleEdges" ]
    self.cruncherOptions = []
    self.boogieOptions = []
    self.bugleOptions = []
    self.invInferConfigFile = "inference.cfg"
    self.skip = { "clang": False,
             "opt": False,
             "bugle": False,
             "vcgen": False,
             "cruncher": False }

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
      if psutilPresent:
        children = psutil.Process(self.popenObject.pid).get_children(True)
      self.popenObject.terminate()
      if psutilPresent:
        for child in children:
          child.terminate()

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


def showVersionAndExit():
  """ This will check if using gpuverify from development directory.
      If so this will invoke Mercurial to find out version information.
      If this is a deployed version we will try to read the version from
      a file instead
  """
  print(getversion.getVersionString())
  raise GPUVerifyException(ErrorCodes.SUCCESS)

def GPUVerifyWarn(msg):
  print("GPUVerify: warning: " + msg)

class GPUVerifyArgumentParser(argparse.ArgumentParser):
  def error (self, message):
    raise GPUVerifyException(ErrorCodes.COMMAND_LINE_ERROR, message)

class ShowVersionAction(argparse.Action):
  def __call__ (self,a,b,c,d=None):
    showVersionAndExit()

class ldict(dict):
  def __getattr__ (self, method_name):
    if method_name in self:
      return self[method_name]
    else:
      raise AttributeError, method_name

def parse_args(argv):
    parser = GPUVerifyArgumentParser(description="GPUVerify frontend", usage="gpuverify [options] <kernel>")

    parser.add_argument("kernel", nargs='?', type=file, help="a kernel to verify") # nargs='?' because of needing to put KI in this script

    general = parser.add_argument_group("GENERAL OPTIONS")

    general.add_argument("--version", nargs=0, action=ShowVersionAction)

    general.add_argument("-D", dest='defines',  action='append', help="Define symbol", metavar="<value>")
    general.add_argument("-I", dest='includes', action='append', help="Add directory to include search path", metavar="<value>")

    mode = general.add_mutually_exclusive_group()
    mode.add_argument("--findbugs", dest='mode', action='store_const', const=AnalysisMode.FINDBUGS, help="Run tool in bug-finding mode")
    mode.add_argument("--verify",   dest='mode', action='store_const', const=AnalysisMode.VERIFY, help="Run tool in verification mode")

    general.add_argument("--loop-unwind=", type=non_negative, help="Explore traces that pass through at most X loop heads", metavar="X") #default=2

    general.add_argument("--no-benign",        action='store_true', help="Do not tolerate benign data races")
    general.add_argument("--only-divergence",  action='store_true', help="Only check for barrier divergence, not races")
    general.add_argument("--only-intra-group", action='store_true', help="Do not check for inter-group races")

    verbosity = general.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action='store_true', help="Show subcommands and use verbose output")
    verbosity.add_argument("--silent",  action='store_true', help="Silent on success; only show errors/timing")

    general.add_argument("--time", action='store_true', help="Show timing information")
    general.add_argument("--time-as-csv=", help="Print timing as CSV with label X", metavar="X")

    general.add_argument("--timeout=", type=non_negative, default=300, help="Allow each component to run for X seconds before giving up. A timout of 0 disables the timeout. Default is 300s", metavar="X")
    general.add_argument("--memout=",  type=non_negative, default=0,   help="Give Boogie a hard memory limit of X megabytes. A memout of 0 disables the memout. Default is unlimited.", metavar="X")

    general.add_argument("--opencl", dest='source_language', action='store_const', const=SourceLanguage.OpenCL, help="Force OpenCL")
    general.add_argument("--cuda",   dest='source_language', action='store_const', const=SourceLanguage.CUDA,   help="Force CUDA")

    sizing = parser.add_argument_group("SIZING")
    lsize = sizing.add_mutually_exclusive_group()
    numg = sizing.add_mutually_exclusive_group()

    lsize.add_argument("--local_size=",    dest='group_size', type=dimensions, help="Specify whether a work-group is 1D, 2D, or 3D, and specify size for each dimension. This corresponds to the `local_work_size` parameter of `clEnqueueNDRangeKernel`. Use * for an unconstrained value", metavar="(X|X,Y|X,Y,Z)")
    sizing.add_argument("--global_size=", dest='global_size', type=dimensions, help="Specify whether the NDRange is 1D, 2D, or 3D, and specify size for each dimension. This corresponds to the `global_work_size` parameter of `clEnqueueNDRangeKernel`. Use * for an unconstrained value")
    numg.add_argument("--num_groups=",    dest='num_groups',  type=dimensions, help="Specify whether a grid of work-groups is 1D, 2D, or 3D, and specify size for each dimension. Use * for an unconstrained value")

    lsize.add_argument("--blockDim=",     dest='group_size',  type=dimensions, help="Specify thread block size. Synonym for --local_size")
    numg.add_argument("--gridDim=",       dest='num_groups',  type=dimensions, help="Specify grid of thread blocks. Synonym for --num_groups")

    advanced = parser.add_argument_group("ADVANCED OPTIONS")

    bitwidth = advanced.add_mutually_exclusive_group()
    bitwidth.add_argument("--32-bit", dest='size_t', action='store_const', const=32, help="Assume 32-bit pointer size (default)")
    bitwidth.add_argument("--64-bit", dest='size_t', action='store_const', const=64, help="Assume 64-bit pointer size")

    abstraction = advanced.add_mutually_exclusive_group()
    abstraction.add_argument("--adversarial-abstraction", action='store_true', help="Completely abstract shared state, so that reads are non-deterministic")
    abstraction.add_argument("--equality-abstraction",    action='store_true', help="Make shared arrays non-deterministic, but consistent between threads, at barriers")

    advanced.add_argument("--array-equalities",   action='store_true', help="Generate equality candidate invariants for array variables")
    advanced.add_argument("--asymmetric-asserts", action='store_true', help="Emit assertions only for the first thread. Sound, and may lead to faster verification, but can yield false positives")

    advanced.add_argument("--boogie-file=", type=file, action='append', help="Specify a supporting .bpl file to be used during verification", metavar="X.bpl")
    advanced.add_argument("--bugle-lang=", dest='source_language', choices=["cl","cu"], type=lambda x: SourceLanguage.OpenCL if x == "cl" else SourceLanguage.CUDA, help="Bitcode language if passing in a bitcode file")

    advanced.add_argument("--call-site-analysis",           action='store_true', help="Turn on call site analysis")
    advanced.add_argument("--math-int",                     action='store_true', help="Represent integer types using mathematical integers instead of bit-vectors")
    advanced.add_argument("--inverted-tracking",            action='store_true', help="Use do_not_track instead of track. This may result in faster detection of races for some solvers")
    advanced.add_argument("--no-annotations",               action='store_true', help="Ignore all source-level annotations")
    advanced.add_argument("--only-requires",                action='store_true', help="Ignore all source-level annotations except for requires")
    advanced.add_argument("--no-barrier-access-checks",     action='store_true', help="Turn off access checks for barrier invariants")
    advanced.add_argument("--no-constant-write-checks",     action='store_true', help="Turn off access checks for writes to constant space")
    advanced.add_argument("--no-inline",                    action='store_true', help="Turn off automatic inlining by Bugle")
    advanced.add_argument("--no-loop-predicate-invariants", action='store_true', help="Turn off automatic generation of loop invariants related to predicates, which can be incorrect")
    advanced.add_argument("--no-smart-predication",         action='store_true', help="Turn off smart predication")
    advanced.add_argument("--no-uniformity-analysis",       action='store_true', help="Turn off uniformity analysis")
    advanced.add_argument("--no-refined-atomics",           action='store_true', help="Disable return-value abstraction refinement for atomics")
    advanced.add_argument("--only-log",                     action='store_true', help="Log accesses to arrays, but do not check for aces. This can be useful for determining access pattern invariants")

    advanced.add_argument("--params=", type=params, help="If K is a kernel whose non-array parameters are (x1,...,xn), then it adds the precondition (x1==v1 && ... && xn=vn). An asterisk can be used to denote an unconstrained parameter", metavar="K,v1,...,vn")

    advanced.add_argument("--warp-sync=", type=non_negative, help="Synchronize threads within warps of size X. Defaults to 'resync' method, unless one of the following two options are set", metavar="X")
    twopass = advanced.add_mutually_exclusive_group()
    twopass.add_argument("--no-warp",   action='store_true', help="Only check inter-warp races")
    twopass.add_argument("--only-warp", action='store_true', help="Only check intra-warp races")

    advanced.add_argument("--race-instrumenter=", choices=["standard","watchdog-single","watchdog-multiple"], default="standard", help="Choose which method of race instrumentation to use")
    advanced.add_argument("--solver=", choices=["z3","cvc4"], default="z3",                                     help="Choose which SMT theorem prover to use in the backend. Default is z3")
    advanced.add_argument("--logic=", choices=["ALL_SUPPORTED","QF_ALL_SUPPORTED"], default="QF_ALL_SUPPORTED", help="Define the logic for the cvc4 SMT solver backend. Default is QF_ALL_SUPPORTED")

    development = parser.add_argument_group("DEVELOPMENT OPTIONS")
    development.add_argument("--debug",         action='store_true', help="Enable debugging of GPUVerify components: exceptions will not be suppressed")
    development.add_argument("--keep-temps",    action='store_true', help="Keep intermediate bc, gbpl, bpl, and cbpl files")
    development.add_argument("--gen-smt2",      action='store_true', help="Generate smt2 file")

    development.add_argument("--clang-opt=",    dest='clang_options',    action='append', help="Specify option to be passed to Clang")
    development.add_argument("--opt-opt=",      dest='opt_options',      action='append', help="Specify option to be passed to LLVM optimization pass")
    development.add_argument("--bugle-opt",     dest='bugle_options',    action='append', help="Specify option to be passed to Bugle")
    development.add_argument("--vcgen-opt=",    dest='vcgen_options',    action='append', help="Specify option to be passed to VC generation")
    development.add_argument("--cruncher-opt=", dest='cruncher_options', action='append', help="Specify option to be passed to invariant cruncher")
    development.add_argument("--boogie-opt=",   dest='boogie_options',   action='append', help="Specify option to be passed to Boogie")

    development.add_argument("--stop-at-opt",  dest='stop', action='store_const', const="opt",      help="Stop after LLVM optimization pass")
    development.add_argument("--stop-at-gbpl", dest='stop', action='store_const', const="bugle",    help="Stop after generating gbpl")
    development.add_argument("--stop-at-bpl",  dest='stop', action='store_const', const="vcgen",    help="Stop after generating bpl")
    development.add_argument("--stop-at-cbpl", dest='stop', action='store_const', const="cruncher", help="Stop after generating an annotated bpl")

    inference = parser.add_argument_group("INVARIANT INFERENCE OPTIONS")
    inference.add_argument("--no-infer", dest='inference', action='store_false', help="Turn off invariant inference") # original also has --noinfer
    inference.add_argument("--omit-infer=", action='append', help="Do not generate invariants tagged 'X'") 
    inference.add_argument("--staged-inference",   action='store_true', help="Perform invariant inference in stages; this can boost performance for complex kernels (but this is not guaranteed)")
    inference.add_argument("--parallel-inference", action='store_true', help="Use multiple solver instances in parallel to acceleate invariant inference (but this is not guaranteed)")
    inference.add_argument("--dynamic-analysis",   action='store_true', help="Use dynamic analysis to falsify invariants")
    inference.add_argument("--scheduling=", choices=["all-together","unsound-first"], help="Choose a parallel scheduling strategy from the following: 'unsound-first' or 'all-together'. By default the scheduler executes first any dynamic engines, then any unsound static engines, and the sound static engines. The 'unsound-first' strategy executes any unsound engines (either static or dynamic) together before the sound engines. The 'all-together' strategy executes all engines together")
    inference.add_argument("--infer-sliding=",       type=non_negative, default=0, help="Potentially launches a new refutation engine every X seconds", metavar="X")
    inference.add_argument("--infer-config-file=", default="inference.cfg", help="Specify a custom configuration file to be used during invariant inference", metavar="X.cfg")
    inference.add_argument("--infer-info",         action='store_true', help="Prints information about the inference process")
    inference.add_argument("--k-induction-depth=", type=non_negative, default=-1, help="Applies k-induction with k=X to all loops", metavar="X")
    inference.add_argument("--refutation-engine=", choices=["houdini","dynamic","lmi","lei","lu1","lu2"], help="Chose a refutation engine from the following: 'houdini', 'dynamic', 'lmi' (ignore loop-maintained invariants), 'lei' (ignore loop-entry invariants), or 'lu' (loop unrolling). If an unsound refutation engine is chosen, the result cannot be trusted", metavar="X")

    undocumented = parser.add_argument_group("UNDOCUMENTED")

    undocumented.add_argument("--infer-sliding-limit=", type=non_negative, default=0)
    undocumented.add_argument("--delay-houdini=",       type=non_negative, default=0)
    undocumented.add_argument("--dynamic-error-limit=", type=non_negative, default=0)
    undocumented.add_argument("--debug-houdini", action='store_true')

    interceptor = parser.add_argument_group("BATCH PROCESSING")
    interceptor.add_argument("--show-intercepted", action='store_true')
    interceptor.add_argument("--check-intercepted=", type=non_negative, metavar="X")
    interceptor.add_argument("--check-all-intercepted", action='store_true')

    if len(argv) == 0:
      argv = [ '--help' ]

    args = vars(parser.parse_args(argv))

    # Technically unnecessary but I like it prettier
    args = ldict((k[:-1],v) if k.endswith('=') else (k,v) for k,v in args.iteritems())

    args['batch_mode'] = any(args[x] for x in ['show_intercepted','check_intercepted','check_all_intercepted'])

    if not args['batch_mode']:
      if not args['kernel']:
        parser.error("Must provide a kernel for normal mode")
      name = args['kernel'].name
      _unused,ext = SplitFilenameExt(name)
    
      starts = defaultdict(lambda : "clang", { '.bc': "opt", '.opt.bc': "bugle", '.gbpl': "vcgen", '.bpl.': "cruncher", '.cbpl': "boogie" })
      args['start'] = starts[ext]

      if not args['stop']:
        args['stop'] = "boogie"

      if not args['source_language']:
        if ext == '.cl':
          args['source_language'] = SourceLanguage.OpenCL
        elif ext == '.cu':
          args['source_language'] = SourceLanguage.CUDA
        else:
          if ext == '.bc':
            proc = subprocess.Popen([gvfindtools.llvmBinDir + "/llvm-nm", name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            lines,stderr = proc.communicate()
          else:
            lines = ''.join(args['kernel'].readlines())
          if any(x in lines for x in ["get_local_size","get_global_size","__kernel"]):
            args['source_language'] = SourceLanguage.OpenCL
          elif "blockDim" in lines:
            args['source_language'] = SourceLanguage.CUDA
          else:
            parser.error("Could not infer source language")

      # Use '//' to ensure flooring division for Python3
      if args['num_groups'] and args['global_size']:
        parser.error("--num_groups and --global_size are mutually exclusive.")
      elif args['group_size'] and args['global_size']:
        if len(args['group_size']) != len(args['global_size']):
          parser.error("Dimensions of --local_size and --global_size must match.")
        args['num_groups'] = [(a//b) for (a,b) in zip(args['global_size'],args['group_size'])]
        for i in (i for (a,b,c,i) in zip(args.num_groups,args.group_size,args.global_size,range(len(args.num_groups))) if a * b != c):
          parser.error("Dimension " + str(i) + " of global_size does not divide by the same dimension in local_size")
      elif args['group_size'] and args['num_groups']:
        pass
      else:
        parser.error("Not enough size arguments")

      if args['verbose']:
        print("Got {} groups of size {}".format("x".join(map(str,args['num_groups'])), "x".join(map(str,args['group_size']))))

    if not args['size_t']:
        args['size_t'] = 32

    if args.loop_unwind:
      args.mode = AnalysisMode.FINDBUGS
    if args.mode == AnalysisMode.FINDBUGS and not args.loop_unwind:
      args.loop_unwind = 2

    if not args['mode']:
        args['mode'] = AnalysisMode.ALL

    return args

def dimensions (string):
  string = string.strip()
  string = string[1:-1] if string[0]+string[-1] == "[]" else string
  values = map(lambda x: x if x == '*' else int(x), string.split(","))
  if not (1 <= len(values) and len(values) <= 3) or len([x for x in values if x > 0]) < len(values):
    raise argparse.ArgumentTypeError("Dimensions must be a vector of 1-3 positive integers")
  return values

def params (string):
  string = string.strip()
  string = string[1:-1] if string[0]+string[-1] == "[]" else string
  values = string.split(",")
  values = values[:1] + map(lambda x: x if x == '*' else int(x), values[1:])
  return values

def non_negative (string):
  try:
    i = int(string)
    if i < 0:
      raise argparse.ArgumentTypeError("negative value " + i + " provided as argument")
    return i
  except ValueError:
    raise argparse.ArgumentTypeError("Argument must be a positive integer")


def processOptions(args):
  CommandLineOptions = copy.deepcopy(DefaultCmdLineOptions())
  _f, ext = SplitFilenameExt(args['kernel'].name)
  if ext in [ ".bc", ".opt.bc", ".gbpl", ".bpl", ".cbpl" ]:
    CommandLineOptions.skip["clang"] = True
  if ext in [        ".opt.bc", ".gbpl", ".bpl", ".cbpl" ]:
    CommandLineOptions.skip["opt"] = True
  if ext in [                   ".gbpl", ".bpl", ".cbpl" ]:
    CommandLineOptions.skip["bugle"] = True
  if ext in [                            ".bpl", ".cbpl" ]:
    CommandLineOptions.skip["vcgen"] = True
  if ext in [                                    ".cbpl" ]:
    CommandLineOptions.skip["cruncher"] = True

  CommandLineOptions.sourceFiles.append(args['kernel'].name)

  CommandLineOptions.defines += args['defines'] or []
  CommandLineOptions.includes += args['includes'] or []
  if args['no_annotations'] or args['only_requires']:
    # Must be added after include of opencl or cuda header
    noAnnotationsHeader = [ "-include", "annotations/no_annotations.h" ]
    clangOpenCLOptions.extend(noAnnotationsHeader)
    clangCUDAOptions.extend(noAnnotationsHeader)
    if args['only_requires']:
      clangOpenCLDefines.append("ONLY_REQUIRES")
      clangCUDADefines.append("ONLY_REQUIRES")

  CommandLineOptions.clangOptions += sum([a.split(" ") for a in args['clang_options'] or []],[])
  CommandLineOptions.optOptions += sum([a.split(" ") for a in args['opt_options'] or []],[])
  CommandLineOptions.bugleOptions += sum([a.split(" ") for a in args['bugle_options'] or []],[])
  CommandLineOptions.vcgenOptions += sum([a.split(" ") for a in args['vcgen_options'] or []],[])
  CommandLineOptions.cruncherOptions += sum([a.split(" ") for a in args['cruncher_options'] or []],[])
  CommandLineOptions.boogieOptions += sum([a.split(" ") for a in args['boogie_options'] or []],[])
  
  CommandLineOptions.vcgenOptions += ["/noCandidate:"+a for a in args['omit_infer'] or []]
  CommandLineOptions.vcgenOptions += [ "/params:" + ','.join(map(str,args['params'])) ] if args['params'] else []

  CommandLineOptions.cruncherOptions += [x.name for x in args['boogie_file'] or []] or []
  CommandLineOptions.invInferConfigFile = args['infer_config_file'] or "inference.cfg"

  CommandLineOptions.cruncherOptions += [ "/blockHighestDim:" + str(len(args.group_size) - 1) ]
  CommandLineOptions.boogieOptions += [ "/blockHighestDim:" + str(len(args.group_size) - 1) ]
  CommandLineOptions.cruncherOptions += [ "/gridHighestDim:" + str(len(args.num_groups) - 1) ]
  CommandLineOptions.boogieOptions += [ "/gridHighestDim:" + str(len(args.num_groups) - 1) ]

  if args['source_language'] == SourceLanguage.CUDA:
    CommandLineOptions.cruncherOptions += [ "/sourceLanguage:cu" ]
    CommandLineOptions.boogieOptions += [ "/sourceLanguage:cu" ]
  
  return CommandLineOptions

class GPUVerifyInstance (object):
  def Verbose (self, msg):
    if (self.verbose):
      print(msg)

  def __init__ (self, args):
    """
    This function should NOT be called directly instead call main()
    It is assumed that argv has had sys.argv[0] removed
    """

    self.timing = {}

    CommandLineOptions = processOptions(args)
    args.bugle_lang = "cl" if args.source_language == SourceLanguage.OpenCL else "cu"

    self.stop = args.stop

    cleanUpHandler.setVerbose(args.verbose)

    filename, ext = SplitFilenameExt(args['kernel'].name)

    CommandLineOptions.defines += [ '__BUGLE_' + str(args.size_t) + '__' ]
    if (args.size_t == 32):
      CommandLineOptions.clangOptions += [ "-target", "nvptx--" ]
    elif (args.size_t == 64):
      CommandLineOptions.clangOptions += [ "-target", "nvptx64--" ]
    else:
      raise GPUVerifyException(ErrorCodes.COMMAND_LINE_ERROR, "unknown size_t size '" + str(args.size_t) + "'")

    if args.source_language == SourceLanguage.OpenCL:
      CommandLineOptions.clangOptions += clangOpenCLOptions
      CommandLineOptions.clangOptions += clangInlineOptions
      CommandLineOptions.includes += clangOpenCLIncludes
      CommandLineOptions.defines += clangOpenCLDefines
      CommandLineOptions.defines.append("__" + str(len(args.group_size)) + "D_WORK_GROUP")
      CommandLineOptions.defines.append("__" + str(len(args.num_groups)) + "D_GRID")
      for (index, value) in enumerate(args.group_size):
        if value == '*':
          CommandLineOptions.defines += [ "__LOCAL_SIZE_" + str(index) + "_FREE" ]
        else:
          CommandLineOptions.defines += [ "__LOCAL_SIZE_" + str(index) + "=" + str(value) ]
      for (index, value) in enumerate(args.num_groups):
        if value == '*':
          CommandLineOptions.defines += [ "__NUM_GROUPS_" + str(index) + "_FREE" ]
        elif type(value) is tuple:
          CommandLineOptions.defines += [ "__NUM_GROUPS_" + str(index) + "_FREE" ]
          CommandLineOptions.defines += [ "__GLOBAL_SIZE_" + str(index) + "=" + str(value[1]) ]
        else:
          CommandLineOptions.defines += [ "__NUM_GROUPS_" + str(index) + "=" + str(value) ]
      if (args.size_t == 32):
        CommandLineOptions.clangOptions += [ "-Xclang", "-mlink-bitcode-file",
                                             "-Xclang", gvfindtools.libclcInstallDir + "/lib/clc/nvptx--.bc" ]
      elif (args.size_t == 64):
        CommandLineOptions.clangOptions += [ "-Xclang", "-mlink-bitcode-file",
                                             "-Xclang", gvfindtools.libclcInstallDir + "/lib/clc/nvptx64--.bc" ]

    elif args.source_language == SourceLanguage.CUDA:
      CommandLineOptions.clangOptions += clangCUDAOptions
      CommandLineOptions.includes += clangCUDAIncludes
      CommandLineOptions.defines += clangCUDADefines
      CommandLineOptions.defines.append("__" + str(len(args.group_size)) + "D_THREAD_BLOCK")
      CommandLineOptions.defines.append("__" + str(len(args.num_groups)) + "D_GRID")
      for (index, value) in enumerate(args.group_size):
        if value == '*':
          CommandLineOptions.defines += [ "__BLOCK_DIM_" + str(index) + "_FREE" ]
        else:
          CommandLineOptions.defines += [ "__BLOCK_DIM_" + str(index) + "=" + str(value) ]
      for (index, value) in enumerate(args.num_groups):
        if value == '*':
          CommandLineOptions.defines += [ "__GRID_DIM_" + str(index) + "_FREE" ]
        else:
          CommandLineOptions.defines += [ "__GRID_DIM_" + str(index) + "=" + str(value) ]

    # Intermediate filenames
    bcFilename = filename + '.bc'
    optFilename = filename + '.opt.bc'
    gbplFilename = filename + '.gbpl'
    cbplFilename = filename + '.cbpl'
    bplFilename = filename + '.bpl'
    locFilename = filename + '.loc'
    smt2Filename = filename + '.smt2'
    if not args.keep_temps:
      inputFilename = filename + ext
      def DeleteFile(filename):
        """ Delete the filename if it exists; but don't delete the original input """
        if filename == inputFilename: return
        try: os.remove(filename)
        except OSError: pass
      cleanUpHandler.register(DeleteFile, bcFilename)
      if not args.stop == 'opt': cleanUpHandler.register(DeleteFile, optFilename)
      if not args.stop == 'bugle': cleanUpHandler.register(DeleteFile, gbplFilename)
      if not args.stop == 'bugle': cleanUpHandler.register(DeleteFile, locFilename)
      if not args.stop == 'cruncher': cleanUpHandler.register(DeleteFile, cbplFilename)
      if not args.stop == 'vcgen': cleanUpHandler.register(DeleteFile, bplFilename)

    CommandLineOptions.clangOptions += ["-o", bcFilename]
    CommandLineOptions.clangOptions += ["-x", "cl" if args.source_language == SourceLanguage.OpenCL else "cuda", (filename + ext)]

    CommandLineOptions.optOptions += [ "-o", optFilename, bcFilename ]

    if args.start == 'clang':
      CommandLineOptions.bugleOptions += [ "-l", "cl" if args.source_language == SourceLanguage.OpenCL else "cu", "-s", locFilename, "-o", gbplFilename, optFilename ]
    elif not CommandLineOptions.skip['bugle']:
      lang = args.bugle_lang
      if not lang: # try to infer
        try:
          proc = subprocess.Popen([ gvfindtools.llvmBinDir + "/llvm-nm", filename + ext ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
          stdout, stderr = proc.communicate()
          if "get_local_size" in stdout: lang = 'cl'
          if "blockDim" in stdout: lang = 'cu'
        except: pass
      if not lang:
        raise GPUVerifyException(ErrorCodes.COMMAND_LINE_ERROR, "must specify --bugle-lang=[cl|cu] when given a bitcode .bc file")
      assert lang in [ "cl", "cu" ]
      CommandLineOptions.bugleOptions += [ "-l", lang, "-s", locFilename, "-o", gbplFilename, optFilename ]

    if args.math_int:
      CommandLineOptions.bugleOptions += [ "-i", "math" ]
      CommandLineOptions.vcgenOptions += [ "/mathInt" ]
    else:
      CommandLineOptions.cruncherOptions += [ "/proverOpt:OPTIMIZE_FOR_BV=true" ]
      CommandLineOptions.boogieOptions += [ "/proverOpt:OPTIMIZE_FOR_BV=true" ]
      if args.solver == "z3":
        CommandLineOptions.cruncherOptions += [ "/z3opt:RELEVANCY=0", "/z3opt:SOLVER=true" ]
        CommandLineOptions.boogieOptions += [ "/z3opt:RELEVANCY=0", "/z3opt:SOLVER=true" ]
      
    if not args.no_inline:
      CommandLineOptions.bugleOptions += [ "-inline" ]

    if args.warp_sync:
      CommandLineOptions.vcgenOptions += [ "/doWarpSync:" + str(args.warp_sync) ]
      if args.no_warp:
        CommandLineOptions.vcgenOptions += [ "/noWarp" ]
      if args.only_warp:
        CommandLineOptions.vcgenOptions += [ "/onlyWarp" ]
    if args.no_refined_atomics:
      CommandLineOptions.vcgenOptions += [ "/noRefinedAtomics" ]
    if args.adversarial_abstraction:
      CommandLineOptions.vcgenOptions += [ "/adversarialAbstraction" ]
    if args.equality_abstraction:
      CommandLineOptions.vcgenOptions += [ "/equalityAbstraction" ]
    if args.no_benign:
      CommandLineOptions.vcgenOptions += [ "/noBenign" ]
    if args.only_divergence:
      CommandLineOptions.vcgenOptions += [ "/onlyDivergence" ]
    if args.only_intra_group:
      CommandLineOptions.vcgenOptions += [ "/onlyIntraGroupRaceChecking" ]
      CommandLineOptions.cruncherOptions += [ "/onlyIntraGroupRaceChecking" ]
      CommandLineOptions.boogieOptions += [ "/onlyIntraGroupRaceChecking" ]
    if args.only_log:
      CommandLineOptions.vcgenOptions += [ "/onlyLog" ]
    if args.mode == AnalysisMode.FINDBUGS or (not args.inference):
      CommandLineOptions.vcgenOptions += [ "/noInfer" ]
    if args.no_barrier_access_checks:
      CommandLineOptions.vcgenOptions += [ "/noBarrierAccessChecks" ]
    if args.no_constant_write_checks:
      CommandLineOptions.vcgenOptions += [ "/noConstantWriteChecks" ]
    if args.no_loop_predicate_invariants:
      CommandLineOptions.vcgenOptions += [ "/noLoopPredicateInvariants" ]
    if args.no_smart_predication:
      CommandLineOptions.vcgenOptions += [ "/noSmartPredication" ]
    if args.no_uniformity_analysis:
      CommandLineOptions.vcgenOptions += [ "/noUniformityAnalysis" ]
    if args.inverted_tracking:
      CommandLineOptions.vcgenOptions += [ "/invertedTracking" ]
    if args.asymmetric_asserts:
      CommandLineOptions.vcgenOptions += [ "/asymmetricAsserts" ]
    if args.staged_inference:
      CommandLineOptions.vcgenOptions += [ "/stagedInference" ]
      CommandLineOptions.cruncherOptions += [ "/stagedInference" ]
    if args.call_site_analysis:
      CommandLineOptions.vcgenOptions += [ "/callSiteAnalysis" ]

    CommandLineOptions.vcgenOptions += [ "/print:" + filename, gbplFilename ] #< ignore .bpl suffix

    if args.mode == AnalysisMode.FINDBUGS:
      CommandLineOptions.boogieOptions += [ "/loopUnroll:" + str(args.loop_unwind) ]

    if args.k_induction_depth >= 0:
      CommandLineOptions.cruncherOptions += [ "/kInductionDepth:" + str(args.k_induction_depth) ]
      CommandLineOptions.boogieOptions += [ "/kInductionDepth:" + str(args.k_induction_depth) ]

    if args.memout > 0:
      CommandLineOptions.cruncherOptions.append("/z3opt:-memory:" + str(args.memout))
      CommandLineOptions.boogieOptions.append("/z3opt:-memory:" + str(args.memout))

    CommandLineOptions.cruncherOptions += [ "/noinfer" ]
    CommandLineOptions.cruncherOptions += [ "/contractInfer" ]
    CommandLineOptions.cruncherOptions += [ "/concurrentHoudini" ]
  
    if args.refutation_engine:
      CommandLineOptions.cruncherOptions += [ "/refutationEngine:" + args.refutation_engine ]
      self.stop = 'cruncher'
    if args.infer_info:
      CommandLineOptions.cruncherOptions += [ "/inferInfo" ]
      CommandLineOptions.cruncherOptions += [ "/trace" ]
    if args.debug_houdini:
      CommandLineOptions.cruncherOptions += [ "/debugConcurrentHoudini" ]
  
    if args.parallel_inference:
      CommandLineOptions.cruncherOptions += [ "/parallelInference" ]
      if args.infer_sliding > 0:
        CommandLineOptions.cruncherOptions += [ "/inferenceSliding:" + str(args.infer_sliding) ]
        CommandLineOptions.cruncherOptions += [ "/parallelInferenceScheduling:all-together" ]
        if args.infer_sliding_limit > 0:
          CommandLineOptions.cruncherOptions += [ "/inferenceSlidingLimit:" + str(args.infer_sliding_limit) ]
        else:
          CommandLineOptions.cruncherOptions += [ "/inferenceSlidingLimit:1" ]
      elif args.delay_houdini > 0:
        CommandLineOptions.cruncherOptions += [ "/delayHoudini:" + str(args.delay_houdini) ]
        CommandLineOptions.cruncherOptions += [ "/parallelInferenceScheduling:all-together" ]
      else:
        CommandLineOptions.cruncherOptions += [ "/parallelInferenceScheduling:" + args.scheduling ]
      if args.dynami_error_limit > 0:
        CommandLineOptions.cruncherOptions += [ "/dynamicErrorLimit:" + str(args.dynamic_error_limit) ]
  
    if args.dynamic_analysis:
      CommandLineOptions.cruncherOptions += [ "/dynamicAnalysis" ]
    CommandLineOptions.cruncherOptions += [ "/z3exe:" + gvfindtools.z3BinDir + os.sep + "z3.exe" ]
    CommandLineOptions.cruncherOptions += [ "/cvc4exe:" + gvfindtools.cvc4BinDir + os.sep + "cvc4.exe" ]

    if args.solver == "cvc4":
      CommandLineOptions.cruncherOptions += [ "/proverOpt:SOLVER=cvc4" ]
      CommandLineOptions.boogieOptions += [ "/proverOpt:SOLVER=cvc4" ]
      CommandLineOptions.boogieOptions += [ "/cvc4exe:" + gvfindtools.cvc4BinDir + os.sep + "cvc4.exe" ]
      CommandLineOptions.cruncherOptions += [ "/proverOpt:LOGIC=" + args.logic ]
      CommandLineOptions.boogieOptions += [ "/proverOpt:LOGIC=" + args.logic ]
    else:
      CommandLineOptions.boogieOptions += [ "/z3exe:" + gvfindtools.z3BinDir + os.sep + "z3.exe" ]

    if args.gen_smt2:
      CommandLineOptions.cruncherOptions += [ "/proverLog:" + smt2Filename ]
      CommandLineOptions.boogieOptions += [ "/proverLog:" + smt2Filename ]
    if args.debug:
      CommandLineOptions.vcgenOptions += [ "/debugGPUVerify" ]
      CommandLineOptions.cruncherOptions += [ "/debugGPUVerify" ]
      CommandLineOptions.boogieOptions += [ "/debugGPUVerify" ]

    CommandLineOptions.cruncherOptions += CommandLineOptions.defaultOptions
    CommandLineOptions.boogieOptions += CommandLineOptions.defaultOptions
    CommandLineOptions.cruncherOptions += [ "/invInferConfigFile:" + os.path.dirname(os.path.abspath(__file__)) + os.sep + CommandLineOptions.invInferConfigFile ]
    CommandLineOptions.cruncherOptions += [ bplFilename ]

    if args.race_instrumenter == "watchdog-single":
      CommandLineOptions.bugleOptions += [ "-race-instrumentation=watchdog-single" ]
      CommandLineOptions.vcgenOptions += [ "/watchdogRaceChecking:SINGLE" ]
      CommandLineOptions.cruncherOptions += [ "/watchdogRaceChecking:SINGLE" ]
      CommandLineOptions.boogieOptions += [ "/watchdogRaceChecking:SINGLE" ]
    if args.race_instrumenter == "watchdog-multiple":
      CommandLineOptions.bugleOptions += [ "-race-instrumentation=watchdog-multiple" ]
      CommandLineOptions.vcgenOptions += [ "/watchdogRaceChecking:MULTIPLE" ]
      CommandLineOptions.cruncherOptions += [ "/watchdogRaceChecking:MULTIPLE" ]
      CommandLineOptions.boogieOptions += [ "/watchdogRaceChecking:MULTIPLE" ]

    if args.inference and (not args.mode == AnalysisMode.FINDBUGS):
      CommandLineOptions.boogieOptions += [ cbplFilename ]
    else:
      CommandLineOptions.boogieOptions += [ bplFilename ]
      CommandLineOptions.skip['cruncher'] = True

    self.includes = CommandLineOptions.includes
    self.defines = CommandLineOptions.defines
    self.clangOptions = CommandLineOptions.clangOptions
    self.optOptions = CommandLineOptions.optOptions
    self.bugleOptions = CommandLineOptions.bugleOptions
    self.vcgenOptions = CommandLineOptions.vcgenOptions
    self.cruncherOptions = CommandLineOptions.cruncherOptions
    self.boogieOptions = CommandLineOptions.boogieOptions

    self.skip = CommandLineOptions.skip
    self.mode = args.mode
    self.sourceFiles = CommandLineOptions.sourceFiles
    self.SL = args.source_language
    self.loopUnwindDepth = args.loop_unwind
    self.onlyDivergence = args.only_divergence
    self.onlyIntraGroup = args.only_intra_group

    self.verbose = args.verbose
    self.silent = args.silent
    self.time = args.time or (args.time_as_csv is not None)
    self.timeCSVLabel = args.time_as_csv
    self.debugging = args.debug
    self.timeout = args.timeout

  def run(self, command,timeout=0):
    """ Run a command with an optional timeout. A timeout of zero
        implies no timeout.
    """
    popenargs={}
    if self.verbose:
      print(" ".join(command))
    else:
      popenargs['bufsize']=0
      if __name__ != '__main__':
        # We don't want messages to go to stdout if being used as module
        popenargs['stdout']=subprocess.PIPE

    if self.silent:
      popenargs['stdout']=subprocess.PIPE

    # Redirect stderr to whatever stdout is redirected to
    popenargs['stderr']=subprocess.STDOUT

    # Redirect stdin, othewise terminal text becomes unreadable after timeout
    popenargs['stdin']=subprocess.PIPE

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

    # We do not return stderr, as it was redirected to stdout
    return stdout, proc.returncode

  def getMonoCmdLine(self):
    if os.name == 'posix':
      # Check mono in path
      import distutils.spawn
      if distutils.spawn.find_executable('mono') == None:
        raise GPUVerifyException(ErrorCodes.CONFIGURATION_ERROR, "Could not find the mono executable in your PATH")
      if self.debugging:
        return [ 'mono' , '--debug' ]
      else:
        return ['mono']
    else:
      return [] # Presumably using Windows so don't need mono

  def RunTool(self,ToolName, Command, ErrorCode, timeout=0):
    """ Run a tool.
        If the timeout is set to 0 then there will be no timeout.
    """
    assert ToolName in Tools
    self.Verbose("Running " + ToolName)
    try:
      start = timeit.default_timer()
      stdout, returnCode = self.run(Command, timeout)
      end = timeit.default_timer()
    except Timeout:
      if self.time:
        self.timing[ToolName] = timeout
      raise GPUVerifyException(ErrorCodes.TIMEOUT, ToolName + " timed out. " + \
                               "Use --timeout=N with N > " + str(timeout)    + \
                               " to increase timeout, or --timeout=0 to "    + \
                               "disable timeout.")
    except (OSError,WindowsError) as e:
      raise GPUVerifyException(ErrorCode, "While invoking " + ToolName       + \
                               ": " + str(e) + "\nWith command line args:\n" + \
                               pprint.pformat(Command))
    if self.time:
      self.timing[ToolName] = end-start
    if returnCode != ErrorCodes.SUCCESS:
      if self.silent and stdout: print(stdout, file=sys.stderr)
      raise GPUVerifyException(ErrorCode, stdout)

  def invoke (self):
    """ RUN CLANG """
    if not self.skip["clang"]:
      self.RunTool("clang",
              [gvfindtools.llvmBinDir + "/clang"] +
              self.clangOptions +
              [("-I" + str(o)) for o in self.includes] +
              [("-D" + str(o)) for o in self.defines],
              ErrorCodes.CLANG_ERROR,
              self.timeout)

    """ RUN OPT """
    if not self.skip["opt"]:
      self.RunTool("opt",
              [gvfindtools.llvmBinDir + "/opt"] +
              self.optOptions,
              ErrorCodes.OPT_ERROR,
              self.timeout)

    if self.stop == 'opt': return 0

    """ RUN BUGLE """
    if not self.skip["bugle"]:
      self.RunTool("bugle",
              [gvfindtools.bugleBinDir + "/bugle"] +
              self.bugleOptions,
              ErrorCodes.BUGLE_ERROR,
              self.timeout)

    if self.stop == 'bugle': return 0

    """ RUN GPUVERIFYVCGEN """
    if not self.skip["vcgen"]:
      self.RunTool("gpuverifyvcgen",
              self.getMonoCmdLine() +
              [gvfindtools.gpuVerifyBinDir + "/GPUVerifyVCGen.exe"] +
              self.vcgenOptions,
              ErrorCodes.GPUVERIFYVCGEN_ERROR,
              self.timeout)

    if self.stop == 'vcgen': return 0

    """ RUN GPUVERIFYCRUNCHER """
    if not self.skip["cruncher"]:
      self.RunTool("gpuverifycruncher",
                self.getMonoCmdLine() +
                [gvfindtools.gpuVerifyBinDir + os.sep + "GPUVerifyCruncher.exe"] +
                self.cruncherOptions,
                ErrorCodes.BOOGIE_ERROR,
                self.timeout)

    if self.stop == 'cruncher': return 0

    """ RUN GPUVERIFYBOOGIEDRIVER """
    self.RunTool("gpuverifyboogiedriver",
            self.getMonoCmdLine() +
            [gvfindtools.gpuVerifyBinDir + "/GPUVerifyBoogieDriver.exe"] +
            self.boogieOptions,
            ErrorCodes.BOOGIE_ERROR,
            self.timeout)

    """ SUCCESS - REPORT STATUS """
    if self.silent:
      return 0

    if self.mode == AnalysisMode.FINDBUGS:
      print("No defects were found while analysing: " + ", ".join(self.sourceFiles))
      print("Notes:")
      print("- use --loop-unwind=N with N > " + str(self.loopUnwindDepth) + " to search for deeper bugs")
      print("- re-run in verification mode to try to prove absence of defects")
    else:
      print("Verified: " + ", ".join(self.sourceFiles))
      if not self.onlyDivergence:
        print("- no data races within " + ("work groups" if self.SL == SourceLanguage.OpenCL else "thread blocks"))
        if not self.onlyIntraGroup:
          print("- no data races between " + ("work groups" if self.SL == SourceLanguage.OpenCL else "thread blocks"))
      print("- no barrier divergence")
      print("- no assertion failures")
      print("(but absolutely no warranty provided)")

    return 0

  def showTiming(self, exitCode):
    if self.timeCSVLabel is not None:
      times = [ self.timing.get(tool, 0.0) for tool in Tools ]
      total = sum(times)
      times.append(total)
      row = [ '%.3f' % t for t in times ]
      label = self.timeCSVLabel
      if len(label) > 0: row.insert(0, label)
      if exitCode is ErrorCodes.SUCCESS:
        row.insert(1,'PASS')
      else:
        row.insert(1,'FAIL(' + str(exitCode) + ')')
      print(','.join(row))
    else:
      total = sum(self.timing.values())
      print("Timing information (%.2f secs):" % total)
      if self.timing:
        padTool = max([ len(tool) for tool in self.timing.keys() ])
        padTime = max([ len('%.3f secs' % t) for t in self.timing.values() ])
        for tool in Tools:
          if tool in self.timing:
            print("- %s : %s" % (tool.ljust(padTool), ('%.3f secs' % self.timing[tool]).rjust(padTime)))
      else:
        print("- no tools ran")

def subtool_args (args):
  pass_flags = { 'verbose': "--verbose", 'debug': "--debug" }
#, 'clang_options', 'opt_options', 'bugle_options', 'cruncher_options', 'boogie_options']
  return [pass_flags[x] for x,v in args.iteritems() if v and (x in pass_flags)]

def do_batch_mode (args):

  kernels = []
  for path, subdirs, files in os.walk(".gpuverify"):
    kernels += map(lambda x: path+os.sep+x, files)
  kernels = sorted(kernels)

  def parse_file (file):
    code = [x.rstrip() for x in file.readlines()]
    k_args = filter(lambda x: x != "", code[0][len("//"):].split(" "))
    opts = filter(lambda x: x != "", code[1][len("//"):].split(" "))
    return code,k_args,opts

  if args.show_intercepted:
    for index,file_name in enumerate(kernels):
      with open(file_name) as file:
        code,k_args,opts = parse_file(file)
        built = code[2][len("//"):]
        ran = code[3][len("//"):]
        print("["+str(index)+"] " + file_name+": " + ' '.join(k_args))
        print(built)
        print(ran)

  if args.check_intercepted:
    with open(kernels[args.check_intercepted]) as file:
      code,k_args,opts = parse_file(file)
      main(parse_args(subtool_args(args) + k_args + opts + [file.name]))

  if args.check_all_intercepted:
    for f in kernels:
      with open(f) as file:
        code,k_args,opts = parse_file(file)
        my_args = subtool_args(args) + k_args + opts + [f]
        print(my_args)
        try:
          main(parse_args(my_args))
        except GPUVerifyException as e:
          print(str(e), file=sys.stderr)
        except KeyboardInterrupt:
          raise


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
  gv_instance = GPUVerifyInstance(argv)
  def handleTiming (exitCode):
    if gv_instance.time:
      gv_instance.showTiming(exitCode)
    sys.stderr.flush()
    sys.stdout.flush()

  def doCleanUp(timing, exitCode=ErrorCodes.SUCCESS):
    if timing:
      # We must call this before cleaning up globals
      # because it depends on them
      cleanUpHandler.register(handleTiming, exitCode)

    # We should call this last.
    cleanUpHandler.call()
    cleanUpHandler.clear() # Clean up for next use

  try:
    gv_instance.invoke()
  except GPUVerifyException as e:
    doCleanUp(timing=True, exitCode=e.getExitCode())
    raise
  except Exception:
    # Something went very wrong
    doCleanUp(timing=False, exitCode=0) # It doesn't matter what the exitCode is
    raise

  doCleanUp(timing=True) # Do this outside try block so we don't call twice!
  return ErrorCodes.SUCCESS

debug = False

if __name__ == '__main__':
  """
  Entry point for GPUVerify as a script
  """

  # These are the exception error codes that won't be printed if they are thrown
  ignoredErrors = [ ErrorCodes.SUCCESS, ErrorCodes.BOOGIE_ERROR ]

  try:
    args = parse_args(sys.argv[1:])
    debug = args.debug
    if args.batch_mode:
      do_batch_mode(args)
    else:
      main(args)
  except GPUVerifyException as e:
    # We assume that globals are not cleaned up when running as a script so it 
    # is safe to read CommandLineOptions
    if (not (e.getExitCode() in ignoredErrors)) or debug:
      if e.getExitCode() == ErrorCodes.COMMAND_LINE_ERROR:
        # For command line errors only show the message and not internal details
        print('GPUVerify: {0}'.format(e.msg), file=sys.stderr)
      else:
        # Show all exception info for everything else not ignored
        print(str(e), file=sys.stderr)
    sys.exit(e.getExitCode())

  sys.exit(ErrorCodes.SUCCESS)
