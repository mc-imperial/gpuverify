#!/usr/bin/env python
# vim: set shiftwidth=2 tabstop=2 expandtab softtabstop=2:
from __future__ import print_function

import pickle
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
import tempfile
from collections import defaultdict, namedtuple
import copy
import distutils.spawn

if sys.version_info.major == 3:
  import io
else:
  # In python2.7 importing io.StringIO() doesn't work
  # very well because it expects unicode strings
  # use StringIO instead
  import StringIO as io

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

class ConfigurationError(Exception):
  def __init__ (self, msg):
    self.msg = msg
  def __str__ (self):
    return "GPUVerify: CONFIGURATION_ERROR error ({}): {}".format(ErrorCodes.CONFIGURATION_ERROR,self.msg)

# To properly kill child processes cross platform
try:
  import psutil
except ImportError:
  raise ConfigurationError("psutil required. "
                           "`pip install psutil` to get it.")

# Try to import the paths need for GPUVerify's tools
try:
  import gvfindtools
  # Initialise the paths (only needed for deployment version of gvfindtools.py)
  gvfindtools.init(sys.path[0])
except ImportError:
  raise ConfigurationError("Cannot find 'gvfindtools.py' "
                           "Did you forget to create it from a template?")

# WindowsError is not defined on UNIX systems, this works around that
try:
  WindowsError
except NameError:
  class WindowsError(Exception):
    pass

if os.name == 'posix':
  # Check mono in path
  if distutils.spawn.find_executable('mono') == None:
    raise ConfigurationError("Could not find the mono executable in your PATH")

class ArgumentParserError(Exception):
  def __init__ (self, msg):
    self.msg = msg
  def __str__ (self):
    return "GPUVerify: COMMAND_LINE_ERROR error ({}): {}".format(ErrorCodes.COMMAND_LINE_ERROR,self.msg)

class GPUVerifyException(Exception):
  def __init__(self, code, msg=None):
    self.code = code
    self.msg = msg

  def __str__(self):
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

class BatchCaller(object):
  """
  This class allows functions to be registered (similar to atexit)
  and later called using the call() method
  """

  def __init__(self, verbose=False):
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

cleanUpHandler = BatchCaller()

""" Timing for the toolchain pipeline """
Tools = ["clang", "opt", "bugle", "gpuverifyvcgen", "gpuverifycruncher", "gpuverifyboogiedriver"]
Extensions = { 'clang': ".bc", 'opt': ".opt.bc", 'bugle': ".gbpl", 'gpuverifyvcgen': ".bpl", 'gpuverifycruncher': ".cbpl" }



""" We support three analysis modes """
class AnalysisMode(object):
  """ ALL is the default mode.  Right now it is the same as VERIFY,
      but in future this mode will run verification and bug-finding in parallel
  """
  ALL=0
  FINDBUGS=1
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
    raise ConfigurationError('Could not find Bugle Inline Check plugin')

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
  This class defines some of the default options for the tool
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

def GPUVerifyWarn(msg):
  print("GPUVerify: warning: " + msg)

class GPUVerifyArgumentParser(argparse.ArgumentParser):
  def error (self, message):
    raise ArgumentParserError(message)

class ShowVersionAction(argparse.Action):
  def __call__ (self,a,b,c,d=None):
    """ This will check if using gpuverify from development directory.
        If so this will invoke Mercurial to find out version information.
        If this is a deployed version we will try to read the version from
        a file instead
    """
    print(getversion.getVersionString())
    sys.exit()

class ldict(dict):
  def __getattr__ (self, method_name):
    if method_name in self:
      return self[method_name]
    elif method_name == "batch_mode":
      return any(self[x] for x in
                 ['show_intercepted','check_intercepted',
                  'check_all_intercepted'])
    elif method_name == "dimensions":
      return len(self['group_size'])
    else:
      raise AttributeError(method_name)


def parse_args(argv):
    parser = GPUVerifyArgumentParser(description="GPUVerify frontend",
                                     usage="gpuverify [options] <kernel>")

    # nargs='?' because of needing to put KI in this script
    parser.add_argument("kernel", nargs='?', type=argparse.FileType('r'),
                        help="a kernel to verify")

    general = parser.add_argument_group("GENERAL OPTIONS")

    general.add_argument("--version", nargs=0, action=ShowVersionAction)

    general.add_argument("-D", dest='defines',  action='append',
                         help="Define symbol", metavar="<value>")
    general.add_argument("-I", dest='includes', action='append',
                         help="Add directory to include search path", metavar="<value>")

    mode = general.add_mutually_exclusive_group()
    mode.add_argument("--findbugs", dest='mode', action='store_const',
                      const=AnalysisMode.FINDBUGS, help="Run tool in bug-finding mode")
    mode.add_argument("--verify",   dest='mode', action='store_const',
                      const=AnalysisMode.VERIFY, help="Run tool in verification mode")

    general.add_argument("--loop-unwind=", type=non_negative, #default=2,
                         help="Explore traces that pass through at most X loop heads. \
                         Implies --findbugs",
                         metavar="X")

    general.add_argument("--no-benign",        action='store_true',
                         help="Do not tolerate benign data races")
    general.add_argument("--only-divergence",  action='store_true',
                         help="Only check for barrier divergence, not races")
    general.add_argument("--only-intra-group", action='store_true',
                         help="Do not check for inter-group races")

    verbosity = general.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action='store_true',
                           help="Show subcommands and use verbose output")
    verbosity.add_argument("--silent",  action='store_true',
                           help="Silent on success; only show errors/timing")

    general.add_argument("--time", action='store_true', help="Show timing information")
    general.add_argument("--time-as-csv=",
                         help="Print timing as CSV with label X", metavar="X")

    general.add_argument("--timeout=", type=non_negative, default=300,
                         help="Allow each component to run for X seconds before giving up. \
                         A timout of 0 disables the timeout. \
                         Default is 300s", metavar="X")
    general.add_argument("--memout=",  type=non_negative, default=0,
                         help="Give Boogie a hard memory limit of X megabytes. \
                         A memout of 0 disables the memout. \
                         Default is unlimited.", metavar="X")

    general.add_argument("--opencl", dest='source_language', action='store_const',
                         const=SourceLanguage.OpenCL, help="Force OpenCL")
    general.add_argument("--cuda",   dest='source_language', action='store_const',
                         const=SourceLanguage.CUDA,   help="Force CUDA")

    sizing = parser.add_argument_group("SIZING",
                                       "Define the dimensionality and size of each \
                                       dimension as a 1-, 2-, or 3-tuple, \
                                       optionally wrapped in []. E.g., [1,2] or 2,3,4. \
                                       Use * for an unconstrained value")
    lsize = sizing.add_mutually_exclusive_group()
    numg = sizing.add_mutually_exclusive_group()

    lsize.add_argument("--local_size=",    dest='group_size', type=dimensions,
                       help="Specify the dimensions of a work-group. \
                       This corresponds to the `local_work_size` parameter \
                       of `clEnqueueNDRangeKernel`.")
    sizing.add_argument("--global_size=", dest='global_size', type=dimensions,
                        help="Specify dimensions of the NDRange. \
                        This corresponds to the `global_work_size` parameter \
                        of `clEnqueueNDRangeKernel`. \
                        Mutually exclusive with --num_groups")
    numg.add_argument("--num_groups=",    dest='num_groups',  type=dimensions,
                      help="Specify the dimensions of a grid of work-groups. \
                      Mutually exclusive with --group_size")

    lsize.add_argument("--blockDim=",     dest='group_size',  type=dimensions,
                       help="Specify the thread block size.")
    numg.add_argument("--gridDim=",       dest='num_groups',  type=dimensions,
                      help="Specify the grid of thread blocks.")

    advanced = parser.add_argument_group("ADVANCED OPTIONS")

    bitwidth = advanced.add_mutually_exclusive_group()
    bitwidth.add_argument("--32-bit", dest='size_t', action='store_const', const=32,
                          help="Assume 32-bit pointer size (default)")
    bitwidth.add_argument("--64-bit", dest='size_t', action='store_const', const=64,
                          help="Assume 64-bit pointer size")

    abstraction = advanced.add_mutually_exclusive_group()
    abstraction.add_argument("--adversarial-abstraction", action='store_true',
                             help="Completely abstract shared state, \
                             so that reads are non-deterministic")
    abstraction.add_argument("--equality-abstraction",    action='store_true',
                             help="At barriers, make shared arrays non-deterministic \
                             but consistent between threads")

    advanced.add_argument("--array-equalities",   action='store_true',
                          help="Generate equality candidate invariants \
                          for array variables")
    advanced.add_argument("--asymmetric-asserts", action='store_true',
                          help="Emit assertions only for the first thread. \
                          Sound, and may lead to faster verification, \
                          but can yield false positives")

    advanced.add_argument("--boogie-file=", type=argparse.FileType('r'), action='append',
                          help="Specify a supporting .bpl file to be used \
                          during verification", metavar="X.bpl")
    advanced.add_argument("--bugle-lang=", dest='source_language', choices=["cl","cu"],
                          type=lambda x: SourceLanguage.OpenCL if x == "cl" else SourceLanguage.CUDA,
                          help="Bitcode language if passing in a bitcode file")

    advanced.add_argument("--call-site-analysis",           action='store_true',
                          help="Turn on call site analysis")
    advanced.add_argument("--math-int",                     action='store_true',
                          help="Represent integer types using mathematical integers \
                          instead of bit-vectors")
    advanced.add_argument("--inverted-tracking",            action='store_true',
                          help="Use do_not_track instead of track. \
                          This may result in faster detection of races for some solvers")
    advanced.add_argument("--no-annotations",               action='store_true',
                          help="Ignore all source-level annotations")
    advanced.add_argument("--only-requires",                action='store_true',
                          help="Ignore all source-level annotations except for requires")
    advanced.add_argument("--invariants-as-candidates",     action='store_true',
                          help="Interpret all source-level invariants as candidates")
    advanced.add_argument("--no-barrier-access-checks",     action='store_true',
                          help="Turn off access checks for barrier invariants")
    advanced.add_argument("--no-constant-write-checks",     action='store_true',
                          help="Turn off access checks for writes to constant space")
    advanced.add_argument("--no-inline",                    action='store_true',
                          help="Turn off automatic inlining by Bugle")
    advanced.add_argument("--no-loop-predicate-invariants", action='store_true',
                          help="Turn off automatic generation of loop invariants \
                          related to predicates, which can be incorrect")
    advanced.add_argument("--no-smart-predication",         action='store_true',
                          help="Turn off smart predication")
    advanced.add_argument("--no-uniformity-analysis",       action='store_true',
                          help="Turn off uniformity analysis")
    advanced.add_argument("--no-refined-atomics",           action='store_true',
                          help="Disable return-value abstraction refinement for atomics")
    advanced.add_argument("--only-log",                     action='store_true',
                          help="Log accesses to arrays, but do not check for races. \
                          This can be useful for determining access pattern invariants")

    advanced.add_argument("--kernel-args=", type=params,
                          help="For kernel K with scalar parameters (x1,...,xn), \
                          adds the precondition (x1==v1 && ... && xn=vn). \
                          Use * to denote an unconstrained parameter",
                          metavar="K,v1,...,vn")

    advanced.add_argument("--warp-sync=", type=non_negative,
                          help="Synchronize threads within warps of size X. \
                          Defaults to 'resync' analysis method, \
                          unless one of the following two options are set",
                          metavar="X")
    twopass = advanced.add_mutually_exclusive_group()
    twopass.add_argument("--no-warp",   action='store_true',
                         help="Only check inter-warp races")
    twopass.add_argument("--only-warp", action='store_true',
                         help="Only check intra-warp races")

    advanced.add_argument("--race-instrumenter=",
                          choices=["standard","watchdog-single","watchdog-multiple"],
                          default="standard",
                          help="Choose which method of race instrumentation to use")
    advanced.add_argument("--solver=", choices=["z3","cvc4"], default="z3",
                          help="Choose which SMT theorem prover to use in the backend. \
                          Default is z3")
    advanced.add_argument("--logic=", choices=["ALL_SUPPORTED","QF_ALL_SUPPORTED"],
                          default="QF_ALL_SUPPORTED",
                          help="Define the logic for the CVC4 SMT solver backend. \
                          Default is QF_ALL_SUPPORTED")

    development = parser.add_argument_group("DEVELOPMENT OPTIONS")
    development.add_argument("--debug",         action='store_true',
                             help="Enable debugging of GPUVerify components: \
                             exceptions will not be suppressed")
    development.add_argument("--keep-temps",    action='store_true',
                             help="Keep intermediate bc, gbpl, and cbpl files")
    development.add_argument("--gen-smt2",      action='store_true',
                             help="Generate smt2 file")

    development.add_argument("--clang-opt=",    dest='clang_options',
                             action='append',
                             help="Specify option to be passed to Clang")
    development.add_argument("--opt-opt=",      dest='opt_options',
                             action='append',
                             help="Specify option to be passed to optimization pass")
    development.add_argument("--bugle-opt",     dest='bugle_options',
                             action='append',
                             help="Specify option to be passed to Bugle")
    development.add_argument("--vcgen-opt=",    dest='vcgen_options',
                             action='append',
                             help="Specify option to be passed to generation")
    development.add_argument("--cruncher-opt=", dest='cruncher_options',
                             action='append',
                             help="Specify option to be passed to cruncher")
    development.add_argument("--boogie-opt=",   dest='boogie_options',
                             action='append',
                             help="Specify option to be passed to Boogie")

    development.add_argument("--stop-at-opt",  dest='stop',
                             action='store_const', const="opt",
                             help="Stop after LLVM optimization pass")
    development.add_argument("--stop-at-gbpl", dest='stop',
                             action='store_const', const="bugle",
                             help="Stop after generating gbpl")
    development.add_argument("--stop-at-bpl",  dest='stop',
                             action='store_const', const="vcgen",
                             help="Stop after generating bpl")
    development.add_argument("--stop-at-cbpl", dest='stop',
                             action='store_const', const="cruncher",
                             help="Stop after generating an annotated bpl")

    inference = parser.add_argument_group("INVARIANT INFERENCE OPTIONS")
    inference.add_argument("--no-infer", dest='inference', action='store_false',
                           help="Turn off invariant inference")
    inference.add_argument("--omit-infer=", action='append',
                           help="Do not generate invariants tagged 'X'")
    inference.add_argument("--staged-inference",   action='store_true',
                           help="Perform invariant inference in stages; \
                           this can boost performance for complex kernels \
                           (but this is not guaranteed)")
    inference.add_argument("--infer-info",         action='store_true',
                           help="Prints information about the process")
    inference.add_argument("--k-induction-depth=", type=non_negative,
                           default=-1,
                           help="Applies k-induction with k=X to all loops",
                           metavar="X")

    undocumented = parser.add_argument_group("UNDOCUMENTED")
    undocumented.add_argument("--debug-houdini", action='store_true')

    interceptor = parser.add_argument_group("BATCH MODE")
    interceptor.add_argument("--show-intercepted", action='store_true')
    interceptor.add_argument("--check-intercepted=", type=non_negative,
                             action='append', metavar="X")
    interceptor.add_argument("--check-all-intercepted", action='store_true')
    interceptor.add_argument("--cache")


    def to_ldict (parsed):
      return ldict((k[:-1],v) if k.endswith('=') else (k,v) for k,v in vars(parsed).items())

    args = to_ldict(parser.parse_args(argv))

    if not args.batch_mode:
      if not args['kernel']:
        parser.error("Must provide a kernel for normal mode")
      name = args['kernel'].name

      # Try reading the first line of the kernel file as arguments
      header = args.kernel.readline()
      if header.startswith("//"):
        try:
          p = parser.parse_args(strip_dudspace(header[len("//"):].rstrip().split(" ")))
          file_args = to_ldict(p)
          # Then override anything set via the command line
          update_args(file_args,args)
          args = file_args
        except Exception as e:
          pass # Probably doesn't parse -- worth a try

      _unused,ext = SplitFilenameExt(name)

      starts = defaultdict(lambda : "clang",
                           { '.bc': "opt", '.opt.bc': "bugle", '.gbpl': "vcgen",
                             '.bpl': "cruncher", '.cbpl': "boogie" })
      args['start'] = starts[ext]

      if not args['stop']:
        args['stop'] = "boogie"

      if not args['source_language'] and args['start'] in ['clang','opt','bugle']:
        if ext == '.cl':
          args['source_language'] = SourceLanguage.OpenCL
        elif ext == '.cu':
          args['source_language'] = SourceLanguage.CUDA
        else:
          if ext in ['.bc','.opt.bc']:
            proc = subprocess.Popen([gvfindtools.llvmBinDir + "/llvm-nm", name],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
      if args.num_groups and args.global_size:
        parser.error("--num_groups and --global_size are mutually exclusive.")
      elif args.group_size and args.global_size:
        if len(args.group_size) != len(args.global_size):
          parser.error("Dimensions of --local_size and --global_size must match.")
        args['num_groups'] = [(a//b) for (a,b) in zip(args.global_size,args.group_size)]
        # The below returns a sequence of dimensions that aren't integer multiples
        for i in (i for (i, a,b,c) in zip(list(range(len(args.num_groups))),
                                          args.num_groups,args.group_size,args.global_size) if a * b != c):
          parser.error("Dimension " + str(i) +
                       " of global_size does not divide by the same dimension in local_size")
      elif args.group_size and args.num_groups:
        pass
      else:
        if args.source_language == SourceLanguage.OpenCL:
          parser.error("Must specify thread dimensions with --local_size and --global_size")
        else:
          parser.error("Must specify thread dimensions with --blockDim and --gridDim")

      if args.verbose:
        print("Got {} groups of size {}".format("x".join(map(str,args['num_groups'])),
                                                "x".join(map(str,args['group_size']))))

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
  values = [x if x == '*' else int(x) for x in string.split(",")]
  if (len(values) not in [1,2,3]) or len([x for x in values if x > 0]) < len(values):
    raise argparse.ArgumentTypeError("Dimensions must be a vector of 1-3 positive integers")
  return values

def params (string):
  string = string.strip()
  string = string[1:-1] if string[0]+string[-1] == "[]" else string
  values = string.split(",")
  if not all(x == '*' or x.startswith("0x") for x in values[1:]):
    raise argparse.ArgumentTypeError("kernel args are hex values or *")
  values = values[:1] + [x if x == '*' else x[len("0x"):] for x in values[1:]]
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

  # Must be added after include of opencl or cuda header
  if args['no_annotations'] or args['only_requires']:
    noAnnotationsHeader = [ "-include", "annotations/no_annotations.h" ]
    clangOpenCLOptions.extend(noAnnotationsHeader)
    clangCUDAOptions.extend(noAnnotationsHeader)
    if args['only_requires']:
      clangOpenCLDefines.append("ONLY_REQUIRES")
      clangCUDADefines.append("ONLY_REQUIRES")
  if args['invariants_as_candidates']:
    candidateAnnotationsHeader = [ "-include", "annotations/candidate_annotations.h" ]
    clangOpenCLOptions.extend(candidateAnnotationsHeader)
    clangCUDAOptions.extend(candidateAnnotationsHeader)

  CommandLineOptions.clangOptions += sum([a.split(" ") for a in args['clang_options'] or []],[])
  CommandLineOptions.optOptions += sum([a.split(" ") for a in args['opt_options'] or []],[])
  CommandLineOptions.bugleOptions += sum([a.split(" ") for a in args['bugle_options'] or []],[])
  CommandLineOptions.vcgenOptions += sum([a.split(" ") for a in args['vcgen_options'] or []],[])
  CommandLineOptions.cruncherOptions += sum([a.split(" ") for a in args['cruncher_options'] or []],[])
  CommandLineOptions.boogieOptions += sum([a.split(" ") for a in args['boogie_options'] or []],[])
  
  CommandLineOptions.vcgenOptions += ["/noCandidate:"+a for a in args['omit_infer'] or []]
  if args.kernel_args:
    CommandLineOptions.vcgenOptions += [ "/kernelArgs:" + ','.join(map(str,args['kernel_args'])) ]
    CommandLineOptions.cruncherOptions += [ "/proc:$" + args.kernel_args[0] ]
    CommandLineOptions.boogieOptions   += [ "/proc:$" + args.kernel_args[0] ]

  CommandLineOptions.cruncherOptions += [x.name for x in args['boogie_file'] or []] or []
  
  CommandLineOptions.boogieOptions += [ "/blockHighestDim:" + str(len(args.group_size) - 1) ]
  CommandLineOptions.cruncherOptions += [ "/blockHighestDim:" + str(len(args.group_size) - 1) ]
  CommandLineOptions.boogieOptions += [ "/gridHighestDim:" + str(len(args.num_groups) - 1) ]
  CommandLineOptions.cruncherOptions += [ "/gridHighestDim:" + str(len(args.num_groups) - 1) ]
  
  if args['source_language'] == SourceLanguage.CUDA:
    CommandLineOptions.boogieOptions += [ "/sourceLanguage:cu" ]
    CommandLineOptions.cruncherOptions += [ "/sourceLanguage:cu" ]
  
  return CommandLineOptions

class GPUVerifyInstance (object):

  def __init__ (self, args, out=None):
    """
    This function should NOT be called directly instead call main()
    It is assumed that argv has had sys.argv[0] removed
    """

    self.out = out

    if os.name == 'posix':
      if args.debug:
        self.mono = [ 'mono' , '--debug' ]
      else:
        self.mono = [ 'mono' ]
    else:
      self.mono = [] # Presumably using Windows so don't need mono

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

      assert(args.size_t in [32,64])
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
      assert args.bugle_lang in [ "cl", "cu" ]
      CommandLineOptions.bugleOptions += [ "-l", args.bugle_lang, "-s", locFilename, "-o", gbplFilename, optFilename ]

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
  
    if args.infer_info:
      CommandLineOptions.cruncherOptions += [ "/trace" ]
    if args.debug_houdini:
      CommandLineOptions.cruncherOptions += [ "/debugConcurrentHoudini" ]
  
    if args.solver == "cvc4":      
      CommandLineOptions.cruncherOptions += [ "/proverOpt:SOLVER=cvc4" ]
      CommandLineOptions.cruncherOptions += [ "/cvc4exe:" + gvfindtools.cvc4BinDir + os.sep + "cvc4.exe" ]
      CommandLineOptions.cruncherOptions += [ "/proverOpt:LOGIC=" + args.logic ]
      CommandLineOptions.boogieOptions += [ "/proverOpt:SOLVER=cvc4" ]
      CommandLineOptions.boogieOptions += [ "/cvc4exe:" + gvfindtools.cvc4BinDir + os.sep + "cvc4.exe" ]
      CommandLineOptions.boogieOptions += [ "/proverOpt:LOGIC=" + args.logic ]
    else:
      CommandLineOptions.cruncherOptions += [ "/z3exe:" + gvfindtools.z3BinDir + os.sep + "z3.exe" ]
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
    self.debug = args.debug
    self.timeout = args.timeout

  def run(self, command):
    """ Run a command with an optional timeout. A timeout of zero
        implies no timeout.
    """
    popenargs={}
    if self.verbose:
      print(" ".join(command))
    else:
      popenargs['bufsize']=0

    # We don't want messages to go to stdout
    stdoutFile = tempfile.SpooledTemporaryFile()
    popenargs["stdout"] = self.out #stdoutFile

    # Redirect stderr to whatever stdout is redirected to
    if __name__ != '__main__':
      popenargs['stderr']=subprocess.PIPE


    # Redirect stdin, othewise terminal text becomes unreadable after timeout
    popenargs['stdin']=subprocess.PIPE

    proc = psutil.Popen(command,**popenargs)
    if args.timeout > 0:
      try:
        return_code = proc.wait(timeout=self.timeout)
      except psutil.TimeoutExpired:
        children = proc.get_children(True)
        proc.terminate()
        for child in children:
          try:
            child.terminate()
          except psutil.NoSuchProcess:
            pass
        raise
    else:
      return_code = proc.wait()

    stdout, stderr = proc.communicate()
    stdoutFile.seek(0)
    stdout = stdoutFile.read()
    stdoutFile.close()
    # We do not return stderr, as it was redirected to stdout
    return stdout, return_code

  def RunTool(self,ToolName, Command, ErrorCode):
    """ Returns a triple (succeeded, timedout, stdout), and stores the return code """
    assert ToolName in Tools
    if self.verbose:
      print("Running " + ToolName)
    try:
      start = timeit.default_timer()
      stdout, returnCode = self.run(Command)
      stdout = stdout.decode() # For python3, we get bytes not a str, so convert
      end = timeit.default_timer()
    except psutil.TimeoutExpired:
      self.timing[ToolName] = self.timeout
      return False, True, "{} timed out. Use --timeout=N with N > {} to increase timeout, or --timeout=0 to disable timeout.\n".format(ToolName, self.timeout)
    except (OSError,WindowsError) as e:
      print("Error while invoking {} : {}".format(ToolName, str(e)))
      print("With command line args:")
      print(pprint.pformat(Command))
      raise
    self.timing[ToolName] = end-start
    # if returnCode != ErrorCodes.SUCCESS:
    #   if self.silent and stdout: print(stdout, file=sys.stderr)
    return (returnCode == ErrorCodes.SUCCESS, False, stdout)

  def invoke (self):
    """ Returns (returncode, outstring) """

    timeout = False
    success = True
    stdout = ""

    if not self.skip["clang"]:
      success, timeout, stdout = self.RunTool("clang",
              [gvfindtools.llvmBinDir + "/clang"] +
              self.clangOptions +
              [("-I" + str(o)) for o in self.includes] +
              [("-D" + str(o)) for o in self.defines],
              ErrorCodes.CLANG_ERROR)

    if timeout: return ErrorCodes.TIMEOUT, stdout
    if not success: return ErrorCodes.CLANG_ERROR, stdout

    if not self.skip["opt"]:
      success, timeout, stdout = self.RunTool("opt",
              [gvfindtools.llvmBinDir + "/opt"] +
              self.optOptions,
              ErrorCodes.OPT_ERROR)

    if timeout: return ErrorCodes.TIMEOUT, stdout
    if not success: return ErrorCodes.OPT_ERROR, stdout

    if self.stop == 'opt': return 0, stdout

    if not self.skip["bugle"]:
      success, timeout, stdout = self.RunTool("bugle",
              [gvfindtools.bugleBinDir + "/bugle"] +
              self.bugleOptions,
              ErrorCodes.BUGLE_ERROR)

    if timeout: return ErrorCodes.TIMEOUT, stdout
    if not success: return ErrorCodes.BUGLE_ERROR, stdout

    if self.stop == 'bugle': return 0, stdout

    if not self.skip["vcgen"]:
      success, timeout, stdout = self.RunTool("gpuverifyvcgen",
              self.mono +
              [gvfindtools.gpuVerifyBinDir + "/GPUVerifyVCGen.exe"] +
              self.vcgenOptions,
              ErrorCodes.GPUVERIFYVCGEN_ERROR)

    if timeout: return ErrorCodes.TIMEOUT, stdout
    if not success: return ErrorCodes.GPUVERIFYVCGEN_ERROR, stdout

    if self.stop == 'vcgen': return 0, stdout

    if not self.skip["cruncher"]:
      success, timeout, stdout = self.RunTool("gpuverifycruncher",
                self.mono +
                [gvfindtools.gpuVerifyBinDir + os.sep + "GPUVerifyCruncher.exe"] +
                self.cruncherOptions,
                ErrorCodes.BOOGIE_ERROR)
    if timeout: return ErrorCodes.TIMEOUT, stdout
    if not success: return ErrorCodes.BOOGIE_ERROR, stdout

    if self.stop == 'cruncher': return 0, stdout

    success, timeout, stdout = self.RunTool("gpuverifyboogiedriver",
            self.mono +
            [gvfindtools.gpuVerifyBinDir + "/GPUVerifyBoogieDriver.exe"] +
            self.boogieOptions,
            ErrorCodes.BOOGIE_ERROR)
    if timeout: return ErrorCodes.TIMEOUT, stdout
    if not success: return ErrorCodes.BOOGIE_ERROR, stdout

    if self.silent:
      return 0, ""

    string_builder = io.StringIO()

    if self.mode == AnalysisMode.FINDBUGS:
      print("No defects were found while analysing: " + ", ".join(self.sourceFiles), file=string_builder)
      print("Notes:", file=string_builder)
      print("- use --loop-unwind=N with N > " + str(self.loopUnwindDepth) + " to search for deeper bugs", file=string_builder)
      print("- re-run in verification mode to try to prove absence of defects", file=string_builder)
    else:
      print("Verified: " + ", ".join(self.sourceFiles), file=string_builder)
      if not self.onlyDivergence:
        print("- no data races within " + ("work groups" if self.SL == SourceLanguage.OpenCL else "thread blocks"), file=string_builder)
        if not self.onlyIntraGroup:
          print("- no data races between " + ("work groups" if self.SL == SourceLanguage.OpenCL else "thread blocks"), file=string_builder)
      print("- no barrier divergence", file=string_builder)
      print("- no assertion failures", file=string_builder)
      print("(but absolutely no warranty provided)", file=string_builder)

    return 0, string_builder.getvalue()

  def showTiming(self, exitCode):
    """ Returns the timing as a string """
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
      return ','.join(row)
    else:
      total = sum(self.timing.values())
      print("Timing information (%.2f secs):" % total, file=self.out)
      if self.timing:
        padTool = max([ len(tool) for tool in list(self.timing.keys()) ])
        padTime = max([ len('%.3f secs' % t) for t in list(self.timing.values()) ])
        string_builder = io.StringIO()
        for tool in Tools:
          if tool in self.timing:
            print("- %s : %s" % (tool.ljust(padTool), ('%.3f secs' % self.timing[tool]).rjust(padTime)), file=string_builder)
        return string_builder.getvalue()
      else:
        return "- no tools ran"


def update_args (base, new):
  for k,v in new.items():
    if not base[k] and v:
      if base.verbose or new.verbose:
        print("Setting {} to {}".format(k,v))
      base[k] = v
    elif v and base[k] != v:
      if base.verbose or new.verbose:
        print("Supplanting {}={} for {}".format(k,base[k],v))
      base[k] = v

def strip_dudspace (argv):
  return [x for x in argv if x != ""]

def parse_header (file):
  code = [x.rstrip() for x in file.readlines()]
  header_args = strip_dudspace(code[0][len("//"):].split(" "))
  return code[1:],header_args

# cache is of form map(code, list(known_safe))
def in_cache (f,args,success_cache,failure_cache):
  with open(f) as file:
    code = file.readlines()
  while code[0].startswith("//"):
    code = code[1:]
  code = ''.join(code)
  # If any match all arguments (or *)
  if code in success_cache and any(all(v in ['*',args[k]] for k,v in trial.items()) for trial in success_cache[code]):
    return True
  # If any match all arguments
  elif code in failure_cache and any(all(v == args[k] for k,v in trial.items()) for trial in failure_cache[code]):
    return False
  return None

def add_to_cache (f,args,cache):
  with open(f) as file:
    code = file.readlines()
    while code[0].startswith("//"):
      code = code[1:]
    code = ''.join(code)
    if code not in cache:
      cache[code] = []
    cache[code].append(dict((k,v) for k,v in args.items() if k in ['group_size','num_groups','kernel_args']))
    if args.verbose:
      print("added to cache")

# TODO: Alter in_cache and add_to_cache
def verify_batch (files, success_cache={}):
  failure_cache = {}
  success = []
  failure = []
  for i,f in enumerate(files):
    x = parse_args([f])
    rc = in_cache(f,x,success_cache,failure_cache)
    # Only check if we've never seen it before
    if rc is None:
      rc = main(x,open(os.devnull,'w')) == ErrorCodes.SUCCESS
      add_to_cache(f,x,success_cache if rc else failure_cache)
    if rc:
      success.append((f,x,i))
    else:
      failure.append((f,x,i))

  print("GPUVerify kernel analyer checked {} kernels.".format(len(success) + len(failure)))
  print("Successfully verified {} kernels.".format(len(success)))
  print("Failed to verify {} kernels.".format(len(failure)))

  print("")
  print("Successes:")
  for s,args,i in success:
    print("[{}]: Verification of {} ({}) succeeded with: local_size={} global_size={} args={}"
          .format(i,
                  args['kernel_args'][0],
                  s,
                  ",".join(map(str,args['group_size'])),
                  ",".join(map(str,args['global_size'])),
                  ",".join(map(str,args['kernel_args'][1:]))
                )
    )

  print("")
  print("Failures:")
  for f,args,i in failure:
    print("[{}]: Verification of {} ({}) failed with: local_size={} global_size={} args={}"
          .format(i,
                  args['kernel_args'][0],
                  f,
                  ",".join(map(str,args['group_size'])),
                  ",".join(map(str,args['global_size'])),
                  ",".join(map(str,args['kernel_args'][1:]))
                )
    )

def do_batch_mode (host_args): 
  kernels = []
  for path, subdirs, files in os.walk(".gpuverify"):
    kernels += [path+os.sep+x for x in files]
  kernels = sorted(kernels)

  if host_args.show_intercepted:
    for index,file_name in enumerate(kernels):
      with open(file_name) as file:
        code,header_args, = parse_header(file)
        built = code[0][len("//"):]
        ran = code[1][len("//"):]
        print("["+str(index)+"] " + file_name+": " + ' '.join(header_args))
        print(built)
        print(ran)

  try:
    success_cache = pickle.load(open(args.cache))
  except Exception:
    success_cache = {}

  if host_args.check_intercepted:
    verify_batch([kernels[i] for i in host_args.check_intercepted], success_cache)
  elif host_args.check_all_intercepted:
    verify_batch(kernels, success_cache)

  if args.cache:
    pickle.dump(success_cache,open(args.cache,"w"))

def main(argv, out=sys.stdout):
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
  gv_instance = GPUVerifyInstance(argv, out)
  def handleTiming (exitCode):
    if gv_instance.time:
      print(gv_instance.showTiming(exitCode))
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
    returnCode = gv_instance.invoke()
  except GPUVerifyException as e:
    doCleanUp(timing=True, exitCode=e.code)
    raise
  except Exception:
    # Something went very wrong
    doCleanUp(timing=False, exitCode=0) # It doesn't matter what the exitCode is
    raise

  doCleanUp(timing=True) # Do this outside try block so we don't call twice!
  return returnCode

debug = False

if __name__ == '__main__':
  """
  Entry point for GPUVerify as a script
  """

  # These are the exception error codes that won't be printed if they are thrown
  ignoredErrors = [ ErrorCodes.SUCCESS, ErrorCodes.BOOGIE_ERROR ]

  try:
    args = parse_args(sys.argv[1:] or [ '--help' ])
    debug = args.debug
    if args.batch_mode:
      do_batch_mode(args)
    else:
      rc, out = main(args)
      sys.stdout.write(out)
      sys.exit(rc)
  except ConfigurationError as e:
    print(str(e), file=sys.stderr)
    sys.exit(ErrorCodes.CONFIGURATION_ERROR)
  except ArgumentParserError as e:
    print(str(e), file=sys.stderr)
    sys.exit(ErrorCodes.COMMAND_LINE_ERROR)
  except KeyboardInterrupt:
    sys.exit(ErrorCodes.CTRL_C)
  except GPUVerifyException as e:
    # We assume that globals are not cleaned up when running as a script so it 
    # is safe to read CommandLineOptions
    if (not (e.code in ignoredErrors)) or debug:
      if e.code == ErrorCodes.COMMAND_LINE_ERROR:
        # For command line errors only show the message and not internal details
        print('GPUVerify: {0}'.format(e.msg), file=sys.stderr)
      else:
        # Show all exception info for everything else not ignored
        print(str(e), file=sys.stderr)
    sys.exit(e.code)

  sys.exit(ErrorCodes.SUCCESS)
