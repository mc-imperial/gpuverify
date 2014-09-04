#!/usr/bin/env python
# vim: set shiftwidth=2 tabstop=2 expandtab softtabstop=2:
from __future__ import print_function

import pickle
import os
import subprocess
import sys
import timeit
import pprint
import tempfile
from collections import namedtuple
import copy
import distutils.spawn

if sys.version_info.major == 3:
  import io
else:
  # In python2.7 importing io.StringIO() doesn't work
  # very well because it expects unicode strings
  # use StringIO instead
  import StringIO as io

# To properly kill child processes cross platform
try:
  import psutil
except ImportError:
  sys.stderr.write("GPUVerify requires Python to be equipped with the psutil module.\n")
  sys.stderr.write("On Windows we recommend installing psutil from a prebuilt binary:\n")
  sys.stderr.write("  https://pypi.python.org/pypi?:action=display&name=psutil#downloads\n")
  sys.stderr.write("On Linux/OSX, we recommend installing psutil with pip:\n")
  sys.stderr.write("  pip install psutil\n")
  sys.exit(1)

from GPUVerifyScript.argument_parser import ArgumentParserError, parse_arguments
from GPUVerifyScript.constants import AnalysisMode, SourceLanguage
from GPUVerifyScript.error_codes import ErrorCodes
from GPUVerifyScript.json_loader import JSONError, json_load
import getversion

class ConfigurationError(Exception):
  def __init__ (self, msg):
    self.msg = msg
  def __str__ (self):
    return "GPUVerify: CONFIGURATION_ERROR error ({}): {}".format(ErrorCodes.CONFIGURATION_ERROR,self.msg)

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

if gvfindtools.useMono:
  # Check mono in path
  if distutils.spawn.find_executable('mono') == None:
    raise ConfigurationError("Could not find the mono executable in your PATH")


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
                       "cl_khr_fp16",
                       "cl_clang_storage_class_specifiers",
                       "__OPENCL_VERSION__=120"
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
    # Make sure we make a copy so we don't change the global list
    self.includes = list(clangCoreIncludes)
    self.defines = list(clangCoreDefines)
    self.clangOptions = list(clangCoreOptions)
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

def ShowVersion():
    """ This will check if using gpuverify from development directory.
        If so this will invoke Mercurial to find out version information.
        If this is a deployed version we will try to read the version from
        a file instead
    """
    print(getversion.getVersionString())
    sys.exit()

def parse_args(argv):
    args = parse_arguments(argv, gvfindtools.defaultSolver, gvfindtools.llvmBinDir)

    if args.version:
      ShowVersion()

    if not args.batch_mode and args.verbose and args.num_groups and args.group_size:
      print("Got {} groups of size {}".format("x".join(map(str,args.num_groups)),
                                              "x".join(map(str,args.group_size))))

    return args

def processOptions(args):
  CommandLineOptions = copy.deepcopy(DefaultCmdLineOptions())
  ext = args.kernel_ext
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

  CommandLineOptions.defines += args.defines
  CommandLineOptions.includes += args.includes

  CommandLineOptions.clangOptions += sum([a.split() for a in args.clang_options],[])
  CommandLineOptions.optOptions += sum([a.split() for a in args.opt_options],[])
  CommandLineOptions.bugleOptions += sum([a.split() for a in args.bugle_options],[])
  CommandLineOptions.vcgenOptions += sum([a.split() for a in args.vcgen_options],[])
  CommandLineOptions.cruncherOptions += sum([a.split() for a in args.cruncher_options],[])
  CommandLineOptions.boogieOptions += sum([a.split() for a in args.boogie_options],[])

  CommandLineOptions.vcgenOptions += ["/noCandidate:" + a for a in args.omit_infer or []]
  for ka in args.kernel_args:
    CommandLineOptions.vcgenOptions += [ "/kernelArgs:" + ','.join(map(str, ka)) ]
    CommandLineOptions.bugleOptions += [ "-k", ka[0] ]

  for ka in args.kernel_arrays:
    CommandLineOptions.bugleOptions += [ "-kernel-array-sizes=" + ','.join(map(str, ka)) ]
    CommandLineOptions.bugleOptions += [ "-k", ka[0] ]

  if len(args.kernel_args) > 0 or len(args.kernel_arrays) > 0:
    CommandLineOptions.bugleOptions += [ "-only-explicit-entry-points" ]

  CommandLineOptions.cruncherOptions += [x.name for x in args.boogie_file]

  if args.group_size:
    CommandLineOptions.boogieOptions += [ "/blockHighestDim:" + str(len(args.group_size) - 1) ]
    CommandLineOptions.cruncherOptions += [ "/blockHighestDim:" + str(len(args.group_size) - 1) ]
  if args.num_groups:
    CommandLineOptions.boogieOptions += [ "/gridHighestDim:" + str(len(args.num_groups) - 1) ]
    CommandLineOptions.cruncherOptions += [ "/gridHighestDim:" + str(len(args.num_groups) - 1) ]

  if args.source_language == SourceLanguage.CUDA:
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

    if gvfindtools.useMono:
      if args.debug:
        self.mono = [ 'mono' , '--debug' ]
      else:
        self.mono = [ 'mono' ]
    else:
      self.mono = []

    self.timing = {}

    CommandLineOptions = processOptions(args)

    self.stop = args.stop

    cleanUpHandler.setVerbose(args.verbose)

    filename = args.kernel_name

    CommandLineOptions.defines += [ '__BUGLE_' + str(args.size_t) + '__' ]
    if (args.size_t == 32):
      CommandLineOptions.clangOptions += [ "-target", "nvptx--" ]
    elif (args.size_t == 64):
      CommandLineOptions.clangOptions += [ "-target", "nvptx64--" ]

    if args.source_language == SourceLanguage.OpenCL and not CommandLineOptions.skip["clang"]:
      CommandLineOptions.clangOptions += clangOpenCLOptions
      CommandLineOptions.clangOptions += clangInlineOptions
      CommandLineOptions.includes += clangOpenCLIncludes
      CommandLineOptions.defines += clangOpenCLDefines

      # Must be added after include of opencl header
      if args.no_annotations or args.only_requires:
        CommandLineOptions.clangOptions += [ "-include", "annotations/no_annotations.h" ]
        if args.only_requires:
          CommandLineOptions.defines.append("ONLY_REQUIRES")
      if args.invariants_as_candidates:
        CommandLineOptions.clangOptions += [ "-include", "annotations/candidate_annotations.h" ]

      CommandLineOptions.defines.append("__" + str(len(args.group_size)) + "D_WORK_GROUP")
      CommandLineOptions.defines.append("__" + str(len(args.num_groups)) + "D_GRID")
      for (index, value) in enumerate(args.group_size):
        if value == '*':
          CommandLineOptions.defines.append("__LOCAL_SIZE_" + str(index) + "_FREE")
        else:
          CommandLineOptions.defines.append("__LOCAL_SIZE_" + str(index) + "=" + str(value))
      for (index, value) in enumerate(args.num_groups):
        if value == '*':
          CommandLineOptions.defines.append("__NUM_GROUPS_" + str(index) + "_FREE")
        elif type(value) is tuple:
          CommandLineOptions.defines.append("__NUM_GROUPS_" + str(index) + "_FREE")
          CommandLineOptions.defines.append("__GLOBAL_SIZE_" + str(index) + "=" + str(value[1]))
        else:
          CommandLineOptions.defines.append("__NUM_GROUPS_" + str(index) + "=" + str(value))

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

      # Must be added after include of cuda header
      if args.no_annotations or args.only_requires:
        CommandLineOptions.clangOptions += [ "-include", "annotations/no_annotations.h" ]
        if args.only_requires:
          CommandLineOptions.defines.append("ONLY_REQUIRES")
      if args.invariants_as_candidates:
        CommandLineOptions.clangOptions += [ "-include", "annotations/candidate_annotations.h" ]

      CommandLineOptions.defines.append("__" + str(len(args.group_size)) + "D_THREAD_BLOCK")
      CommandLineOptions.defines.append("__" + str(len(args.num_groups)) + "D_GRID")
      for (index, value) in enumerate(args.group_size):
        if value == '*':
          CommandLineOptions.defines.append("__BLOCK_DIM_" + str(index) + "_FREE")
        else:
          CommandLineOptions.defines.append("__BLOCK_DIM_" + str(index) + "=" + str(value))
      for (index, value) in enumerate(args.num_groups):
        if value == '*':
          CommandLineOptions.defines.append("__GRID_DIM_" + str(index) + "_FREE")
        else:
          CommandLineOptions.defines.append("__GRID_DIM_" + str(index) + "=" + str(value))

    # Intermediate filenames
    bcFilename = filename + '.bc'
    optFilename = filename + '.opt.bc'
    gbplFilename = filename + '.gbpl'
    cbplFilename = filename + '.cbpl'
    bplFilename = filename + '.bpl'
    locFilename = filename + '.loc'
    smt2Filename = filename + '.smt2'
    if not args.keep_temps:
      def DeleteFile(filename):
        """ Delete the filename if it exists; but don't delete the original input """
        if filename == args.kernel.name: return
        try: os.remove(filename)
        except OSError: pass
      cleanUpHandler.register(DeleteFile, bcFilename)
      if not args.stop == 'opt': cleanUpHandler.register(DeleteFile, optFilename)
      if not args.stop == 'bugle': cleanUpHandler.register(DeleteFile, gbplFilename)
      if not args.stop == 'bugle': cleanUpHandler.register(DeleteFile, locFilename)
      if not args.stop == 'cruncher': cleanUpHandler.register(DeleteFile, cbplFilename)
      if not args.stop == 'vcgen': cleanUpHandler.register(DeleteFile, bplFilename)

    CommandLineOptions.clangOptions += ["-o", bcFilename]
    CommandLineOptions.clangOptions += ["-x", "cl" if args.source_language == SourceLanguage.OpenCL else "cuda", args.kernel.name]

    CommandLineOptions.optOptions += [ "-o", optFilename, bcFilename ]

    if not CommandLineOptions.skip['bugle']:
      CommandLineOptions.bugleOptions += [ "-l", "cl" if args.source_language == SourceLanguage.OpenCL else "cu", "-s", locFilename, "-o", gbplFilename, optFilename ]

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
    if args.adversarial_abstraction:
      CommandLineOptions.vcgenOptions += [ "/adversarialAbstraction" ]
    if args.equality_abstraction:
      CommandLineOptions.vcgenOptions += [ "/equalityAbstraction" ]
    if args.no_benign_tolerance:
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
    if args.asymmetric_asserts:
      CommandLineOptions.vcgenOptions += [ "/asymmetricAsserts" ]
    if args.staged_inference:
      CommandLineOptions.vcgenOptions += [ "/stagedInference" ]
      CommandLineOptions.cruncherOptions += [ "/stagedInference" ]

    CommandLineOptions.vcgenOptions += [ "/print:" + filename, gbplFilename ] #< ignore .bpl suffix

    if args.mode == AnalysisMode.FINDBUGS:
      CommandLineOptions.boogieOptions += [ "/loopUnroll:" + str(args.loop_unwind) ]

    if args.k_induction_depth > 0:
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

    if args.solver == "cvc4":
      CommandLineOptions.cruncherOptions += [ "/proverOpt:SOLVER=cvc4" ]
      CommandLineOptions.cruncherOptions += [ "/cvc4exe:" + gvfindtools.cvc4BinDir + os.sep + "cvc4.exe" ]
      CommandLineOptions.cruncherOptions += [ "/proverOpt:LOGIC=QF_ALL_SUPPORTED" ]
      CommandLineOptions.boogieOptions += [ "/proverOpt:SOLVER=cvc4" ]
      CommandLineOptions.boogieOptions += [ "/cvc4exe:" + gvfindtools.cvc4BinDir + os.sep + "cvc4.exe" ]
      CommandLineOptions.boogieOptions += [ "/proverOpt:LOGIC=QF_ALL_SUPPORTED" ]
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

    if args.race_instrumenter == "original":
      CommandLineOptions.bugleOptions += [ "-race-instrumentation=original" ]
      CommandLineOptions.vcgenOptions += [ "/raceChecking:ORIGINAL" ]
      CommandLineOptions.cruncherOptions += [ "/raceChecking:ORIGINAL" ]
      CommandLineOptions.boogieOptions += [ "/raceChecking:ORIGINAL" ]
    if args.race_instrumenter == "watchdog-single":
      CommandLineOptions.bugleOptions += [ "-race-instrumentation=watchdog-single" ]
      CommandLineOptions.vcgenOptions += [ "/raceChecking:SINGLE" ]
      CommandLineOptions.cruncherOptions += [ "/raceChecking:SINGLE" ]
      CommandLineOptions.boogieOptions += [ "/raceChecking:SINGLE" ]
    if args.race_instrumenter == "watchdog-multiple":
      CommandLineOptions.bugleOptions += [ "-race-instrumentation=watchdog-multiple" ]
      CommandLineOptions.vcgenOptions += [ "/raceChecking:MULTIPLE" ]
      CommandLineOptions.cruncherOptions += [ "/raceChecking:MULTIPLE" ]
      CommandLineOptions.boogieOptions += [ "/raceChecking:MULTIPLE" ]

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

  def RunTool(self,ToolName, Command):
    """ Returns a triple (succeeded, timeout, stdout) """
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
    return (returnCode == 0, False, stdout)

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
              [("-D" + str(o)) for o in self.defines])

    if timeout: return ErrorCodes.TIMEOUT, stdout
    if not success: return ErrorCodes.CLANG_ERROR, stdout

    if not self.skip["opt"]:
      success, timeout, stdout = self.RunTool("opt",
              [gvfindtools.llvmBinDir + "/opt"] +
              self.optOptions)

    if timeout: return ErrorCodes.TIMEOUT, stdout
    if not success: return ErrorCodes.OPT_ERROR, stdout

    if self.stop == 'opt': return 0, stdout

    if not self.skip["bugle"]:
      success, timeout, stdout = self.RunTool("bugle",
              [gvfindtools.bugleBinDir + "/bugle"] +
              self.bugleOptions)

    if timeout: return ErrorCodes.TIMEOUT, stdout
    if not success: return ErrorCodes.BUGLE_ERROR, stdout

    if self.stop == 'bugle': return 0, stdout

    if not self.skip["vcgen"]:
      success, timeout, stdout = self.RunTool("gpuverifyvcgen",
              self.mono +
              [gvfindtools.gpuVerifyBinDir + "/GPUVerifyVCGen.exe"] +
              self.vcgenOptions)

    if timeout: return ErrorCodes.TIMEOUT, stdout
    if not success: return ErrorCodes.GPUVERIFYVCGEN_ERROR, stdout

    if self.stop == 'vcgen': return 0, stdout

    if not self.skip["cruncher"]:
      success, timeout, stdout = self.RunTool("gpuverifycruncher",
                self.mono +
                [gvfindtools.gpuVerifyBinDir + os.sep + "GPUVerifyCruncher.exe"] +
                self.cruncherOptions)

    if timeout: return ErrorCodes.TIMEOUT, stdout
    if not success: return ErrorCodes.BOOGIE_ERROR, stdout

    if self.stop == 'cruncher': return 0, stdout

    success, timeout, stdout = self.RunTool("gpuverifyboogiedriver",
            self.mono +
            [gvfindtools.gpuVerifyBinDir + "/GPUVerifyBoogieDriver.exe"] +
            self.boogieOptions)

    if timeout: return ErrorCodes.TIMEOUT, stdout
    if not success: return ErrorCodes.BOOGIE_ERROR, stdout

    if self.silent:
      return 0, ""

    string_builder = io.StringIO()

    if self.mode == AnalysisMode.FINDBUGS:
      print("No defects were found while analysing: " + ", ".join(self.sourceFiles), file=string_builder)
      print("Notes:", file=string_builder)
      print("- Use --loop-unwind=N with N > " + str(self.loopUnwindDepth) + " to search for deeper bugs.", file=string_builder)
      print("- Re-run in verification mode to try to prove absence of defects.", file=string_builder)
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

def parse_header (file):
  code = [x.rstrip() for x in file.readlines()]
  header_args = code[0][len("//"):].split()
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
    with open(f, "r") as kernel_file:
      kernel_args = [f]
      header = kernel_file.readline()
      if header.startswith("//"):
        kernel_args += header[len("//"):].split()
      x = parse_args(kernel_args)
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

def do_json_mode(args):
  kernels = json_load(args.kernel)
  base_path = os.path.dirname(args.kernel.name)

  for kernel in kernels:
    kernel_args = copy.deepcopy(args)
    kernel_name = os.path.join(base_path, kernel.kernel_file)
    kernel_args.kernel = open(kernel_name, "r")
    kernel_args.kernel_name, kernel_args.kernel_ext = \
      os.path.splitext(kernel_name)
    kernel_args.source_language = SourceLanguage.OpenCL
    kernel_args.group_size = kernel.local_size
    kernel_args.global_size = kernel.global_size
    kernel_args.num_groups = kernel.num_groups
    if "compiler_flags" in kernel:
      kernel_args.defines += kernel.compiler_flags.defines
      kernel_args.includes += kernel.compiler_flags.includes
    if "kernel_arguments" in kernel:
      scalar_args = \
        [arg for arg in kernel.kernel_arguments if arg.type == "scalar"]
      scalar_vals = \
        [arg.value if "value" in arg else "*" for arg in scalar_args]
      kernel_args.kernel_args = [[kernel.entry_point] + scalar_vals]
      array_args = \
        [arg for arg in kernel.kernel_arguments if arg.type == "array"]
      array_sizes = \
        [arg.size if "size" in arg else "*" for arg in array_args]
      kernel_args.kernel_arrays = [[kernel.entry_point] + array_sizes]
    _, out = main(kernel_args)
    sys.stdout.write(out)

def main(argv, out=sys.stdout):
  """ This wraps GPUVerify's real main function so
      that we can handle exceptions and trigger our own exit
      commands.

      This is the entry point that should be used if you want
      to use this file as a module rather than as a script.
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
  except Exception:
    # Something went very wrong
    doCleanUp(timing=False, exitCode=0) # It doesn't matter what the exitCode is
    raise

  doCleanUp(timing=True, exitCode=returnCode[0]) # Do this outside try block so we don't call twice!
  return returnCode

debug = False

if __name__ == '__main__':
  try:
    args = parse_args(sys.argv[1:] or [ '--help' ])
    debug = args.debug
    if args.batch_mode:
      do_batch_mode(args)
    elif args.json:
      do_json_mode(args)
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
  except JSONError as e:
    print(str(e), file=sys.stderr)
    sys.exit(ErrorCodes.JSON_ERROR)
  except KeyboardInterrupt:
    sys.exit(ErrorCodes.CTRL_C)

  sys.exit(ErrorCodes.SUCCESS)
