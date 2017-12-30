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

# To properly kill child processes cross platform
try:
  import psutil
except ImportError:
  sys.stderr.write("GPUVerify requires Python to be equipped with the psutil module.\n")
  sys.stderr.write("We recommend installing psutil with pip:\n")
  sys.stderr.write("  pip install psutil\n")
  raise ConfigurationError("Module psutil not found")

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

  def __init__(self, verbose, outFile):
    self.calls = []
    self.verbose = verbose
    self.outFile = outFile

    # The type we will use to represent function calls
    self.fcallType = namedtuple('FCall',['function', 'nargs', 'kargs'])

  def register(self, function, *nargs, **kargs):
    """Register function.

    function : The function to call

    The remaining arguments can be positional or keyword arguments
    to pass to the function.
    """
    call = self.fcallType(function, nargs, kargs)
    self.calls.append(call)

  def call(self):
    """Call registered functions
    """
    for call in self.calls:
      if self.verbose:
        print("Clean up handler Calling " + str(call.function.__name__) + '(' + \
              str(call.nargs) + ', ' + str(call.kargs) + ')', file = self.outFile)
      call.function(*(call.nargs), **(call.kargs))

""" Timing for the toolchain pipeline """
Tools = ["clang", "opt", "bugle", "gpuverifyvcgen", "gpuverifycruncher", "gpuverifyboogiedriver"]
Extensions = { 'clang': ".bc", 'opt': ".opt.bc", 'bugle': ".gbpl", 'gpuverifyvcgen': ".bpl", 'gpuverifycruncher': ".cbpl" }

if os.name == "posix":
  linux_plugin = gvfindtools.bugleBinDir + "/libbugleInlineCheckPlugin.so"
  mac_plugin = gvfindtools.bugleBinDir + "/libbugleInlineCheckPlugin.dylib"
  if os.path.isfile(linux_plugin):
    bugleInlineCheckPlugin = linux_plugin
  elif os.path.isfile(mac_plugin):
    bugleInlineCheckPlugin = mac_plugin
  else:
    raise ConfigurationError('Could not find Bugle Inline Check plugin')

class GPUVerifyInstance (object):
  def __init__ (self, args, outFile, errFile, cleanUpHandler):
    if gvfindtools.useMono:
      if args.debug:
        self.mono = [ 'mono' , '--debug' ]
      else:
        self.mono = [ 'mono' ]
    else:
      self.mono = []

    if args.verbose and args.num_groups and args.group_size:
      print("Got {} groups of size {}".format("x".join(map(str,args.num_groups)),
                                              "x".join(map(str,args.group_size))),
            file = outFile)

    filename = args.kernel_name
    ext = args.kernel_ext

    self.skip = {"clang": False, "opt": False, "bugle": False, "vcgen": False,
      "cruncher": False}

    if ext in [ ".bc", ".opt.bc", ".gbpl", ".bpl", ".cbpl" ]:
      self.skip["clang"] = True
    if ext in [        ".opt.bc", ".gbpl", ".bpl", ".cbpl" ]:
      self.skip["opt"] = True
    if ext in [                   ".gbpl", ".bpl", ".cbpl" ]:
      self.skip["bugle"] = True
    if ext in [                            ".bpl", ".cbpl" ]:
      self.skip["vcgen"] = True
    if ext in [                                    ".cbpl" ]:
      self.skip["cruncher"] = True

    # Intermediate filenames
    bcFilename = filename + '.bc'
    optFilename = filename + '.opt.bc'
    gbplFilename = filename + '.gbpl'
    cbplFilename = filename + '.cbpl'
    bplFilename = filename + '.bpl'
    locFilename = filename + '.loc'

    if not args.keep_temps:
      def DeleteFile(filename):
        """Delete filename if it exists; but do not delete original input"""
        if filename == args.kernel.name:
          return
        try:
          os.remove(filename)
        except OSError:
          pass

      cleanUpHandler.register(DeleteFile, bcFilename)
      if not args.stop == 'opt':
        cleanUpHandler.register(DeleteFile, optFilename)
      if not args.stop == 'bugle':
        cleanUpHandler.register(DeleteFile, gbplFilename)
        cleanUpHandler.register(DeleteFile, locFilename)
      if not args.stop == 'cruncher':
        cleanUpHandler.register(DeleteFile, cbplFilename)
      if not args.stop == 'vcgen':
        cleanUpHandler.register(DeleteFile, bplFilename)

    self.defines = self.getDefines(args)
    self.includes = self.getIncludes(args)

    self.clangOptions = self.getClangOptions(args)
    self.clangOptions += ["-o", bcFilename, args.kernel.name]

    self.optOptions = self.getOptOptions(args)
    self.optOptions += ["-o", optFilename, bcFilename]

    self.bugleOptions = self.getBugleOptions(args)
    self.bugleOptions += ["-s", locFilename, "-o", gbplFilename, optFilename]

    # The .bpl suffix needs to be ignored for /print:
    self.vcgenOptions = self.getVCGenOptions(args)
    self.vcgenOptions += ["/print:" + filename, gbplFilename]

    self.cruncherOptions = self.getCruncherOptions(args)
    self.cruncherOptions += [bplFilename]

    self.boogieOptions = self.getBoogieOptions(args)
    if args.inference and args.mode != AnalysisMode.FINDBUGS:
      self.boogieOptions += [ cbplFilename ]
    else:
      self.boogieOptions += [ bplFilename ]
      self.skip["cruncher"] = True

    self.timing = {}
    self.outFile = outFile
    self.errFile = errFile
    self.stop = args.stop
    self.mode = args.mode
    self.silent = args.silent
    self.sourceFiles = [args.kernel.name]
    self.SL = args.source_language
    self.loopUnwindDepth = args.loop_unwind
    self.onlyDivergence = args.only_divergence
    self.onlyIntraGroup = args.only_intra_group

    self.verbose = args.verbose
    self.time = args.time or (args.time_as_csv is not None)
    self.timeCSVLabel = args.time_as_csv
    self.debug = args.debug
    self.timeout = args.timeout

  def getDefines(self, args):
    defines = ['__BUGLE_' + str(args.size_t) + '__']

    if args.source_language == SourceLanguage.CUDA:
      defines.append("__" + str(len(args.group_size)) + "D_THREAD_BLOCK")
      defines.append("__" + str(len(args.num_groups)) + "D_GRID")

      for index, value in enumerate(args.group_size):
        if value == '*':
          defines.append("__BLOCK_DIM_" + str(index) + "_FREE")
        else:
          defines.append("__BLOCK_DIM_" + str(index) + "=" + str(value))

      for index, value in enumerate(args.num_groups):
        if value == '*':
          defines.append("__GRID_DIM_" + str(index) + "_FREE")
        else:
          defines.append("__GRID_DIM_" + str(index) + "=" + str(value))

      if args.warp_sync:
        defines.append("__WARP_SIZE=" + str(args.warp_sync))
      else:
        defines.append("__WARP_SIZE=32")

    elif args.source_language == SourceLanguage.OpenCL:
      defines += ["__OPENCL_VERSION__=120"]
      defines.append("__" + str(len(args.group_size)) + "D_WORK_GROUP")
      defines.append("__" + str(len(args.num_groups)) + "D_GRID")

      for index, value in enumerate(args.group_size):
        if value == '*':
          defines.append("__LOCAL_SIZE_" + str(index) + "_FREE")
        else:
          defines.append("__LOCAL_SIZE_" + str(index) + "=" + str(value))

      for index, value in enumerate(args.num_groups):
        if value == '*':
          defines.append("__NUM_GROUPS_" + str(index) + "_FREE")
        elif type(value) is tuple:
          defines.append("__NUM_GROUPS_" + str(index) + "_FREE")
          defines.append("__GLOBAL_SIZE_" + str(index) + "=" + str(value[1]))
        else:
          defines.append("__NUM_GROUPS_" + str(index) + "=" + str(value))

      if args.global_offset:
        for index, value in enumerate(args.global_offset):
          defines.append("__GLOBAL_OFFSET_" + str(index) + "=" + str(value))

    if args.only_requires:
      defines.append("ONLY_REQUIRES")

    defines += args.defines
    return defines

  def getIncludes(self, args):
    includes = [gvfindtools.bugleSrcDir + "/include-blang"]

    if args.source_language == SourceLanguage.CUDA:
      pass
    elif  args.source_language == SourceLanguage.OpenCL:
      includes.append(gvfindtools.libclcInstallDir + "/include")

    includes += args.includes
    return includes

  def getClangOptions(self, args):
    options = ["-Wall", "-g", "-gcolumn-info", "-emit-llvm", "-c"]
    options += ["-Xclang", "-disable-O0-optnone"]

    if args.error_limit:
      options.append("-ferror-limit=" + str(args.error_limit))

    if args.source_language == SourceLanguage.CUDA:
      # clang figures out the correct nvptx triple based on the target triple
      # for the host code. The pointer width used matches that of the host.
      if (args.size_t == 32):
        options += [ "-target", "i386--" ] # gives nvptx-nvidia-cuda
      elif (args.size_t == 64):
        options += [ "-target", "x86_64--" ] # gives nvptx64-nvidia-cuda

      options += ["--cuda-device-only", "-nocudainc", "-nocudalib"]
      options += ["--cuda-gpu-arch=sm_35", "-x", "cuda"]
      options += ["-Xclang", "-fcuda-is-device", "-include", "cuda.h"]
    elif args.source_language == SourceLanguage.OpenCL:
      if (args.size_t == 32):
        options += [ "-target", "nvptx--" ]
      elif (args.size_t == 64):
        options += [ "-target", "nvptx64--" ]

      options += ["-x", "cl"]
      options += ["-Xclang", "-cl-std=CL1.2", "-O0", "-fno-builtin",
        "-include", "opencl.h"]

      if os.name == "posix":
        options += ["-Xclang", "-load", "-Xclang", bugleInlineCheckPlugin,
          "-Xclang", "-add-plugin", "-Xclang", "inline-check"]

      options += ["-Xclang", "-mlink-bitcode-file", "-Xclang"]
      if (args.size_t == 32):
        options.append(gvfindtools.libclcInstallDir + "/lib/clc/nvptx--.bc")
      elif (args.size_t == 64):
        options.append(gvfindtools.libclcInstallDir + "/lib/clc/nvptx64--.bc")

    # Must be added after include of opencl/cuda header
    if args.no_annotations or args.only_requires:
      options += [ "-include", "annotations/no_annotations.h" ]
    if args.invariants_as_candidates:
      options += [ "-include", "annotations/candidate_annotations.h" ]

    options += sum([a.split() for a in args.clang_options], [])
    return options

  def getSourceLanguageString(self, args):
    if args.source_language == SourceLanguage.CUDA:
      return "cu"
    elif args.source_language == SourceLanguage.OpenCL:
      return "cl"

  def getOptOptions(self, args):
    options = ["-mem2reg", "-globaldce"]
    options += sum([a.split() for a in args.opt_options], [])
    return options

  def getBugleOptions(self, args):
    options = []

    if args.source_language:
      options += ["-l", self.getSourceLanguageString(args)]

    if args.math_int:
      options += [ "-i", "math" ]
    if not args.no_inline:
      options.append("-inline")

    if args.race_instrumenter == "original":
      options.append("-race-instrumentation=original")
    elif args.race_instrumenter == "watchdog-single":
      options.append("-race-instrumentation=watchdog-single")
    elif args.race_instrumenter == "watchdog-multiple":
      options.append("-race-instrumentation=watchdog-multiple")

    options += sum([["-k", a[0]] for a in args.kernel_args], [])
    options += ["-kernel-array-sizes=" + ','.join(map(str, a)) for a in \
      args.kernel_arrays]
    options += sum([["-k", a[0]] for a in args.kernel_arrays], [])

    if len(args.kernel_args) > 0 or len(args.kernel_arrays) > 0:
      options.append("-only-explicit-entry-points")

    options += sum([a.split() for a in args.bugle_options], [])
    return options

  def getVCGenOptions(self, args):
    options = ["/noPruneInfeasibleEdges"]

    if args.math_int:
      options.append("/mathInt")
    if args.warp_sync:
      options.append("/doWarpSync:" + str(args.warp_sync))
    if args.adversarial_abstraction:
      options.append("/adversarialAbstraction")
    if args.equality_abstraction:
      options.append("/equalityAbstraction")
    if args.check_array_bounds:
      options.append("/checkArrayBounds")
    if args.no_benign_tolerance:
      options.append("/noBenign")
    if args.only_divergence:
      options.append("/onlyDivergence")
    if args.only_intra_group:
      options.append("/onlyIntraGroupRaceChecking")
    if args.only_log:
      options.append("/onlyLog")
    if args.no_barrier_access_checks:
      options.append("/noBarrierAccessChecks")
    if args.asymmetric_asserts:
      options.append("/asymmetricAsserts")

    if args.mode == AnalysisMode.FINDBUGS or (not args.inference):
      options.append("/noInfer")

    if args.debug:
      options.append("/debugGPUVerify")

    if args.race_instrumenter == "original":
      options.append("/raceChecking:ORIGINAL")
    elif args.race_instrumenter == "watchdog-single":
      options.append("/raceChecking:SINGLE")
    elif args.race_instrumenter == "watchdog-multiple":
      options.append("/raceChecking:MULTIPLE")

    options += ["/kernelArgs:" + ','.join(map(str,a)) for a in args.kernel_args]
    options += ["/noCandidate:" + a for a in args.omit_infer]
    options += sum([a.split() for a in args.vcgen_options], [])
    return options

  def getSharedCruncherAndBoogieOptions(self, args):
    options = ["/nologo", "/typeEncoding:m", "/mv:-", "/doModSetAnalysis",
      "/useArrayTheory", "/doNotUseLabels", "/enhancedErrorMessages:1"]

    if args.error_limit:
      options.append("/errorLimit:" + str(args.error_limit))

    if args.source_language:
      options.append("/sourceLanguage:" + self.getSourceLanguageString(args))

    if args.group_size:
      options.append("/blockHighestDim:" + str(len(args.group_size) - 1))
    if args.num_groups:
      options.append("/gridHighestDim:" + str(len(args.num_groups) - 1))

    if not args.math_int:
      options.append("/proverOpt:OPTIMIZE_FOR_BV=true")
      if args.solver == "z3":
        options += ["/z3opt:smt.relevancy=0"]

    if args.solver == "z3":
      options.append("/z3exe:" + gvfindtools.z3BinDir + os.sep + "z3.exe")
    elif args.solver == "cvc4":
      options.append("/proverOpt:SOLVER=cvc4")
      options.append("/cvc4exe:" + gvfindtools.cvc4BinDir + os.sep + "cvc4.exe")
      options.append("/proverOpt:LOGIC=QF_ALL_SUPPORTED")

    if args.gen_smt2:
      options.append("/proverLog:" + args.kernel_name + ".smt2")

    if args.only_intra_group:
      options.append("/onlyIntraGroupRaceChecking")

    if args.race_instrumenter == "original":
      options.append("/raceChecking:ORIGINAL")
    elif args.race_instrumenter == "watchdog-single":
      options.append("/raceChecking:SINGLE")
    elif args.race_instrumenter == "watchdog-multiple":
      options.append("/raceChecking:MULTIPLE")

    if args.k_induction_depth > 0:
      options.append("/kInductionDepth:" + str(args.k_induction_depth))

    if args.debug:
      options.append("/debugGPUVerify")

    return options

  def getCruncherOptions(self, args):
    options = self.getSharedCruncherAndBoogieOptions(args)
    options += ["/noinfer", "/contractInfer", "/concurrentHoudini"]

    if args.infer_info:
      options.append("/trace")

    options += [f.name for f in args.boogie_file]
    options += sum([a.split() for a in args.cruncher_options], [])
    return options

  def getBoogieOptions(self, args):
    options = self.getSharedCruncherAndBoogieOptions(args)

    if args.mode == AnalysisMode.FINDBUGS:
      options.append("/loopUnroll:" + str(args.loop_unwind))

    options += sum([a.split() for a in args.boogie_options], [])
    return options

  def run(self, command):
    """ Run a command with an optional timeout. A timeout of zero
        implies no timeout.
    """
    popenargs={}
    if self.verbose:
      print(" ".join(command), file = self.outFile)
      self.outFile.flush()
    else:
      popenargs['bufsize']=0

    popenargs["stdout"] = self.outFile
    popenargs["stderr"] = self.errFile

    # Redirect stdin, othewise terminal text becomes unreadable after timeout
    popenargs['stdin']=subprocess.PIPE

    proc = psutil.Popen(command, **popenargs)
    if "get_children" not in dir(proc):
      proc.get_children = proc.children
    if args.timeout > 0:
      try:
        return_code = proc.wait(timeout = self.timeout)
      except psutil.TimeoutExpired:
        children = proc.get_children(recursive=True)
        proc.terminate()
        for child in children:
          try:
            child.terminate()
          except psutil.NoSuchProcess:
            pass
        raise
    else:
      return_code = proc.wait()

    return return_code

  def runTool(self, ToolName, Command):
    """ Returns a triple (succeeded, timeout) """
    assert ToolName in Tools
    if self.verbose:
      print("Running " + ToolName, file=self.outFile)
      self.outFile.flush()
    try:
      start = timeit.default_timer()
      exitCode = self.run(Command)
      end = timeit.default_timer()
    except psutil.TimeoutExpired:
      self.timing[ToolName] = self.timeout
      print("{} timed out. Use --timeout=N with N > {} to increase timeout, or --timeout=0 to disable timeout.\n".format(ToolName, self.timeout), file=self.outFile)
      return False, True
    except (OSError,WindowsError) as e:
      print("Error while invoking {} : {}".format(ToolName, str(e)))
      print("With command line args:")
      print(pprint.pformat(Command))
      raise
    self.timing[ToolName] = end-start
    return exitCode, False

  def interpretBoogieDriverCrucherExitCode(self, ExitCode):
    assert ExitCode != 0
    # See GPUVerifyLib/ToolExitCodes.cs
    if ExitCode == 3:
      # A verification error report will be emitted
      return ErrorCodes.NOT_ALL_VERIFIED
    elif ExitCode == 2:
      return ErrorCodes.BOOGIE_OTHER_ERROR
    elif ExitCode == 1:
      # Something went very wrong internally
      return ErrorCodes.BOOGIE_INTERNAL_ERROR
    else:
      assert False

  def invoke (self):
    """ Returns (returncode, outstring) """

    if not self.skip["clang"]:
      success, timeout = self.runTool("clang",
              [gvfindtools.llvmBinDir + "/clang"] +
              self.clangOptions +
              [("-I" + str(o)) for o in self.includes] +
              [("-D" + str(o)) for o in self.defines])

      if timeout: return ErrorCodes.TIMEOUT
      if success != 0: return ErrorCodes.CLANG_ERROR

    if not self.skip["opt"]:
      success, timeout = self.runTool("opt",
              [gvfindtools.llvmBinDir + "/opt"] +
              self.optOptions)

      if timeout: return ErrorCodes.TIMEOUT
      if success != 0: return ErrorCodes.OPT_ERROR

    if self.stop == 'opt': return ErrorCodes.SUCCESS

    if not self.skip["bugle"]:
      success, timeout = self.runTool("bugle",
              [gvfindtools.bugleBinDir + "/bugle"] +
              self.bugleOptions)

      if timeout: return ErrorCodes.TIMEOUT
      if success != 0: return ErrorCodes.BUGLE_ERROR

    if self.stop == 'bugle': return ErrorCodes.SUCCESS

    if not self.skip["vcgen"]:
      success, timeout = self.runTool("gpuverifyvcgen",
              self.mono +
              [gvfindtools.gpuVerifyBinDir + "/GPUVerifyVCGen.exe"] +
              self.vcgenOptions)

      if timeout: return ErrorCodes.TIMEOUT
      if success != 0: return ErrorCodes.GPUVERIFYVCGEN_ERROR

    if self.stop == 'vcgen': return ErrorCodes.SUCCESS

    if not self.skip["cruncher"]:
      success, timeout = self.runTool("gpuverifycruncher",
                self.mono +
                [gvfindtools.gpuVerifyBinDir + os.sep + "GPUVerifyCruncher.exe"] +
                self.cruncherOptions)

      if timeout: return ErrorCodes.TIMEOUT
      if success != 0:
        return self.interpretBoogieDriverCrucherExitCode(success)

    if self.stop == 'cruncher': return ErrorCodes.SUCCESS

    success, timeout = self.runTool("gpuverifyboogiedriver",
            self.mono +
            [gvfindtools.gpuVerifyBinDir + "/GPUVerifyBoogieDriver.exe"] +
            self.boogieOptions)

    if timeout: return ErrorCodes.TIMEOUT
    if success != 0:
      return self.interpretBoogieDriverCrucherExitCode(success)

    if self.silent:
      return ErrorCodes.SUCCESS

    if self.mode == AnalysisMode.FINDBUGS:
      print("No defects were found while analysing: " + ", ".join(self.sourceFiles), file=self.outFile)
      print("Notes:", file=self.outFile)
      print("- Use --loop-unwind=N with N > " + str(self.loopUnwindDepth) + " to search for deeper bugs.", file=self.outFile)
      print("- Re-run in verification mode to try to prove absence of defects.", file=self.outFile)
    else:
      print("Verified: " + ", ".join(self.sourceFiles), file=self.outFile)
      if not self.onlyDivergence:
        print("- no data races within " + ("work groups" if self.SL == SourceLanguage.OpenCL else "thread blocks"), file=self.outFile)
        if not self.onlyIntraGroup:
          print("- no data races between " + ("work groups" if self.SL == SourceLanguage.OpenCL else "thread blocks"), file=self.outFile)
      print("- no barrier divergence", file=self.outFile)
      print("- no assertion failures", file=self.outFile)
      if args.check_array_bounds:
        print("- no out-of-bounds array accesses (for arrays where size information is available)", file=self.outFile)
      print("(but absolutely no warranty provided)", file=self.outFile)

    return ErrorCodes.SUCCESS

  def getTiming(self, exitCode):
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
      string_builder = io.StringIO()
      print("Timing information (%.2f secs):" % sum(self.timing.values()), file=string_builder)
      if self.timing:
        padTool = max([ len(tool) for tool in list(self.timing.keys()) ])
        padTime = max([ len('%.3f secs' % t) for t in list(self.timing.values()) ])
        for tool in Tools:
          if tool in self.timing:
            print("- %s : %s" % (tool.ljust(padTool), ('%.3f secs' % self.timing[tool]).rjust(padTime)), file=string_builder)
        return string_builder.getvalue()[:-1]
      else:
        return "- no tools ran"

def json_list_kernels(kernels, json_to_kernel_map, success_cache):
  index_width = len(str(len(json_to_kernel_map) - 1))
  index_format = "[{: >" + str(index_width) + "}] Name: {}"
  prefix = ' ' * (index_width + 2)

  file_format     = prefix + " File: {}"
  size_format     = prefix + " Local size = [{}] Global size = [{}]"
  compiler_format = prefix + " Compiler flags: {}"
  arg_format      = prefix + " Argument {}: {}"
  arg_val_format  = prefix + " Argument {}: {} {}"
  build_format    = prefix + " Built at {}:{}"
  run_format      = prefix + " Ran at {}:{}"
  cache_format    = prefix + " In success cache"

  for index, k in enumerate(json_to_kernel_map):
    print(index_format.format(index, kernels[k].entry_point))
    print(file_format.format(kernels[k].kernel_file))
    print(size_format.format(",".join(map(str, kernels[k].local_size)),
                             ",".join(map(str, kernels[k].global_size))))
    if "compiler_flags" in kernels[k]:
      print(compiler_format.format(kernels[k].compiler_flags.original))
    if "kernel_arguments" in kernels[k]:
      for arg_index, arg in enumerate(kernels[k].kernel_arguments):
        if arg.type == "scalar" and arg.value:
          print(arg_val_format.format(arg_index, "scalar with value", arg.value))
        elif arg.type == "array" and arg.size:
          print(arg_val_format.format(arg_index, "array of size", arg.size))
        else:
          print(arg_format.format(arg_index, arg.type))
    if "host_api_calls" in kernels[k]:
      for call in kernels[k].host_api_calls:
        if call.function_name == "clCreateProgramWithSource":
          print(build_format.format(call.compilation_unit, call.line_number))
      for call in kernels[k].host_api_calls:
        if call.function_name == "clEnqueueNDRangeKernel":
          print(run_format.format(call.compilation_unit, call.line_number))
    if kernels[k] in success_cache:
      print(cache_format)

def json_verify_kernel(args, base_path, kernel, success_cache):
  if kernel in success_cache:
    return ErrorCodes.SUCCESS, "Verified: Found result in success cache"

  kernel_args = copy.deepcopy(args)
  kernel_name = os.path.join(base_path, kernel.kernel_file)
  try:
    kernel_args.kernel = open(kernel_name, "r")
  except IOError as e:
    raise JSONError(str(e))
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
    scalar_args = [arg for arg in kernel.kernel_arguments \
                     if arg.type == "scalar" or arg.type == "sampler"]
    scalar_vals = [arg.value if "value" in arg else "*" \
                     for arg in scalar_args]
    kernel_args.kernel_args = [[kernel.entry_point] + scalar_vals]
    array_args = [arg for arg in kernel.kernel_arguments \
                    if arg.type == "array" or arg.type == "image"]
    array_sizes = [arg.size if "size" in arg else "*" \
                     for arg in array_args]
    kernel_args.kernel_arrays = [[kernel.entry_point] + array_sizes]
  outFile = tempfile.SpooledTemporaryFile()
  return_code = main(kernel_args, outFile, subprocess.STDOUT)
  outFile.seek(0)
  out_data = outFile.read()[:-1]
  outFile.close()

  if return_code == ErrorCodes.SUCCESS:
    success_cache.append(kernel)

  return return_code, out_data

def json_verify_all(args, kernels, json_to_kernel_map, success_cache):
  base_path = os.path.dirname(args.kernel.name)
  Result = namedtuple("Result", ["error_code", "output", "kernel"])
  results = []

  progress_format = "Executed {} of " + str(len(kernels)) + \
    " verification tasks for " + str(len(json_to_kernel_map)) + \
    " intercepted kernels"
  progress_text = ""
  for i, k in enumerate(kernels):
    progress_text = progress_format.format(i)
    print(progress_text, end = '\r')
    sys.stdout.flush()
    return_code, out_data = json_verify_kernel(args, base_path, k, success_cache)
    results.append(Result(return_code, out_data, k))
  print(' ' * len(progress_text), end = '\r')

  success = []
  failure = []
  for i, k in enumerate(json_to_kernel_map):
    if results[k].error_code == ErrorCodes.SUCCESS:
      success.append((i, results[k]))
    else:
      failure.append((i, results[k]))

  print("GPUVerify kernel analyzer checked {} kernels.".format(len(success) + len(failure)))
  print("Successfully verified {} kernels.".format(len(success)))
  print("Failed to verify {} kernels.".format(len(failure)))

  index_width = len(str(len(json_to_kernel_map) - 1))
  result_format = "[{: >" + str(index_width) + "}]: Verification of {} ({}) {} with: local size = [{}] global size = [{}]"
  if len(success) > 0:
    print("")
    print("Successes:")
    for s, r in success:
      print(result_format.format(s,
                                 r.kernel.entry_point,
                                 r.kernel.kernel_file,
                                 "succeeded",
                                 ",".join(map(str, r.kernel.local_size)),
                                 ",".join(map(str, r.kernel.global_size))
                                 ))

  if len(success) > 0 and len(failure) > 0:
    print("")

  if len(failure) > 0:
    print("Failures:")
    for f, r in failure:
      print(result_format.format(f,
                                 r.kernel.entry_point,
                                 r.kernel.kernel_file,
                                 "failed",
                                 ",".join(map(str, r.kernel.local_size)),
                                 ",".join(map(str, r.kernel.global_size))
                                 ))

def json_verify_intercepted(args, kernels, json_to_kernel_map, success_cache):
  if args.verify_intercepted >= len(json_to_kernel_map):
    raise JSONError("No kernel " + str(args.verify_intercepted) + " in JSON file")

  k = kernels[json_to_kernel_map[args.verify_intercepted]]
  return_code, out_data = json_verify_kernel(args, os.path.dirname(args.kernel.name), k, success_cache)

  result_format = "Verification of {} ({}) {} with: local size = [{}] global size = [{}]"
  print(result_format.format(k.entry_point,
                             k.kernel_file,
                             "succeeded" if return_code == ErrorCodes.SUCCESS else "failed",
                             ",".join(map(str, k.local_size)),
                             ",".join(map(str, k.global_size))
                            ))

  if return_code != ErrorCodes.SUCCESS:
    print("The error message is:")
    print(out_data)

def do_json_mode(args):
  kernels, json_to_kernel_map = json_load(args.kernel)

  if args.cache != None:
    try:
      success_cache = pickle.load(open(args.cache))
    except:
      success_cache = []
  else:
    success_cache = []

  if args.list_intercepted:
    json_list_kernels(kernels, json_to_kernel_map, success_cache)
  elif args.verify_all_intercepted:
    json_verify_all(args, kernels, json_to_kernel_map, success_cache)
  elif args.verify_intercepted != None:
    json_verify_intercepted(args, kernels, json_to_kernel_map, success_cache)
  else:
    base_path = os.path.dirname(args.kernel.name)
    for kernel in kernels:
      print("Verifying " + kernel.entry_point)
      return_code, out_data = json_verify_kernel(args, base_path, kernel, success_cache)
      print(out_data)

  if args.cache != None:
    pickle.dump(success_cache, open(args.cache, "w"))

def main(args, out, err):
  """ This wraps GPUVerify's real main function so
      that we can handle exceptions and trigger our own exit
      commands.

      This is the entry point that should be used if you want
      to use this file as a module rather than as a script.
  """
  cleanUpHandler = BatchCaller(args.verbose, out)
  gv_instance = GPUVerifyInstance(args, out, err, cleanUpHandler)

  def handleTiming(exitCode):
    if gv_instance.time:
      print(gv_instance.getTiming(exitCode), file = out)

  def doCleanUp(timing, exitCode):
    if timing:
      # We must call this before cleaning up globals
      # because it depends on them
      cleanUpHandler.register(handleTiming, exitCode)

    # We should call this last.
    cleanUpHandler.call()

  try:
    returnCode = gv_instance.invoke()
  except Exception:
    # Something went very wrong
    doCleanUp(timing = False, exitCode = 0) # It doesn't matter what the exitCode is
    raise

  doCleanUp(timing = True, exitCode = returnCode) # Do this outside try block so we don't call twice!
  return returnCode

if __name__ == '__main__':
  try:
    args = parse_arguments(sys.argv[1:], gvfindtools.defaultSolver,
      gvfindtools.llvmBinDir, getversion)
    if args.json:
      do_json_mode(args)
      sys.exit(ErrorCodes.SUCCESS)
    else:
      rc = main(args, sys.stdout, sys.stderr)
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
