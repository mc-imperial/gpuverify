"""Module for parsing GPUVerify command line arguments"""

import argparse
import os
import subprocess

from .constants import AnalysisMode, SourceLanguage
from .error_codes import ErrorCodes
from .util import is_hex_string, is_positive_string, GlobalSizeError, \
  get_num_groups

class ArgumentParserError(Exception):
  def __init__(self, msg):
    self.msg = msg

  def __str__(self):
    return "GPUVerify: COMMAND_LINE_ERROR error ({}): {}" \
      .format(ErrorCodes.COMMAND_LINE_ERROR, self.msg)

class __ArgumentParser(argparse.ArgumentParser):
  def error (self, message):
    raise ArgumentParserError(message)

class _VersionAction(argparse.Action):
  def __init__(self, option_strings, version, **kwargs):
    super(_VersionAction, self).__init__(option_strings, nargs = 0, **kwargs)
    self.version = version

  def __call__(self, parser, namespace, values, option_string = None):
    parser.exit(message = self.version.getVersionString())

def __non_negative(string):
  try:
    i = int(string)
  except ValueError:
    raise argparse.ArgumentTypeError("argument must be a non-negative integer")

  if i < 0:
    raise argparse.ArgumentTypeError("negative value {} given as argument" \
      .format(i))

  return i

def __positive(string):
  try:
    i = int(string)
  except ValueError:
    raise argparse.ArgumentTypeError("Argument must be a positive integer")

  if i <= 0:
    raise argparse.ArgumentTypeError("non-positive value {} given as argument" \
      .format(i))

  return i

def __dimensions(string):
  string = string.strip()

  if len(string) == 0:
    raise argparse.ArgumentTypeError("dimensions must be vectors of length 1-3")

  string = string[1:-1] if string[0] == "[" and string[-1] == "]" else string

  try:
    values = [x if x == '*' else int(x) for x in string.split(",")]
  except ValueError:
    raise argparse.ArgumentTypeError("a dimension must be a positive integer")

  if len(values) == 0 or len(values) > 3:
    raise argparse.ArgumentTypeError("dimensions must be vectors of length 1-3")

  if len([x for x in values if isinstance(x, str) or x > 0]) < len(values):
    raise argparse.ArgumentTypeError("a dimension must be a positive integer")

  return values

def __offsets(string):
  string = string.strip()

  if len(string) == 0:
    raise argparse.ArgumentTypeError("offsets must be vectors of length 1-3")

  string = string[1:-1] if string[0] == "[" and string[-1] == "]" else string

  try:
    values = [int(x) for x in string.split(",")]
  except ValueError:
    raise argparse.ArgumentTypeError("an offset must be a positive integer")

  if len(values) == 0 or len(values) > 3:
    raise argparse.ArgumentTypeError("offsets must be vectors of length 1-3")

  if len([x for x in values if x >= 0]) < len(values):
    raise argparse.ArgumentTypeError("an offset must be a positive integer")

  return values

def __kernel_arguments(string):
  values = string.strip().split(",")

  if (len(values) < 1):
    raise argparse.ArgumentTypeError("kernel arguments should include a " +
      "kernel entry point as first element")

  if not all(x == '*' or is_hex_string(x) for x in values[1:]):
    raise argparse.ArgumentTypeError("kernel arguments must be hex values or *")

  return values

def __kernel_arrays(string):
  values = string.strip().split(",")

  if (len(values) < 1):
    raise argparse.ArgumentTypeError("kernel arrays should include a " +
      "kernel entry point as first element")

  if not all(x == '*' or is_positive_string(x) for x in values[1:]):
    raise argparse.ArgumentTypeError("kernel arguments must be ints or *")

  return values

def __build_parser(default_solver, version):
  parser = __ArgumentParser(description = "GPUVerify frontend",
    usage = "gpuverify [options] <kernel>")

  parser.add_argument("kernel", type = argparse.FileType('r'),
    help = "Kernel file to verify")

  general = parser.add_argument_group("GENERAL OPTIONS")

  general.add_argument("--version", action = _VersionAction, version = version,
    help = "Show version information and exit")

  general.add_argument("-D", dest = 'defines',  default = [], action = 'append',
    metavar = "<value>", help = "Define symbol")
  general.add_argument("-I", dest = 'includes', default = [], action = 'append',
    metavar = "<value>", help = "Add directory to include search path")

  mode = general.add_mutually_exclusive_group()
  mode.add_argument("--findbugs", dest = 'mode', action = 'store_const',
    const = AnalysisMode.FINDBUGS, help = "Run tool in bug-finding mode")
  mode.add_argument("--verify", dest = 'mode', action = 'store_const',
    const = AnalysisMode.VERIFY, help = "Run tool in verification mode")

  general.add_argument("--loop-unwind=", type = __non_negative, metavar = "X",
    help = "Explore traces that pass through at most X loop heads. Implies \
      --findbugs")

  general.add_argument("--check-array-bounds", action = 'store_true',
    help = "Enable checking for any array out-of-bounds access")

  general.add_argument("--no-benign-tolerance", action = 'store_true',
    help = "Do not tolerate benign data races")

  general.add_argument("--only-divergence",  action = 'store_true',
    help = "Only check for barrier divergence, not races")
  general.add_argument("--only-intra-group", action = 'store_true',
    help = "Do not check for inter-group races")

  verbosity = general.add_mutually_exclusive_group()
  verbosity.add_argument("--verbose", action = 'store_true',
    help = "Show subcommands and use verbose output")
  verbosity.add_argument("--silent",  action = 'store_true',
    help = "Silent on success; only show errors/timing")

  general.add_argument("--time", action = 'store_true',
    help = "Show timing information")
  general.add_argument("--time-as-csv=", metavar = "X",
    help = "Print timing information as CSV with label X")

  general.add_argument("--timeout=", type = __non_negative, default = 300,
    metavar = "X", help = "Allow each component to run for at most X seconds. \
    A timeout of 0 disables the timeout. The default is 300s")

  general.add_argument("--error-limit=", type = __positive,
    metavar = "X", help = "Limit the number of errors printed to X.")

  language = general.add_mutually_exclusive_group()
  language.add_argument("--opencl", dest = 'source_language',
    action = 'store_const', const = SourceLanguage.OpenCL,
    help = "Assume the kernel is an OpenCL kernel")
  language.add_argument("--cuda", dest = 'source_language',
    action = 'store_const', const = SourceLanguage.CUDA,
    help = "Assume the kernel is a CUDA kernel")

  sizing = parser.add_argument_group("SIZING", "Define the dimensionality and \
    size of each dimension as a 1-, 2-, or 3-tuple, optionally wrapped in []. \
    For example, [1,2] or 2,3,4. Use * for an unconstrained value")

  lsize = sizing.add_mutually_exclusive_group()
  numg = sizing.add_mutually_exclusive_group()

  lsize.add_argument("--local_size=", dest = 'group_size', type = __dimensions,
    help = "Specify the dimensions of an OpenCL work-group. This corresponds \
    to the 'local_work_size' parameter of 'clEnqueueNDRangeKernel'.")
  numg.add_argument("--global_size=", dest = 'global_size', type = __dimensions,
    help = "Specify dimensions of the OpenCL NDRange. This corresponds to the \
    'global_work_size' parameter of 'clEnqueueNDRangeKernel'. Mutually \
    exclusive with --num_groups")
  numg.add_argument("--num_groups=", dest = 'num_groups', type = __dimensions,
    help = "Specify the dimensions of a grid of OpenCL work-groups. Mutually \
    exclusive with --global_size")
  sizing.add_argument("--global_offset=", dest = 'global_offset',
    type = __offsets, help = "Specify the OpenCL global offset. This \
    corresponds to the 'global_offset' parameter of 'clEnqueueNDRangeKernel'.")

  lsize.add_argument("--blockDim=", dest = 'group_size', type = __dimensions,
    help = "Specify the CUDA thread block size")
  numg.add_argument("--gridDim=", dest = 'num_groups', type = __dimensions,
    help = "Specify the CUDA grid size")

  advanced = parser.add_argument_group("ADVANCED OPTIONS")
  advanced.add_argument("--pointer-bitwidth=", dest = 'size_t', type = int,
    choices = [32, 64], default = 32, help = "Set the pointer bitwidth. The \
    default is 32")

  abstraction = advanced.add_mutually_exclusive_group()
  abstraction.add_argument("--adversarial-abstraction", action = 'store_true',
    help = "Abstract shared state, so that reads are non-deterministic")
  abstraction.add_argument("--equality-abstraction", action = 'store_true',
    help = "At barriers, make shared arrays non-deterministic but consistent \
    between threads")

  advanced.add_argument("--asymmetric-asserts", action = 'store_true',
    help = "Emit assertions only for the first thread. Sound, and may lead to \
    faster verification, but can yield false positives")

  advanced.add_argument("--boogie-file=", type = argparse.FileType('r'),
    default = [], action = 'append', metavar = "X.bpl", help = "Specify a \
    supporting .bpl file to be used during verification")

  advanced.add_argument("--math-int", action = 'store_true', help = "Represent \
    integer types using mathematical integers instead of bit-vectors")

  annotations = advanced.add_mutually_exclusive_group()
  annotations.add_argument("--no-annotations", action = 'store_true',
    help = "Ignore all source-level annotations")
  annotations.add_argument("--only-requires", action = 'store_true',
    help = "Ignore all source-level annotations except for requires")

  advanced.add_argument("--invariants-as-candidates", action='store_true',
    help = "Interpret all source-level invariants as candidates")
  advanced.add_argument("--no-barrier-access-checks", action='store_true',
    help = "Turn off access checks for barrier invariants")

  advanced.add_argument("--no-inline", action = 'store_true',
    help = "Turn off automatic function inlining")
  advanced.add_argument("--only-log", action = 'store_true',
    help = "Log accesses to arrays, but do not check for races. This can be \
   useful for determining access pattern invariants")

  advanced.add_argument("--kernel-args=", type = __kernel_arguments,
    default = [], action = 'append', metavar = "K,v1,...,vn", help = "For \
    kernel K with scalar parameters x1, ..., xn, add the preconditions \
    x1 == v1, ..., xn == vn. Use * to denote an unconstrained parameter")
  advanced.add_argument("--kernel-arrays=", type = __kernel_arrays,
    default = [], action = 'append', metavar = "K,s1,...,sn", help = "For \
    kernel K with array parameters p1, ..., pn, assume that sizeof(p1) == \
    s1, ... sizeof(pn) == sn. Use * to denote an unconstrained size")

  advanced.add_argument("--warp-sync=", type = __positive, metavar = "X",
    help = "Synchronize threads within warps of size X.")

  advanced.add_argument("--race-instrumenter=", choices = ["original",
    "watchdog-single", "watchdog-multiple"], default = "watchdog-single",
    help = "Choose which method of race instrumentation to use. The default is \
    watchdog-single")

  advanced.add_argument("--solver=", choices = ["z3", "cvc4"],
    default = default_solver, help = "Select the SMT solver to use as \
    backend. Default is {}".format(default_solver))

  development = parser.add_argument_group("DEVELOPMENT OPTIONS")
  development.add_argument("--debug", action = 'store_true',
    help = "Enable debugging of GPUVerify components: exceptions will not be \
    suppressed")
  development.add_argument("--keep-temps", action = 'store_true',
    help = "Keep the intermediate bc, gbpl, and cbpl files")
  development.add_argument("--gen-smt2", action = 'store_true',
    help = "Generate an smt2 file")

  development.add_argument("--clang-opt=", dest = 'clang_options', default = [],
    action = 'append', help = "Specify option to be passed to Clang")
  development.add_argument("--opt-opt=", dest = 'opt_options', default = [],
    action = 'append', help = "Specify option to be passed to the LLVM \
    optimization pass")
  development.add_argument("--bugle-opt=", dest='bugle_options', default = [],
    action = 'append', help = "Specify option to be passed to Bugle")
  development.add_argument("--vcgen-opt=", dest='vcgen_options', default = [],
    action = 'append', help = "Specify option to be passed to VC generator")
  development.add_argument("--cruncher-opt=", dest = 'cruncher_options',
     default = [], action = 'append', help = "Specify option to be passed to \
     the cruncher")
  development.add_argument("--boogie-opt=", dest='boogie_options', default = [],
    action = 'append', help = "Specify option to be passed to Boogie")

  stopping = advanced.add_mutually_exclusive_group()
  stopping.add_argument("--stop-at-opt",  dest = 'stop', action = 'store_const',
    const = "opt", help = "Stop after LLVM optimization pass")
  stopping.add_argument("--stop-at-gbpl", dest = 'stop', action = 'store_const',
    const = "bugle", help = "Stop after generating gbpl file")
  stopping.add_argument("--stop-at-bpl",  dest = 'stop', action = 'store_const',
    const = "vcgen", help = "Stop after generating bpl file")
  stopping.add_argument("--stop-at-cbpl", dest = 'stop', action = 'store_const',
    const = "cruncher", help = "Stop after generating an annotated bpl file")

  inference = parser.add_argument_group("INVARIANT INFERENCE OPTIONS")
  inference.add_argument("--no-infer", dest = 'inference',
    action = 'store_false', help = "Turn off invariant inference")
  inference.add_argument("--omit-infer=", default = [], action = 'append',
    metavar = "X", help = "Do not generate invariants of type 'X'")
  inference.add_argument("--infer-info", action = 'store_true',
    help = "Prints information about the invariant inference process")
  inference.add_argument("--k-induction-depth=", type = __positive, default = 0,
    metavar = "X", help = "Applies k-induction with k=X to all loops")

  json = parser.add_argument_group("JSON MODE")
  json.add_argument("--json", action = 'store_true', help = "The kernels to be \
    verified are specified through a JSON file")

  json_options = json.add_mutually_exclusive_group()
  json_options.add_argument("--list-intercepted", action = 'store_true',
    help = "List the kernels specified in the JSON file")
  json_options.add_argument("--verify-intercepted=", type = __non_negative,
    metavar = "X", help = "Verify kernel 'X' from the JSON file")
  json_options.add_argument("--verify-all-intercepted", action = 'store_true',
    help = "Verify all the kernels from the JSON file")

  json.add_argument("--cache=", metavar = "X", help = "Use 'X' as result cache")

  return parser

class __ldict(dict):
  def __getattr__(self, name):
    if name in self:
      return self[name]
    else:
      raise AttributeError(name)

  def __setattr__(self, name, value):
    self[name] = value

def __to_ldict(parsed):
  def strip_equals(string):
    return string[:-1] if string.endswith('=') else string

  return __ldict({strip_equals(k) : v for k, v in vars(parsed).items()})

def __split_filename_ext(f):
  filename, ext = os.path.splitext(f)
  if filename.endswith(".opt") and ext == ".bc":
    filename, _ = os.path.splitext(filename)
    ext = ".opt.bc"
  return filename, ext

def __need_source_language(ext):
  return ext not in [".gbpl", ".bpl", ".cbpl"]

def __need_dimensions(ext):
  return ext not in [".bc", ".opt.bc", ".gbpl", ".bpl", ".cbpl"]

def __get_source_language(args, parser, llvm_bin_dir):
  if args.source_language:
    return source_language

  if args.kernel_ext == ".cl":
    return SourceLanguage.OpenCL
  if args.kernel_ext == ".cu":
    return SourceLanguage.CUDA

  if args.kernel_ext in [".bc", ".opt.bc"]:
    command = [os.path.join(llvm_bin_dir,"llvm-nm"), args.kernel.name]
    popenargs = {"stdout" : subprocess.PIPE, "stderr" : subprocess.STDOUT,
      "stdin" : subprocess.PIPE}
    proc = subprocess.Popen(command, **popenargs)
    lines, _ = proc.communicate()
  else:
    lines = ''.join(args['kernel'].readlines())

  if any(x in lines for x in ["get_local_size", "get_global_size", "__kernel"]):
    return SourceLanguage.OpenCL
  elif any(x in lines for x in ["blockDim", "__global__"]):
    return SourceLanguage.CUDA
  else:
    parser.error("Could not infer source language")

def __get_num_groups(args, parser):
  if args.group_size and args.global_size:
    try:
      return get_num_groups(args.group_size, args.global_size)
    except GlobalSizeError as e:
      parser.error(str(e))
  elif args.group_size and args.num_groups:
    return args.num_groups
  elif __need_dimensions(args.kernel_ext):
    if args.source_language == SourceLanguage.OpenCL:
      parser.error("Must specify thread dimensions with --local_size and " +
        "--global_size")
    elif args.source_language == SourceLanguage.CUDA:
      parser.error("Must specify thread dimensions with --blockDim and " +
        "--gridDim")

  return None

def __check_global_offset(args, parser):
  if not args.global_offset:
    return
  elif args.source_language == SourceLanguage.CUDA:
    parser.error("Cannot specify --global_offset for CUDA kernels")
  elif len(args.global_offset) != len(args.num_groups):
    parser.error("Dimensions of global offset and global size must match")
  else:
    return

def parse_arguments(argv, default_solver, llvm_bin_dir, version):
  parser = __build_parser(default_solver, version)
  args = __to_ldict(parser.parse_args(argv))

  if not args.loop_unwind and not args.mode:
    args.mode = AnalysisMode.ALL
  elif args.loop_unwind:
    args.mode = AnalysisMode.FINDBUGS
  elif args.mode == AnalysisMode.FINDBUGS and not args.loop_unwind:
    args.loop_unwind = 2

  if args.version:
    return args

  if args.json:
    if args.group_size or args.num_groups or \
       args.global_size or args.global_offset:
      parser.error("Sizing arguments incompatible with JSON mode")
  else:
    if args.list_intercepted or args.verify_all_intercepted or \
       args.verify_intercepted != None or args.cache != None:
      parser.error("JSON options require JSON mode");

  if not args.stop:
    args.stop = "boogie"

  if args.json:
    return args

  args.kernel_name, args.kernel_ext = __split_filename_ext(args.kernel.name)

  if not args.source_language and __need_source_language(args.kernel_ext):
    args.source_language = __get_source_language(args, parser, llvm_bin_dir)

  args.num_groups = __get_num_groups(args, parser)
  __check_global_offset(args, parser)

  return args
