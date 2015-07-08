"""Module for loading JSON files with GPUVerify invocation data."""

import json
from collections import namedtuple

from .error_codes import ErrorCodes
from .util import is_hex_string, GlobalSizeError, get_num_groups

class JSONError(Exception):
  """Exception type returned by json_load."""
  def __init__(self, msg):
    self.msg = msg

  def __str__(self):
    return "GPUVerify: JSON_ERROR error ({}): {}" \
      .format(ErrorCodes.JSON_ERROR, self.msg)

class __ldict(dict):
  def __getattr__(self, name):
    if name in self:
      return self[name]
    else:
      raise AttributeError(name)

  def __setattr__(self, name, value):
    self[name] = value

def __check_string(data, object_name):
  if not type(data) is unicode:
    raise JSONError(object_name + " expects string")

def __check_hex_string(data, object_name):
  if not type(data) is unicode:
    raise JSONError(object_name + " expects hex string")
  if not is_hex_string(data):
    raise JSONError(object_name + " expects hex string")

def __check_positive_number(data, object_name):
  if not type(data) is int:
    raise JSONError(object_name + " expects number > 0")
  if data <= 0:
    raise JSONError(object_name + " expects number > 0")

def __check_array_of_positive_numbers(data, object_name):
  if not type(data) is list:
    raise JSONError(object_name + " expects an array of numbers > 0")

  if not all(type(i) is int and i > 0 for i in data):
    raise JSONError(object_name + " expects an array of numbers > 0")

def __check_array_of_strings(data, object_name):
  if not type(data) is list:
    raise JSONError(object_name + " expects an array of strings, received " + str(data))

  if not all(type(i) is unicode for i in data):
    raise JSONError(object_name + " expects an array of strings, received " + str(data))

def __check_scalar_argument(data):
  for key, value in data.items():
    if key == "type":
      pass
    elif key == "value":
      __check_hex_string(value, "Scalar kernel argument value")
    else:
      raise JSONError("Unknown value " + str(key))

def __check_address_space(data):
  if data not in ["global", "local", "constant"]:
    raise JSONError("Address space must be 'global', 'local' or 'constant'")

def __check_array_argument(data):
  for key, value in data.items():
    if key == "type":
      pass
    elif key == "size":
      __check_positive_number(value, "Array kernel argument size")
    elif key == "flags":
      __check_array_of_strings(value, key)
    elif key == "data":
      __check_string(value, key)
    elif key == "address_space":
      __check_address_space(value)
    else:
      raise JSONError("Unknown value " + str(key))

def __check_image_argument(data):
  for key, value in data.items():
    if key == "type":
      pass
    else:
      raise JSONError("Unknown value " + str(key))

def __check_sampler_argument(data):
  for key, value in data.items():
    if key == "type":
      pass
    else:
      raise JSONError("Unknown value " + str(key))

def __check_argument(data):
  if not type(data) is dict:
    raise JSONError("kernel arguments need to be objects")
  if not "type" in data:
    raise JSONError("kernel arguments require a 'type' value")

  if data["type"] == "scalar":
    __check_scalar_argument(data)
  elif data["type"] == "array":
    __check_array_argument(data)
  elif data["type"] == "image":
    __check_image_argument(data)
  elif data["type"] == "sampler":
    __check_sampler_argument(data)
  else:
    raise JSONError("Unknown kernel argument type " + str(data["type"]))

def __check_kernel_arguments(data):
  if not type(data) is list:
    raise JSONError("kernel_arguments expects array")

  for i in data:
    __check_argument(i)

def __check_host_api_call(data):
  if not type(data) is dict:
    raise JSONError("api calls need to be objects")
  if not "function_name" in data:
    raise JSONError("api calls require a 'function_name' value")
  if not "compilation_unit" in data:
    raise JSONError("api calls require a 'compilation_unit' value")
  if not "line_number" in data:
    raise JSONError("api calls require a 'line_number' value")

  for key, value in data.items():
    if key in ["function_name", "compilation_unit"]:
      __check_string(value, key)
    elif key == "line_number":
      __check_positive_number(value, key)
    else:
      raise JSONError("Unknown value " + str(key))

def __check_host_api_calls(data):
  if not type(data) is list:
    raise JSONError("host_api_calls expects array")

  for i in data:
    __check_host_api_call(i)

def __check_endianness(data):
  if data not in ["little", "big"]:
    raiseJSONError("endinanness must be 'little' or 'big'")

DefinesIncludes = \
  namedtuple("DefinesIncludes", ["defines", "includes", "original"])

def __extract_defines_and_includes(compiler_flags):
  flags_split = compiler_flags.split()
  defines  = []
  includes = []

  i = 0
  while i < len(flags_split):
    if flags_split[i] == "-D":
      if i + 1 == len(flags_split):
        raise JSONError("compiler flag '-D' requires an argument")
      i += 1
      defines.append(flags_split[i])
    elif flags_split[i].startswith("-D"):
      defines.append(flags_split[i][len("-D"):])
    elif flags_split[i] == "-I":
      if i + 1 == len(flags_split):
        raise JSONError("compiler flag '-I' requires an argument")
      i += 1
      includes.append(flags_split[i])
    elif flags_split[i].startswith("-I"):
      includes.append(flags_split[i][len("-I"):])
    i += 1

  return DefinesIncludes(defines, includes, compiler_flags)

def __process_opencl_entry(data, strict):
  if strict and not "endianness" in data:
    raise JSONError("kernel invocation entries require an 'endianness' value")
  if not "kernel_file" in data:
    raise JSONError("kernel invocation entries require a 'kernel_file' value")
  if not "global_size" in data:
    raise JSONError("kernel invocation entries require a 'global_size' value")
  if not "entry_point" in data:
    raise JSONError("kernel invocation entries require an 'entry_point' value")

  for key, value in list(data.items()):
    if key in ["language", "kernel_file", "entry_point"]:
      __check_string(value, key)
    elif key == "endianness":
      __check_endianness(value)
    elif key in ["local_size", "global_size", "global_offset"]:
      __check_array_of_positive_numbers(value, key)
    elif key == "compiler_flags":
      __check_string(value, key)
      data[key] = __extract_defines_and_includes(value)
    elif key == "kernel_arguments":
      __check_kernel_arguments(value)
      data[key] = [__ldict(arg) for arg in value]
    elif key == "host_api_calls":
      __check_host_api_calls(value)
      data[key] = [__ldict(call) for call in value]
    else:
      raise JSONError("Unknown value " + str(key))

  # local_size is not necessarily specified; if so, use unconstrained values
  if not "local_size" in data:
    data["local_size"] = ["*" for _ in data["global_size"]]

  try:
    data["num_groups"] = get_num_groups(data["local_size"], data["global_size"])
  except GlobalSizeError as e:
    raise JSONError(str(e))

def __process_kernel_entry(data, strict):
  if not type(data) is dict:
    raise JSONError("kernel invocation entries need to be objects")
  if not "language" in data:
    raise JSONError("kernel invocation entries require a 'language' value")

  # Allow for future extension to CUDA
  if data["language"] == "OpenCL":
    __process_opencl_entry(data, strict)
  else:
    raise JSONError("'language' value needs to be 'OpenCL'")

def __filter_duplicates(data):
  new_data = []
  old_new_map = []

  for i in data:
    if i not in new_data:
      new_data.append(i)
    old_new_map.append(new_data.index(i))

  return new_data, old_new_map

def __process_json(data, strict):
  if not type(data) is list:
    raise JSONError("Expecting an array of kernel invocation objects")

  data, json_to_data_map = __filter_duplicates(data)

  for i in data:
    __process_kernel_entry(i, strict)

  return (data, json_to_data_map)

def json_load(json_file, strict = False):
  """Load GPUVerify invocation data from json_file object.

  The function either:
  (a) return a list of dictionaries structured as the objects in the JSON file
      and a list which maps from the orginal position in the JSON file to
      elements of the list with dictionaries, or
  (b) raises a JSONError in case an error is encountered.
  It is checked whether all required values are present and whether all values
  are of the right type.

  The function also:
  a) Extracts 'defines' and 'includes' from the compiler_flags value and
     returns a named tuple (defines, includes) instead of a string.
  b) Computes num_groups from global_size and local_size.

  The 'strict' argument is used to enforce extra-strong checking.  New tools
  that use the JSON format should use strict checking, but strict checking
  is off by default so that existing tools do not break.

  """
  try:
    data = json.load(json_file)
    data, json_to_data_map = __process_json(data, strict)
    return ([__ldict(kernel) for kernel in data], json_to_data_map)
  except ValueError as e:
    raise JSONError(str(e))
  except JSONError:
    raise
