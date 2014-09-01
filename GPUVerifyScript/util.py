"""Module defining some utility functions used in various places."""

def is_hex_string(string):
  """Check if a string represents a hex value"""
  if not string.startswith("0x"):
    return False

  try:
    int(string, 16)
    return True
  except ValueError as e:
    return False

def is_positive_string(string):
  """Check if a string defines a positive integer value"""
  try:
    val = int(string)
    return val > 0
  except ValueError as e:
    return False

class GlobalSizeError(Exception):
  """Exception type returned by get_num_groups."""
  def __init__(self, msg):
    self.msg = msg

  def __str__(self):
    return self.msg

def get_num_groups(group_size, global_size):
  if len(group_size) != len(global_size):
    raise GlobalSizeError("Dimensions of local and global size must match")

  num_groups = []

  for i in range(len(group_size)):
    if global_size[i] == "*":
      num_groups.append("*")
    elif group_size[i] == "*":
      num_groups.append(("*", global_size[i]))
    elif (global_size[i]//group_size[i]) * group_size[i] == global_size[i]:
      # Use '//' to ensure flooring division for Python3
      num_groups.append(global_size[i]//group_size[i])
    else:
      raise GlobalSizeError("Dimension " + str(i) + " of global size does " +
        "not divide by dimension " + str(i) + " of local size")

  return num_groups
