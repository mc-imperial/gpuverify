"""Module defining some utility functions used in various places."""

def is_hex_string(string):
  """Check if a string defines a hex value"""
  if not string.startswith("0x"):
    return False

  try:
    int(string, 16)
    return True
  except ValueError as e:
    return False
