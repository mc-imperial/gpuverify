#!/usr/bin/env python
"""Utility for checking the validity of JSON files accepted by GPUVerify."""
from __future__ import print_function

import json
import sys

# Relative imports not allowed in top-level Python scripts
sys.path.append("..")

from GPUVerifyScript.json_loader import JSONError, json_load

if __name__ == '__main__':
  if len(sys.argv) != 2 or sys.argv[1] in ["-h", "--help"]:
    print("Usage: {} <json_file>".format(sys.argv[0]))
    sys.exit(1)

  try:
    with open(sys.argv[1]) as f:
      json_load(f)
    print("JSON file succesfully read")
    sys.exit(0)
  except IOError as e:
    print(e)
    sys.exit(1)
  except JSONError as e:
    raise
    print(e)
    sys.exit(1)
