#!/usr/bin/env python
""" vim: set sw=2 ts=2 softtabstop=2 expandtab:

  This utility is intended to be used with the
  KernelCounterObserver which records several 
  '*-counter.pickle' files (where * is a number),
  one for each process.

  This utility will read this pickle files and
  report the number of kernels processed by
  each process and in total

"""
import os
import re
import logging
import pickle
import argparse
import sys
import yaml


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  logging.basicConfig(level=logging.INFO)

  parser.add_argument('yaml_dir', help='The directory to search for *-counter.yml files.')

  args = parser.parse_args()
  counterDir = args.yaml_dir

  (root, _NOTUSED, files) = next(os.walk(counterDir))

  matcher = re.compile(r'^(\d+)-counter\.yml$')
  processNumber = [ ]
  pmap = { }
  for f in files:
    m = matcher.match(f)
    if m:
      logging.info('Found file {0}'.format(f))
      number = int(m.group(1))
      processNumber.append(number)
      pmap[number] = f

  if len(processNumber) == 0:
    logging.error('No *-counter.pickle files found')
    sys.exit(1)

  # Get counts
  runningTotal = 0
  counts = [ 0 for dummy in range(0, max(processNumber) +1) ]
  for index in range(0,max(processNumber) + 1):
    if not index in processNumber:
      logging.warning('No pickle file for process {0}. Assuming count of zero'.format(index))
      counts[index] = 0
    else:
      with open(os.path.join(counterDir,pmap[index]), 'r') as f:
        data = yaml.load(f)
        assert isinstance(data, dict)
        assert 'counter' in data
        value = data['counter']
        assert isinstance(value, int)
        assert value >= 0
        counts[index] = data['counter']
        runningTotal += counts[index]


  print('') # new line
  # print header
  print('## Item ## Number of kernels processed ##')
  # Print counts
  for (index, c) in enumerate(counts):
    print('Process {0}: {1}'.format(index, c))

  # Print total
  print('Total    : {0}'.format(runningTotal))

  sys.exit(0)
