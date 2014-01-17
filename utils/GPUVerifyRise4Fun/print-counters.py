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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  logging.basicConfig(level=logging.INFO)

  parser.add_argument('--pickle_dir','-p', help='The directory to search for *-counter.pickle files. Default is current directory', default=os.getcwd())

  args = parser.parse_args()
  pickleDir = args.pickle_dir

  (root, _NOTUSED, files) = next(os.walk(pickleDir))

  matcher = re.compile(r'^(\d+)-counter\.pickle$')
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
      with open(os.path.join(pickleDir,pmap[index]), 'r') as pickleFile:
        counts[index] = pickle.load(pickleFile)
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
