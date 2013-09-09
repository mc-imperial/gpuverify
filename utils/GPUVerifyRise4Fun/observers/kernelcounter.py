import gvapi
import sys
import os
import logging
import pickle

_logging = logging.getLogger(__name__)

"""
  This class implements a simple observer that
  increments a counter when a kernel is processed.
  It attempts to save the results to file on destruction.
  
  FIXME: Concurrency does not work at all. Processes
         will race to write to the file.

"""
class KernelCounterObserver(gvapi.GPUVerifyObserver):
  counterFile = 'counter.pickle'

  def __init__(self):
    self.counter = 0
    
    # Try to load counter form pickle file
    if os.path.exists(self.counterFile):
      _logging.info('Counter pickle file exists. Trying to load counter from it')
      with open(self.counterFile, 'r') as f:
        try:
          self.counter = pickle.load(f)
          _logging.info('Loaded counter value of {} from pickle file {}'.format(self.counter, self.counterFile))
        except Exception as e:
          _logging.error('Failed to load counter from pickle file' + str(e))
          self.counter = 0
    else:
      _logging.info('Counter pickle file does not exist. Counter defaulting to {}'.format(self.counter))

  def __del__(self):
    # Try to save counter
    # The logger may not be available anymore so write to stderr
    sys.stderr.write('Attempting to save counter value of {} to file {}\n'.format(self.counter, self.counterFile))
    with open(self.counterFile, 'w') as f:
      try:
        pickle.dump(self.counter, f)
        f.flush()
      except Exception as e:
       sys.stderr.write('Failed to save counter to {}. Reason : {}\n'.format(self.counterFile, str(e)))
       pass

  def receive(self, source, args, returnCode, output):
    self.counter += 1
    _logging.info('Use count incremented to {}'.format(self.counter))

  def getCount(self):
    return self.counter
