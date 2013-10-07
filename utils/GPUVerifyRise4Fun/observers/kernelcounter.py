import gvapi
import sys
import os
import logging
import pickle
import psutil

_logging = logging.getLogger(__name__)

"""
  This class implements a simple observer that
  increments a counter when a kernel is processed.
  It attempts to save the results to file on every
  execution.

  The class assumes that there will be multiple instances
  running in different processes with the same parent 
  (i.e. forking behaviour of tornado). Each instance
  will get a unique file name to log its counter variable.

  The utility print-counters.py can be easily used to read
  these counters to check the current count.
"""
class KernelCounterObserver(gvapi.GPUVerifyObserver):
  counterFile = '-counter.pickle' # Will later be ammended

  def __init__(self, dirForPickles):
    self.counter = 0
    self.pickleDir = os.path.abspath(dirForPickles)
    self.loadCounter()

    if not os.path.exists(self.pickleDir):
      errorMsg = 'Pickle directory "{0}" does not exist!'.format(self.pickleDir)
      _logging.error(errorMsg)
      raise Exception(errorMsg)

  def loadCounter(self):
    _logging.debug('Performing loadCounter()')

    # It is assumed that multiple instance of this class are each in their
    # own process spawned by Tornado. We need to associate a number with
    # each process that is preserved across executions (so cannot use PID).
    # Instead we generate a list of all child PIDs of our parent (these should
    # all be instances of this web app) order them and use the index of our
    # PID in that list as the number associated with this process. Provided
    # the number of spawned processes does not vary then this will work.

    parentProcess = psutil.Process(os.getppid())
    pidList = [ ]
    for process in parentProcess.get_children(recursive=False):
      pidList.append(process.pid)

    _logging.debug('Found processes ' + str(pidList))
    assert len(pidList) > 0
    pidList.sort()

    # Now get our process number
    prefix = ''
    for (index, pid) in enumerate(pidList):
      if pid == os.getpid():
        prefix = str(index)

    assert prefix != ''
    self.counterFile = prefix + self.counterFile
    self.counterFile = os.path.join(self.pickleDir, self.counterFile)
    _logging.info('Trying to load counter from {0}'.format(self.counterFile))
      
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

  def saveCounter(self):
    # Try to save counter
    #
    # Note we will need to replace logging calls with sys.stderr.write() calls
    # if we call this method from __del__() as logger may have been destroyed already.
    _logging.info('Attempting to save counter value of {} to file {}\n'.format(self.counter, self.counterFile))
    with open(self.counterFile, 'w') as f:
      try:
        pickle.dump(self.counter, f)
        f.flush()
      except Exception as e:
       _logging.error('Failed to save counter to {}. Reason : {}\n'.format(self.counterFile, str(e)))

  def __del__(self):
    # We may want to call saveCounter() here instead
    # if calling in receive() causes performance problems
    pass

  def receive(self, source, args, returnCode, output):
    self.counter += 1
    _logging.info('Use count incremented to {}'.format(self.counter))
    self.saveCounter()

  def getCount(self):
    return self.counter
