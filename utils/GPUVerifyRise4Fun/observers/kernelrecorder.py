# vim: set sw=2 ts=2 softtabstop=2 expandtab:
import gvapi
import logging
import os
import stat
import tempfile

_logging = logging.getLogger(__name__)

# This observer records GPUVerify run information
class KernelRecorderObserver(gvapi.GPUVerifyObserver):
  def __init__(self, dirForKernels):
    self.kernelDir = os.path.abspath(dirForKernels)

  def receive(self, source, args, returnCode, output):
    _logging.info('Logging kernel')
    
    tFile = None
    with tempfile.NamedTemporaryFile(delete=False, dir=self.kernelDir, \
                                     suffix='.txt', prefix='kernel-') as f:
      tFile = os.path.abspath(f.name)
      _logging.debug('Logging to {}'.format(tFile))
      f.write("=== Command Line Arguments ===\n\n" + str(args)       + "\n\n")
      f.write("======= Kernel Source ========\n\n" + source          + "\n\n")
      f.write("======== Return Code =========\n\n" + str(returnCode) + "\n\n")
      f.write("=========== Output ===========\n\n" + output)
      f.close()

    _logging.debug('Logging kernel done')

    if os.name == 'posix':
      # Tempfile makes files only readable by current user
      # Change permissions so readable by group as well.
      os.chmod(tFile, stat.S_IRUSR | stat.S_IRGRP)
