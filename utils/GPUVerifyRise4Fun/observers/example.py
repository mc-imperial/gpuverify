import gvapi
import logging

_logging = logging.getLogger(__name__)

# This is an example Observer. It doesn't really do anything interesting
class ExampleObserver(gvapi.GPUVerifyObserver):
  def receive(self, source, args, returnCode, output):
    _logging.info("Received source:\n" + source)
    _logging.info("Received command line args:" + str(args))
    _logging.info("Received return code " + str(returnCode) )
    _logging.info("Received Output" + output )
    
    if returnCode != gvapi.ErrorCodes.SUCCESS:
      _logging.info("Verification did not succeed!")
