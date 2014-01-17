""" vim: set sw=2 ts=2 softtabstop=2 expandtab:
This module provides a simple
    API to GPUVerify
"""
import config
import sys
import os
import subprocess
import tempfile
import shutil
import re
import logging
import traceback

#Internal logger
_logging = logging.getLogger(__name__)

# Put GPUVerify.py module in search path
sys.path.insert(0, config.GPUVERIFY_ROOT_DIR)
from GPUVerify import ErrorCodes

# Error code to message map
helpMessage = {
ErrorCodes.SUCCESS:"",
ErrorCodes.COMMAND_LINE_ERROR:"Error processing command line.",
ErrorCodes.CLANG_ERROR:"Clang could not compile your kernel to LLVM bitcode.",
ErrorCodes.OPT_ERROR:"Could not perform necessary optimisations to your kernel.",
ErrorCodes.BUGLE_ERROR:"Could not translate LLVM bitcode to Boogie.",
ErrorCodes.GPUVERIFYVCGEN_ERROR:"Could not generate invariants and/or perform two-thread abstraction.",
ErrorCodes.BOOGIE_ERROR:"",
ErrorCodes.TIMEOUT:"Verification timed out.",
ErrorCodes.CONFIGURATION_ERROR:"The web service has been incorrectly configured. Please report this issue to gpuverify-support@googlegroups.com"
}

# Observer design pattern
class GPUVerifyObserver(object):
  """
      Receive a notification of a completed GPUVerify command.
      source     : The input source code as a string
      args       : List of command line options
      returnCode : The return code given by GPUVerify
      output     : The output of the GPUVerify Tool
  """
  def receive(self, source, args, returnCode, output):
    pass

class GPUVerifyTool(object):
  """
      rootPath : Is the root directory of the GPUVerify tool ( development or deploy)
      tempDir  : Is the directory to use for temporary files. If None set then use system default.
  """
  def __init__(self, rootPath, tempDir=None):
    rootPath = os.path.abspath(rootPath)
    if tempDir:
      self.tempDir = os.path.abspath(tempDir)
      if not os.path.exists(tempDir):
        raise Exception('Path to temporary directory must exist')
    else:
      self.tempDir = None

    if not os.path.exists(rootPath):
      raise Exception('Path to GPUVerify root must exist')

    self.toolPath = os.path.join(rootPath,'GPUVerify.py')
    if not os.path.exists(self.toolPath):
      raise Exception('Could not find GPUVerify at "' + self.toolPath + '"')

    self.observers = [ ]
    
  def registerObserver(self, observer):
    """
        Register an observer (of type GPUVerifyObserver) that will receive notifications
        when the runCUDA() or runOpenCL() methods are executed.

    """
    if not isinstance(observer, GPUVerifyObserver):
      raise Exception("Invalid observer")
    else:
      self.observers.append(observer)

  def extractOtherCmdArgs(self, source, args, ignoredArgs=None, additionalArgs=None):
    """
      Extract command line arguments from the first line of the source code
      that are allowed (i.e. some filtering is done) to be passed to runOpencl() 
      or runCUDA() as extraCmdLineArgs

      source : The source code as a string where arguments will be extracted from.
      args   : A list that this function will populate
      ignoredArgs : If not None this list will be populated with arguments that
                    were ignored (except the NDRange arguments like --blockDim).
      additionalArgs : If not None this list be used as additional command line
                       arguments. Just like the arguments in the source code these 
                       will be filtered.
    """
    if len(args) != 0:
      raise Exception("Argument list must be empty")

    firstLine=source.splitlines()[0]

    if not firstLine.startswith('//'):
      raise Exception('First line of source must have // style comment')

    foundArgs=firstLine.split()[1:] #Split on spaces, removing the comment

    if additionalArgs:
        foundArgs.extend(additionalArgs)

    #A whitelist of allowed options (NDRange or Grid Size args are deliberatly not here)
    safeOptions=['--adversarial-abstraction',
                 '--array-equalities',
                 '--asymmetric-asserts',
                 r'--atomic=(r|rw|none)',
                 '--debug',
                 #'--dynamic-analysis', # Note sure if safe, disable for now
                 '--equality-abstraction',
                 '--findbugs',
                 r'--loop-unwind=\d+',
                 '--math-int',
                 '--no-annotations',
                 '--no-barrier-access-checks',
                 '--no-benign',
                 '--no-constant-write-checks',
                 '--no-infer',
                 '--no-loop-predicate-invariants',
                 '--no-refinded-atomics',
                 '--no-smart-predication',
                 '--no-source-loc-infer',
                 '--no-uniformity-analysis',
                 '--only-divergence',
                 '--only-intra-group',
                 '--only-requires',
                 '--parallel-inference',
                 '--time',
                 '--staged-inference',
                 # r'--scheduling=[a-z-]+', # Not sure if safe
                 '--verify',
                 '--verbose',
                 r'--warp-sync=\d{1,3}']

    for arg in foundArgs:
      matcher=None
      for option in safeOptions:
        matcher=re.match(option + r'\b',arg)
        if matcher:
          args.append(matcher.group(0))
          _logging.debug('Accepting command line option "' + args[-1] + '"')
          break
      # Warn about ignored args except the gridDim types as they are handled else where.
      if matcher == None and not [ x for x in 
                                   [ '--blockDim=', 
                                     '--gridDim=', 
                                     '--local_size=', 
                                     '--num_groups='
                                   ] if arg.startswith(x) 
                                 ]: 
        _logging.warning('Ignoring passed command line option "' + arg + '"')
        if ignoredArgs != None:
          ignoredArgs.append(arg)
      


  def extractNDRangeFromSource(self, source, localSize, numGroups):
    """ Given OpenCL kernel source code extract the local_size and
        num_groups parameters from the comment on the first line of the source
        code and place into the respectively passed arrays.

        Example args:
        // --local_size=64
        // --local_size=[128,128]
        // --local_size=[128,128,128]
    """
    if len(localSize) != 0 or len(numGroups) != 0:
      raise Exception("localSize and numGroups arrays must be empty")
    
    self. __extractGridSizeCommon(source, [ ('--local_size=',localSize), ('--num_groups=', numGroups) ] )

  def extractGridSizeFromSource(self, source, blockDim, gridDim):
    """ Given the CUDA kernel source code extract the blockDim and
        gridDim parameters from the comment on the first line of the source
        code and place into the respectively passed arrays.

        Example args:
        // --blockDim=64
        // --blockDim=[128,128]
        // --blockDim=[128,128,128]
    """
    if len(blockDim) != 0 or len(gridDim) != 0:
      raise Exception("blockDim and gridDim arrays must be empty")

    self. __extractGridSizeCommon(source, [ ('--blockDim=',blockDim), ('--gridDim=', gridDim) ] )

  def __extractGridSizeCommon(self, source, gst):
    """ Common functionality for extracting grid dimension command line arguments
        from first line of source code.
        
        source : Source code as string
        gst : GridSize tuple. Example
              [ ('--local_size=', localSize), ('--num_groups=',numGroups) ]:

              where localSize and numGroups are empty lists to be populated.

    """
  
    # \1, \2, \3 extracts first, second and third number
    # or \4 \5 \6 (inserted later in regex)
    digitRegex=r'(\d+)(?:,(?:(\d+)(?:,(\d+))?))?'

    firstLine = source.splitlines()[0]
    _logging.debug('First line "' + firstLine + '"')


    #Loop throught the arguments to collect
    for (cmdArg,argList) in gst:
      matcher = re.match(r'//.*' + cmdArg + r'(?:' + digitRegex + r'|\[' 
                           + digitRegex + r'])(?: .*)?$', firstLine)

      if not matcher:
        raise Exception('Could not extract ' + cmdArg + ' argument from source')

      # Build list of ints
      for number in matcher.groups():
        if number != None:
          argList.append(int(number))

      if len(argList) == 0 or len(argList) > 3 :
        raise Exception('Generated list for ' + cmdArg + ' was of incorrect size (' + len(argList) + ')')


  def runOpenCL(self, source, localSize, numOfGroups, timeout=10, extraCmdLineArgs=None):
   return self.__runCommon( source,
                            [ (localSize,'localSize','--local_size='),
                              (numOfGroups,'numOfGroups','--num_groups=')
                            ],
                            '.cl',
                            timeout,
                            extraCmdLineArgs
                          )

  def __runCommon(self, source, gst, fileExtension, timeout, extraCmdLineArgs):
    """
        This function will excute GPUVerify on source code. This function
        exists because there is a lot of common functionality between checking
        an OpenCL kernel and a CUDA kernel.

        source : The program source code to be checked as a string
        gst    : Grid Size tuples. An array containing two tupples
                 ( blockDimVariable, blockDim-String, blockDim-CommandLine-option )
                 ( gridDimVariable, gridDim-String, gridDim-CommandLine-option )
        fileExtension : 'cl' or 'cu'
        timeout : An integer timeout (0 is no timeout)
        extraCmdLineArgs : A list of command line options
    """
    # Perform sanity check of NDRange/Grid
    for (var,name,__NOT_USED) in gst:
      if type(var) != list:
        raise Exception( name +  ' must be an array')

      if len(var) > 3 or len(var) < 1:
        raise Exception(name + ' must be 1D, 2D or 3D')

      for dim in var:
        if dim < 1:
          raise Exception(name + ' must have positive dimensions')

    # Perform sanity check of timeout
    if timeout < 0:
      raise Exception('timeout must be positive or zero')
    
    # Generate string representation of localSize (blockDim) and numOfGroups (GridDim)
    localSize=str(gst[0][0]).replace(' ','')
    numOfGroups=str(gst[1][0]).replace(' ','')

    cmdArgs = [ gst[0][2] + localSize, gst[1][2] + numOfGroups, '--timeout=' + str(timeout) ]
    if extraCmdLineArgs != None:
      cmdArgs += extraCmdLineArgs


    # Create source file inside self.tempDir
    f = tempfile.NamedTemporaryFile(prefix='gpuverify-source-', 
                                    suffix=fileExtension, 
                                    delete=False,
                                    dir=self.tempDir)
    response=None
    try:
      f.write(source.encode('utf8'))
      f.close()
      
      # Add sourcefile name to cmdArgs
      cmdArgs.append(f.name)

      response = self.__runTool(cmdArgs)
      if response[0] === ErrorCodes.TIMEOUT:
        _logging.error('GPUVerify timed out (ErrorCode:{})'.format(response[0])) 

    finally:
      f.close()
      os.remove(f.name)
    
    # Invoke any observers on the outcome of running the command
    for observer in self.observers:
      # Do not allow observers to cause their exceptions to cause the complete execution to fail.
      try:
        _logging.debug("Executing Observer " + str(observer.__class__))
        observer.receive( source, cmdArgs, response[0], response[1])
      except Exception as e:
        _logging.error("Observer " + str(observer.__class__) + " raised exception " + str(e) + '\n' + traceback.format_exc()  )

    return response


  def runCUDA(self, source, blockDim, gridDim, timeout=10, extraCmdLineArgs=None):
    return self.__runCommon( source,
                             [
                               (blockDim,'blockDim','--blockDim='),
                               (gridDim,'gridDim','--gridDim=')
                             ],
                             '.cu',
                             timeout,
                             extraCmdLineArgs
                           )




  def getVersionString(self):
    ( returnCode, versionString ) = self.__runTool(['--version'])
    if returnCode == 0:

      # Parse version string
      matcher = re.search(r'local-revision\s+:\s+(\d+)',versionString)
      if not matcher:
        raise Exception('Could not parse local-revision string from "' + versionString + '"')
      localID=matcher.group(1)

      matcher = re.search(r'vcgen\s+:\s+(\d+)',versionString)
      if not matcher:
        raise Exception('Could not parse vcgen string from "' + versionString + '"')
      changesetID=matcher.group(1)

      return (localID, changesetID)

    else:
      raise Exception('Could not get version')
    

  def __runTool(self, cmdLineArgs):
    # Make temporary working directory inside self.tempDir
    tempDir = tempfile.mkdtemp(prefix='gpuverify-working-directory-temp',dir=self.tempDir)
    
    returnCode = 0
    message=""
    try: 
      cmdArgs = [ sys.executable, self.toolPath ] + cmdLineArgs
      _logging.debug('Running :' + str(cmdArgs))
      process = subprocess.Popen( cmdArgs,
                                    stdout = subprocess.PIPE,
                                    stderr = subprocess.STDOUT,
                                    cwd = tempDir,
                                    preexec_fn=os.setsid) # Make Sure GPUVerify can't kill us!

      message, _NOT_USED = process.communicate() # Run tool
      returnCode = process.returncode
    except OSError as e:
      returnCode=-1 
      message = 'Internal error. Could not run "' + self.toolPath + '"'
    finally:
      # Remove the tempDir
      shutil.rmtree(tempDir)
    
    return ( returnCode, message )

