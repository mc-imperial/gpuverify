"""
   This module provides a basic meta for the Rise4Fun tool.
   Some fields are expected to be modified.
"""
import copy
import re
import os
import gvapi
import logging
import config
import pprint

#Internal logger
_logging = logging.getLogger(__name__)

class BasicMetaData(object):

  # List of registered languages
  registeredLanguage = [ ]

  def __init__(self):

      #Try to generate a version number that Rise4Fun supports.
      #They want m.n[.p[.q]] where m,n,p and q are in the range 0 - (2^16 -1)

      #FIXME: This is really bad. The localID should not be used as
      # a global version identitifer.
      #
      # Ideally we need to do something clever to the changesetID (640-bits)
      # and reduce it to 64-bits (16 * 4) so we can get a version number
      # that Rise4Fun expects
      version, _NOT_USED = gvapi.GPUVerifyTool(config.GPUVERIFY_ROOT_DIR, 
                                               config.GPUVERIFY_TEMP_DIR).getVersionString()
      version +=".0"

      self.metadata = {
        "Name": "GPUVerify",
        "DisplayName": "GPUVerify",
        "DisableErrorTable": True, # Request to not show Visual Studio style error/warning table on web page
        "Version": version,
        "Email": "fixme@imperial.ac.uk",
        "SupportEmail": "fixme@imperial.ac.uk",
        "TermsOfUseUrl": "", # To be populated
        "PrivacyUrl": "", # To be populated
        "Institution": "Multicore programming Group, Imperial College London",
        "InstitutionUrl": "http://multicore.doc.ic.ac.uk",
        "InstitutionImageUrl": "" , #To be populated 
        "MimeType": "text/", #To be populated
        "SupportsLanguageSyntax": False,
        "Title": "A verifier for CUDA/OpenCL kernels",
        "Description": "This tool checks that a kernel is free from assertion failures, data races and barrier divergence." ,
        "Question": "", #To be populated
        "Url": "http://multicore.doc.ic.ac.uk/tools/GPUVerify/",
        "VideoUrl": "http://www.youtube.com/watch?v=l8ysBPV8OvA",
        "Samples": [], #To be populated
        "Tutorials": [] # To be populated
      }

      # Register a language (self.folderName should be set statically by a child class)
      BasicMetaData.registeredLanguage.append(self.folderName)

      self.languageSyntax = None


  def findSamplesAndTutorials(self,sourceRoot):
    """ This function populates self.metadata with
        the samples and tutorials contained within "sourceRoot".
    """
    # Load Samples from samples/
    kernelMatcher = re.compile(r'^.+\.' + self.getExtension() + r'$')
    for (root, dirs, files) in os.walk(os.path.join(sourceRoot, 'samples')):
      for kernelFile in [ k for k in files if kernelMatcher.match(k) ]:
        # Open the file and read contents to string
        kernelFilePath=os.path.join(root,kernelFile)
        with open(kernelFilePath,'r') as sourceCode:
          _logging.info('Loading sample "' + kernelFilePath + '"')
          data =  sourceCode.read()
          self.metadata['Samples'].append({'Name':kernelFile, 'Source': data})
    
    # Load Tutorials from tutorials/
    # We assume a directory structure where each tutorial is in a different folder
    # and the used sources are all the kernel files in the same folder as the tutorial
    markdownMatch = re.compile(r'^.+\.(markdown|md)')
    tutorialRoot = os.path.join(sourceRoot, 'tutorials')
    for (root, dirs, files) in os.walk(tutorialRoot):
      for markdownFile in [ m for m in files if markdownMatch.match(m) ]:
        # Read tutorial file
        tutorialFilePath=os.path.join(root,markdownFile)
        with open(tutorialFilePath,'r') as tutorialFile:
          _logging.info('Loading tutorial "' + tutorialFilePath + '"')
          data = tutorialFile.read()
          self.metadata['Tutorials'].append({'Name':markdownFile, 'Source': data})

        # Look for source files to go with the tutorial
        for kernelFile in [k for k in files if kernelMatcher.match(k) ]:
          #check for "Samples" dict key 
          if not 'Samples' in self.metadata['Tutorials'][-1]:
            #Add the Samples key
            self.metadata['Tutorials'][-1]['Samples']= []

          # Read Source file
          accompanyingKernelPath=os.path.join(root,kernelFile)
          with open(accompanyingKernelPath,'r') as accommpanyingSourceCode:
            _logging.info('Loading kernel "' + accompanyingKernelPath + '" for tutorial "' + tutorialFilePath + '"')
            data = accommpanyingSourceCode.read()
            self.metadata['Tutorials'][-1]['Samples'].append({'Name':kernelFile, 'Source':data})
  
  def loadLanguageSyntax(self, module):
    if not getattr(module,'syntax'):
      _logging.warning('Failed to load language syntax definition from "' + module.__file__ + '"')
      self.metadata['SupportsLanguageSyntax'] = False
      self.languageSyntax = None
    else:
      self.metadata['SupportsLanguageSyntax'] = True
      self.languageSyntax = module.syntax
      _logging.info('Loaded language syntax definition from "' + module.__file__ + '"')

class OpenCLMetaData(BasicMetaData):
  def getExtension(self):
    return 'cl'

  folderName='opencl'

  def __init__(self, sourceRoot):
    import opencl.syntax
    super(self.__class__, self).__init__()

    # FIXME: use 'x-opencl' and provide own syntax definition
    self.metadata['MimeType'] += 'x-c' #HACK : Use 'C' language definition implicitly

    self.metadata['Question'] = 'Is this OpenCL kernel correct?'

    self.findSamplesAndTutorials(os.path.join(sourceRoot, self.folderName))
    self.loadLanguageSyntax(opencl.syntax)
    _logging.debug("Generated OpenCL metadata:\n" + pprint.pformat(self.metadata))


class CUDAMetaData(BasicMetaData):
  def getExtension(self):
    return 'cu'

  folderName='cuda'

  def __init__(self,sourceRoot):
    import cuda.syntax
    super(self.__class__, self).__init__()

    # FIXME: use 'x-cuda' and provide own syntax definition
    self.metadata['MimeType'] += 'x-c' #HACK: Use 'C' language definition implicitly

    self.metadata['Question'] = 'Is this CUDA kernel correct?'

    self.findSamplesAndTutorials(os.path.join(sourceRoot, self.folderName))
    self.loadLanguageSyntax(cuda.syntax)
    _logging.debug("Generated CUDA metadata:\n" + pprint.pformat(self.metadata))

