#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import argparse
import logging
import re
import subprocess
import pickle 

from GPUVerify import ErrorCodes

GPUVerifyExecutable=sys.path[0] + os.sep + "GPUVerify.py"

class GPUVerifyErrorCodes(ErrorCodes):
    """ Provides extra error codes and a handy dictionary
        to map error codes to strings.
        
        The error codes used in this class represent the 
        potential exit codes of GPUVerify. We add 
        REGEX_MISMATCH_ERROR as that is a possible point of failure
        during testing
    """
    REGEX_MISMATCH_ERROR=-1 #Overwritten by static_init()

    errorCodeToString = {} #Overwritten by static_init()
    @classmethod
    def static_init(cls):
        base=cls.__bases__[0] #Get our parent class
        
        #Assign correct error code number to REGEX_MISMATCH_ERROR
        codes=[e for e in dir(base) if not e.startswith('_') ]
        
        #find the largest code
        largest=getattr(base,codes[0])
        for num in [ getattr(base,x) for x in codes ]:
            if num > largest : largest=num
        
        #We'll take the next error codes
        cls.REGEX_MISMATCH_ERROR=largest +1
        
        #Build reverse mapping dictionary { num:string }
        codes=[e for e in dir(cls) if not e.startswith('_')]
        for (num,string) in [( getattr(cls,x), x)  for x in codes if type(getattr(cls,x)) == int]:
            cls.errorCodeToString[num]=string
            
    @classmethod
    def getValidxfailCodes(cls):
        codes=[]
        for codeTuple in cls.errorCodeToString.iteritems():
            #Skip SUCCESS and REGEX_MISMATCH_ERROR as it isn't sensible to expect a failure
            # at these points.
            if codeTuple[0] == cls.SUCCESS or codeTuple[0] == cls.REGEX_MISMATCH_ERROR :
                continue
            else:
                codes.append(codeTuple)
        
        return codes

            
#Perform the initialisation
GPUVerifyErrorCodes.static_init()

def enum(*sequential):
    #Build a dictionary that maps sequential[i] => i
    enums = dict( zip(sequential, range(len(sequential))) )
    #Build a reverse dictionary
    reverse = dict((value, key) for key, value in enums.iteritems())
    enums['reverseMapping'] = reverse
    return type('Enum', (object,) , enums)

GPUVerifyTesterErrorCodes=enum('SUCCESS', 'FILE_SEARCH_ERROR','KERNEL_PARSE_ERROR', 'TEST_FAILED', 'FILE_OPEN_ERROR', 'GENERAL_ERROR')

class GPUVerifyTestKernel:
    
    def __init__(self,path):
        """
            Initialise. Upon successful parsing of a kernel the follow attributes should be available.
            .expectedReturnCode : The expected return code of GPUVerify
            .gpuverifyCmdArgs   : A list of command line arguments to pass to GPUVerify
            .regex              : A dictionary of regular expressions that map to Success True/False
        """
        logging.debug("Parsing kernel \"{0}\" for test parameters".format(path))
        self.path=path
        
        #Use with so that if exception throw file is still closed
        #Note need to use universal line endings to handle DOS format (groan) kernels
        with open(self.path,'rU') as fileObject:
            
            #Grab expected test outcome
            expectedOutcome=fileObject.readline()
            regexOutcome=re.compile(r'^//(pass|xfail:([A-Z_]+))',re.IGNORECASE)
            matched=regexOutcome.match(expectedOutcome)
            
            #Note that we store the expectedReturnCode in integer form (not the string form)
            if matched == None:
                raise KernelParseError(1,self.path,"First Line should say \"//pass\" or \"//xfail:<ERROR_CODE>\"")
            else:
                if(matched.group(1).lower() == "pass"): 
                    self.expectedReturnCode=GPUVerifyErrorCodes.SUCCESS
                else:
                    xfailCodeAsString=matched.group(2).upper()
                    if xfailCodeAsString in [ t[1] for t in GPUVerifyErrorCodes.getValidxfailCodes() ]:
                        self.expectedReturnCode=getattr(GPUVerifyErrorCodes,xfailCodeAsString)
                    else:
                        raise KernelParseError(1,self.path, "\"" + xfailCodeAsString + "\" is not a valid error code for expected fail.")
                        
                    
            #Grab command line args to pass to GPUVerify
            cmdArgs=fileObject.readline()
            #Slightly naive test
            if not cmdArgs.startswith('//'):
                raise KernelParseError(2,self.path,"Second Line should start with \"//\" and then optionally space seperate arguments to pass to GPUVerify")
            
            self.gpuverifyCmdArgs = cmdArgs[2:].strip().split() #Split on spaces
            
            #Grab (optional regex line(s))
            haveRegexLines=True
            self.regex={}
            lineCounter=3
            while haveRegexLines:
                line=fileObject.readline()
                line=line[:-1] # Strip off \n
                if line.startswith('//') and len(line) > 2:
                    key=line[2:] #Strip //
                    self.regex[key]=None #Use key and do not assign truth value
                    try:
                        re.compile(key)
                    except re.error as e:
                        raise KernelParseError(lineCounter,self.path,"Invalid Regex (" + str(e.__class__) + " : " + str(e) + ")")
                else:
                    haveRegexLines=False; #Break out of loop
                
                lineCounter+=1
            
            #Finished parsing
            logging.debug("Successfully parsed kernel \"{0}\" for test parameters".format(path))
            
     
    def run(self):
        """ Executes GPUVerify on this test's kernel
            using this instance's parameters. This will set the following additional attributes on completion
            .testPassed : Boolean
            .returnedCode : The return code of the test (includes REGEX_MISMATCH_ERROR)
            .gpuverifyReturnCode : GPUVerify's actual return code (doesn't include REGEX_MISMATCH_ERROR)
        """    
        cmdLine=[sys.executable, GPUVerifyExecutable] + self.gpuverifyCmdArgs + [self.path]
        try:
            logging.info("Running test " + self.path)
            logging.debug(self)
            processInstance=subprocess.Popen(cmdLine,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout=processInstance.communicate() #Allow program to run and wait for it to exit.    
            
        except KeyboardInterrupt:
            logging.error("Received keyboard interrupt. Attempting to kill GPUVerify process")
            processInstance.kill()
            raise
        
        
        #Record the true return code of GPUVerify
        self.gpuverifyReturnCode=processInstance.returncode
        
        #Do Regex tests if the rest of the test went okay
        if self.gpuverifyReturnCode == self.expectedReturnCode:
            for regexToMatch in self.regex.keys():
                matcher=re.compile(regexToMatch, re.MULTILINE) #Allow ^ to match the beginning of multiple lines
                if matcher.search(stdout[0]) == None :
                    self.regex[regexToMatch]=False
                    logging.error(self.path + ": Regex \"" + regexToMatch + "\" failed to match output!")
                else:
                    self.regex[regexToMatch]=True
                    logging.debug(self.path + ": Regex \"" + regexToMatch + "\" matched output.")
        
        #Record the test return code.
        if False in self.regex.values():
            self.returnedCode=GPUVerifyErrorCodes.REGEX_MISMATCH_ERROR
        else:
            self.returnedCode=processInstance.returncode
        
        
        #Check if the test failed overall
        if self.returnedCode != self.expectedReturnCode :
            self.testPassed=False
            logging.info(self.path + " FAILED with " + GPUVerifyErrorCodes.errorCodeToString[self.returnedCode] + 
                         " expected " + GPUVerifyErrorCodes.errorCodeToString[self.expectedReturnCode])
            
            #Print output for user to see
            for line in stdout[0].split('\n'):
                print(line)
        else:
            self.testPassed=True
            logging.info(self.path + " PASSED (" + 
                         ("pass" if self.expectedReturnCode == GPUVerifyErrorCodes.SUCCESS else "xfail") + ")")
            
    
    def __str__(self):
        return """GPUVerifyTestKernel:
  File:{0}
  Expected exit code:{1}
  CmdArgs: {2}
  Regex: {3}
""".format(   self.path, 
              GPUVerifyErrorCodes.errorCodeToString[self.expectedReturnCode], 
              self.gpuverifyCmdArgs, 
              self.regex
          )
        



class GPUVerifyTesterError(Exception):
    pass

class KernelParseError(GPUVerifyTesterError):

    def __init__(self, lineNumber, fileName, message):
        self.lineNumber=lineNumber
        self.fileName=fileName
        self.message=message

    def __str__(self):
        return "KernelParseError : Line " + str(self.lineNumber) + " in \"" + self.fileName + "\": " + self.message

class CanonicalisationError(GPUVerifyTesterError):
    
    def __init__(self,path,prefix):
        self.path=path
        self.prefix=prefix
        
    def __str__(self):
        return "CanonicalisationError : Cannot construct a canonical path from \"" + self.path + "\" using prefix \"" + self.prefix + "\""

#This Action will be triggered from the command line
class PrintXfailCodes(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
            for errorCode in GPUVerifyErrorCodes.getValidxfailCodes():
                print(errorCode[1])
                
            sys.exit(GPUVerifyTesterErrorCodes.SUCCESS)

def openPickle(path):
    try:
        with open(path,"rb") as inputFile:
            return pickle.load(inputFile)
    except IOError:
        logging.error("Failed to open pickle file \"" + path + "\"")
        sys.exit(GPUVerifyTesterErrorCodes.FILE_OPEN_ERROR)
    except pickle.UnpicklingError:
        logging.error("Failed to parse pickle file \" + path + \"")
        sys.exit(GPUVerifyTesterErrorCodes.FILE_OPEN_ERROR)

def dumpTestResults(tests,prefix):
    print("Printing results of " + str(len(tests)) + " tests")
    width=80
    for testObj in tests:
        print("\n" + ("#" * width)) #Header bar
        try:
            print("Test:" + getCanonicalTestName(testObj.path, prefix))
        except CanonicalisationError as e:
            logging.error(e)
            print("Test: No canonical name")
            
        print("Full Path: " + testObj.path)
        print("Passed: " + str(testObj.testPassed))
        print("Expected result:" + GPUVerifyErrorCodes.errorCodeToString[testObj.expectedReturnCode])
        print("Actual result:" + GPUVerifyErrorCodes.errorCodeToString[testObj.returnedCode])
        print("GPUVerify return code:" + GPUVerifyErrorCodes.errorCodeToString[testObj.gpuverifyReturnCode])
        if len(testObj.regex) > 0:
            print("\nRegular expression matching:")
            for (regex,succeeded) in testObj.regex.items():
                print("\"" + regex +"\" : " + ("MATCHED" if succeeded else "FAILED TO MATCH"))
        print("\n")
        
        print("#" * width) #Footer bar

#This Action can be triggered from the command line
class dumpTestResultsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        logging.basicConfig(level=getattr(logging, namespace.log_level.upper(), None)) # enable logging
        dumpTestResults(openPickle(values),namespace.canonical_path_prefix)
        sys.exit(GPUVerifyTesterErrorCodes.SUCCESS)
        
#This Action can be triggered from the command line
class comparePickleFiles(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        logging.basicConfig(level=getattr(logging, namespace.log_level.upper(), None)) # enable logging
        doComparison(openPickle(values[0]),values[0],openPickle(values[1]),values[1],namespace.canonical_path_prefix)
        sys.exit(GPUVerifyTesterErrorCodes.SUCCESS)

def getCanonicalTestName(path,prefix):
    """
        This function takes a path and tries to generate a canonical path
        so that it is possible to compare test runs between different machines
        that keep their tests in different directories.
        
        The "prefix" is assumed to be present in every path to a test kernel.
    """
    cPath=""
    try:
        #Try to grab everything from the prefix onwards
        cPath=path[path.index(prefix):]
        
        #Replace Windows slashes with UNIX slashes
        cPath=cPath.replace('\\', '/')
    except ValueError:
        raise CanonicalisationError(path,prefix)
    
    return cPath
  
def doComparison(oldTestList,oldTestName,newTestList,newTestName, canonicalPathPrefix):
    #Perform Comparison

    logging.info("Performing comparison of \"" + newTestName +  "\" run and the run recorded in \"" + oldTestName + "\"")
    changedTestCounter=0
    missingTestCounter=0
    newTestCounter=0
    
    #Create dictionaries mapping canonical path to test
    #We do this now because we want to leave determining canonical path
    #to comparison time so the user has the capability to manipulate how the
    #canonical path is determined
    
    oldTestDic={}
    newTestDic={}
    cPath=""
    for (testDictionary, testList) in [ (oldTestDic, oldTestList),(newTestDic,newTestList) ]:
        for test in testList:
            try:
                cPath=getCanonicalTestName(test.path, canonicalPathPrefix)
            except CanonicalisationError as e:
                logging.error(e)
                cPath=test.path
            
            testDictionary[cPath]=test #Add entry
            
    logging.info("\"" + oldTestName + "\" has " + str(len(oldTestDic)) + " test(s)")
    logging.info("\"" + newTestName + "\" has " + str(len(newTestDic)) + " test(s)")
    
    #Iterate over old tests
    for (cPath, oldTest) in oldTestDic.items():
        if cPath in newTestDic:
            #Found a test common to both test sets
            
            #Look for a change in result
            if oldTest.testPassed != newTestDic[cPath].testPassed:
                logging.info("Test \"" + cPath + "\" result has changed.")
                print("Test \"" + cPath + "\ from \"" + oldTestName + "\":")
                print(oldTest)
                print("Test \"" + cPath + "\ from \"" + newTestName + "\"")
                print(newTestDic[cPath])
                changedTestCounter+=1
        else:
            logging.warning("Test \"" + cPath + "\" present in \"" + oldTestName+ "\" was missing from \"" + newTestName + "\"")
            missingTestCounter+=1
    
    #Iterate over just completed tests to notify about newly introduced tests.
    for cPath in newTestDic.keys():
        if cPath not in oldTestDic:
            logging.info("Test \"" + cPath + "\" was executed in \"" + newTestName + "\" but was not present in \"" + oldTestName + "\"" )
            newTestCounter+=1

    #Print test summary
    logging.info("Summary:")
    logging.info("# of changed tests:" + str(changedTestCounter))
    logging.info("# of missing tests:" + str(missingTestCounter))
    logging.info("The above is number of tests in \"" + oldTestName + "\" that aren't present in \"" + newTestName + "\"")
    logging.info("# of new tests:" + str(newTestCounter))
    logging.info("The above is number of tests in \"" + newTestName + "\" that aren't present in \"" + oldTestName + "\"")

def main(arg):  
    parser = argparse.ArgumentParser(description='A simple script to run GPUVerify on CUDA/OpenCL kernels in its test suite.')
    
    #Add command line options
    
    #Mutually exclusive behaviour options
    parser.add_argument("directory", help="Directory to search recursively for kernels.")
    parser.add_argument("--list-xfail-codes", nargs=0, action=PrintXfailCodes, help="List the valid error codes to use with //xfail: and exit.")
    parser.add_argument("--read-pickle",type=str, action=dumpTestResultsAction, help="Dump detailed log information to console from a pickle format file and exit.")
    parser.add_argument("-c","--compare-pickles",type=str,nargs=2,action=comparePickleFiles,help="Compare two test runs recorded in pickle files then exit. The first file should be an old test run and the second file should be a newer test run.")
    
    #General options
    parser.add_argument("--kernel-regex", type=str, default=r"^kernel\.(cu|cl)$", help="Regex for kernel file names (default: \"%(default)s\")")
    parser.add_argument("-s","--stop-on-fail",action="store_true", default=False, help="Stop running tests after first failure. (default: %(default)s)")
    parser.add_argument("-l","--log-level",type=str, default="INFO",choices=['DEBUG','INFO','WARNING','ERROR'])
    parser.add_argument("-w","--write-pickle",type=str, default="", help="Write detailed log information in pickle format to a file")
    parser.add_argument("-p","--canonical-path-prefix",type=str, default="GPUVerifyTestSuite", help="When trying to generate canonical path names for tests, look for this prefix. (default: \"%(default)s\")")
    parser.add_argument("-r,","--compare-run", type=str ,default="", help="After performing test runs compare the result of that run with the runs recorded in a pickle file.")

    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), None))
    logging.debug("Finished parsing arguments.")
    
    #Check the user isn't trying something silly
    if(args.write_pickle == args.compare_run and len(args.write_pickle) > 0):
        logging.error("Write log and comparison log cannot be the same.")
        return GPUVerifyTesterErrorCodes.GENERAL_ERROR
    
    oldTests=None
    if len(args.compare_run) > 0 :
        oldTests=openPickle(args.compare_run)
        
    
    
    recursionRootPath=os.path.abspath(args.directory)
    
    if not os.path.isdir(recursionRootPath):
        logging.error("\"{}\" does not refer an existing directory".format(recursionRootPath))
        return GPUVerifyTesterErrorCodes.FILE_SEARCH_ERROR
    
    matcher=re.compile(args.kernel_regex)
    cudaCount=0
    openCLCount=0
    kernelFiles=[]
    logging.debug("Recursing {}".format(recursionRootPath))
    for(root,dirs,files) in os.walk(recursionRootPath):
        for f in files:
            if(matcher.match(f) != None):
                if f.endswith('cu'): 
                    cudaCount+=1
                    logging.debug("Found CUDA kernel:\"{}\"".format(f))
                if f.endswith('cl'): 
                    openCLCount+=1
                    logging.debug("Found OpenCL kernel:\"{}\"".format(f))
                    
                kernelFiles.append(os.path.join(root,f))
            
         
    logging.info("Found {0} OpenCL kernels and {1} CUDA kernels.".format(openCLCount,cudaCount))
    
    if len(kernelFiles) == 0:
        logging.error("Could not find any OpenCL or CUDA kernels")
        return GPUVerifyTesterErrorCodes.FILE_SEARCH_ERROR
    
    #Do in place sort of paths so we have a guaranteed order
    kernelFiles.sort()
    tests=[]
    for kernelPath in kernelFiles:
        try: 
            tests.append(GPUVerifyTestKernel(kernelPath))
        except KernelParseError as e:
            logging.error(e)
            if args.stop_on_fail:
                return GPUVerifyTesterErrorCodes.KERNEL_PARSE_ERROR

    #run tests
    logging.info("Running tests...")
    for test in tests:
        test.run()
        if not test.testPassed and args.stop_on_fail :
            logging.error("Stopping on test failure.")
            return GPUVerifyTesterErrorCodes.TEST_FAILED
            
            
    logging.info("Finished running tests.")
    
    if len(args.write_pickle) > 0 :
        logging.info("Writing run information to pickle file \"" + args.write_pickle + "\"")
        with open(args.write_pickle,"wb") as output:
            pickle.dump(tests, output, 2) #Use protocol 2
     
    if oldTests!=None:
        doComparison(oldTests,args.compare_run,tests,"Newly completed tests", args.canonical_path_prefix)
        
        
    return GPUVerifyTesterErrorCodes.SUCCESS
            
    
    

if __name__ == "__main__":
    sys.exit(main(sys.argv))
