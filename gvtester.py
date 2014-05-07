#!/usr/bin/env python
# encoding: utf-8
# vim: set shiftwidth=4 tabstop=4 expandtab softtabstop=4:
from __future__ import print_function
import os
import sys
import argparse
import logging
import re
import subprocess
import pickle
import time
import string
try:
    # Python 2.x
    from Queue import Queue
except ImportError:
    # Python 3.x
    from queue import Queue

import threading
import multiprocessing # Only for determining number of CPU cores available

from GPUVerify import ErrorCodes

GPUVerifyExecutable=sys.path[0] + os.sep + "GPUVerify.py"

printBarWidth=80

class GPUVerifyErrorCodes(ErrorCodes):
    """ Provides extra error codes and a handy dictionary
        to map error codes to strings.

        The error codes used in this class represent the
        potential exit codes of GPUVerify. We add
        REGEX_MISMATCH_ERROR as that is a possible point of failure
        during testing
    """
    # It is unlikely we'll have more than 99 error codes in the parent
    REGEX_MISMATCH_ERROR=100

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

        # Check that REGEX_MISMATCH_ERROR doesn't conflict
        # with an existing error code
        assert largest < cls.REGEX_MISMATCH_ERROR

        #Build reverse mapping dictionary { num:string }
        codes=[e for e in dir(cls) if not e.startswith('_')]
        for (num,string) in [( getattr(cls,x), x)  for x in codes if type(getattr(cls,x)) == int]:
            cls.errorCodeToString[num]=string

    @classmethod
    def getValidxfailCodes(cls):
        codes=[]
        for codeTuple in cls.errorCodeToString.items():
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
    reverse = dict((value, key) for key, value in enums.items())
    enums['reverseMapping'] = reverse
    return type('Enum', (object,) , enums)

GPUVerifyTesterErrorCodes=enum('SUCCESS', 'FILE_SEARCH_ERROR','KERNEL_PARSE_ERROR', 'TEST_FAILED', 'FILE_OPEN_ERROR', 'GENERAL_ERROR')

class GPUVerifyTestKernel(object):

    def __init__(self,path,timeAsCSV,csvFile,additionalOptions=None):
        """

            Initialise CUDA/OpenCL GPUVerify test.
            path                : The absolute path to the test kernel
            timeAsCSV           : Get CSV timing information from GPUVerify
            additionalOptions   : A list of additional command line options to pass to GPUVerify

            Upon successful parsing of a kernel the follow attributes should be available.
            .expectedReturnCode : The expected return code of GPUVerify
            .gpuverifyCmdArgs   : A list of command line arguments to pass to GPUVerify
            .regex              : A dictionary of regular expressions that map to Success True/False
        """
        logging.debug("Parsing kernel \"{0}\" for test parameters".format(path))
        self.path=path
        self.timeAsCSV=timeAsCSV
        self.csvFile = csvFile

        #Use with so that if exception thrown file is still closed
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

            #Perform variable substitution in commandline arguments (e.g. ${KERNEL_DIR})

            #This defines the substitution mapping, we can easily add more :)
            cmdArgsSubstitution = {
            'KERNEL_DIR':os.path.dirname(self.path) 
            }

            for index in range(0,len(self.gpuverifyCmdArgs)):

                if self.gpuverifyCmdArgs[index].find('$') != -1:
                    #We're probably going to do a substitution
                    template=string.Template(self.gpuverifyCmdArgs[index])
                    logging.debug('Performing command line argument substitution on:' + self.gpuverifyCmdArgs[index])
                    self.gpuverifyCmdArgs[index]=template.substitute(cmdArgsSubstitution)
                    logging.debug('Substitution complete, result:' + self.gpuverifyCmdArgs[index])


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

            #Set variables to be used later
            self.testPassed=None
            self.returnedCode=""
            self.gpuverifyReturnCode=""

            #Finished parsing
            logging.debug("Successfully parsed kernel \"{0}\" for test parameters".format(path))

            if self.timeAsCSV:
              self.gpuverifyCmdArgs.append("--time-as-csv=" + self.path)

            #Add additional GPUVerify command line args
            if additionalOptions != None:
              logging.debug("Adding additional command line arguments" + str(additionalOptions))
              self.gpuverifyCmdArgs.extend(additionalOptions)


    def run(self):
        """ Executes GPUVerify on this test's kernel
            using this instance's parameters. This will set the following additional attributes on completion
            .testPassed : Boolean
            .returnedCode : The return code of the test (includes REGEX_MISMATCH_ERROR)
            .gpuverifyReturnCode : GPUVerify's actual return code (doesn't include REGEX_MISMATCH_ERROR)
        """
        threadStr='[' + threading.currentThread().name + '] '

        cmdLine=[sys.executable, GPUVerifyExecutable] \
            + self.gpuverifyCmdArgs + [self.path]
        try:
            logging.info(threadStr + "Running test " + self.path)
            logging.debug(self) # show pre test information

            processInstance=subprocess.Popen(cmdLine,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             cwd=os.path.dirname(self.path)
                                            )
            stdout, stderr = processInstance.communicate() #Allow program to run and wait for it to exit.

        except KeyboardInterrupt:
            logging.error("Received keyboard interrupt. Attempting to kill GPUVerify process")
            processInstance.kill()
            raise

        # Handle byte/str issue in python 3.
        stdout = stdout.decode()
        stderr = stderr.decode()

        #Record the true return code of GPUVerify
        if processInstance.returncode < 0:
          # Treat the test as skipped.
          logging.error(threadStr + 'An external program killed test "'+
                        self.path + '" with signal ' +
                        str(-1*processInstance.returncode))
          return
        else:
          self.gpuverifyReturnCode=processInstance.returncode
          logging.debug("GPUVerify return code:" + GPUVerifyErrorCodes.errorCodeToString[self.gpuverifyReturnCode])

        #Do Regex tests if the rest of the test went okay
        if self.gpuverifyReturnCode == self.expectedReturnCode:
            for regexToMatch in self.regex.keys():
                matcher=re.compile(regexToMatch, re.MULTILINE) #Allow ^ to match the beginning of multiple lines
                if matcher.search(stdout) == None and matcher.search(stderr) == None:
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
            logging.error(threadStr + self.path + " FAILED with " + GPUVerifyErrorCodes.errorCodeToString[self.returnedCode] +
                         " expected " + GPUVerifyErrorCodes.errorCodeToString[self.expectedReturnCode])

            #Print output for user to see
            if logging.getLogger().getEffectiveLevel() != logging.CRITICAL:
                for line in stderr.split('\n'):
                    print(line)
                for line in stdout.split('\n'):
                    print(line)
        else:
            self.testPassed=True
            logging.info(threadStr + self.path + " PASSED (" +
                         ("pass" if self.expectedReturnCode == GPUVerifyErrorCodes.SUCCESS else "xfail") + ")")

        if self.timeAsCSV:
            #Print csv output for user to see
            for line in stdout.split('\n'):
                commaSplitLine = line.split(',')
                if len(commaSplitLine) == 9:
                    self.csvFile.write(line + '\n')
                    self.csvFile.flush()
                    break
        del self.csvFile # We cannot serialise this object so we need to remove it from this class!

        logging.debug(self) #Show after test information

    def hasBeenExecuted(self):
        if self.testPassed == None:
            return False
        else:
            return True

    def __str__(self):
        testString="GPUVerifyTestKernel:\nFull Path:{0}\nExpected exit code:{1}\nCmdArgs: {2}\n".format(
              self.path,
              GPUVerifyErrorCodes.errorCodeToString[self.expectedReturnCode],
              self.gpuverifyCmdArgs,
          )

        if self.testPassed == None:
          #Test has not yet been run
          if len(self.regex) == 0:
              testString+= "No regular expressions.\n"
          else:
              for regex in self.regex.keys():
                  testString+= "Regex:\"{0}\"\n".format(regex)

          testString+= "Kernel has not yet been executed.\n"

        else:
          #Test has been run, show more info
          testString+= "Kernel has been executed.\n"
          testString+= "Passed: " + str(self.testPassed) + "\n"
          testString+= "Actual result:" + GPUVerifyErrorCodes.errorCodeToString[self.returnedCode] + "\n"
          testString+= "GPUVerify return code:" + GPUVerifyErrorCodes.errorCodeToString[self.gpuverifyReturnCode] + "\n"
          if len(self.regex) > 0:
              testString+= "Regular expression matching:\n"
              for (regex,succeeded) in self.regex.items():
                  testString+= "\"" + regex +"\" : " + ("MATCHED" if succeeded else "FAILED TO MATCH") + "\n"
          else:
              testString+="No regular expressions.\n"

        return testString

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

def getPickleOptions():
    """ Returns a dictionary of named options to
        use with pickle commands
    """
    extraOptions = {}
    if sys.version_info.major > 2:
        # Make Python 3 try to produce pickle files readable in python 2
        extraOptions['fix_imports'] = True
    return extraOptions

def openPickle(path):
    try:
        with open(path,"rb") as inputFile:
            return pickle.load(inputFile, **getPickleOptions())
    except IOError:
        logging.error("Failed to open pickle file \"" + path + "\"")
        sys.exit(GPUVerifyTesterErrorCodes.FILE_OPEN_ERROR)
    except pickle.UnpicklingError:
        logging.error("Failed to parse pickle file \" + path + \"")
        sys.exit(GPUVerifyTesterErrorCodes.FILE_OPEN_ERROR)

def dumpTestResults(tests,prefix):
    try:
        summariseTests(tests)
        print("Printing results of " + str(len(tests)) + " tests")
        for testObj in tests:
            print("\n" + ("#" * printBarWidth)) #Header bar
            try:
                print("Test:" + getCanonicalTestName(testObj.path, prefix))
            except CanonicalisationError as e:
                logging.error(e)
                print("Test: No canonical name")

            print(testObj)
            print("#" * printBarWidth) #Footer bar
    except IOError:
        #This can happen if piping output to tool such as less
        sys.exit(0)

#This Action can be triggered from the command line
class dumpTestResultsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        logging.getLogger().setLevel(level=getattr(logging, namespace.log_level.upper(), None)) # enable logging
        dumpTestResults(openPickle(values),namespace.canonical_path_prefix)
        sys.exit(GPUVerifyTesterErrorCodes.SUCCESS)

#This Action can be triggered from the command line
class comparePickleFiles(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        logging.getLogger().setLevel(level=getattr(logging, namespace.log_level.upper(), None)) # enable logging
        
        # Check pickle files exist
        for pFile in [ values[0], values[1]]:
            if not os.path.exists(pFile):
                logging.error("'{0}' does not exist.".format(pFile))
                sys.exit(GPUVerifyTesterErrorCodes.FILE_OPEN_ERROR)

        # First argument should be older set of tests than second argument.
        if os.path.getmtime(values[0]) > os.path.getmtime(values[1]):
            logging.error( ("'{0}' is newer than '{1}'.\nYou probably specified the arguments the wrong way round." +
                           "\nIf you really want to perform the comparision this way round run `touch {1}` first.").format(
                            values[0], values[1]))
            sys.exit(GPUVerifyTesterErrorCodes.GENERAL_ERROR)

        result = doComparison(openPickle(values[0]),values[0],openPickle(values[1]),values[1],namespace.canonical_path_prefix)
        if result in [-1, 0]: sys.exit(GPUVerifyTesterErrorCodes.SUCCESS)
        else: sys.exit(1)

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
    changeIsWorse = False
    for (cPath, oldTest) in oldTestDic.items():
        if cPath in newTestDic:
            #Found a test common to both test sets

            #Look for a change in result
            newTest = newTestDic[cPath]
            if oldTest.testPassed != newTest.testPassed:
                logging.warning('#'*printBarWidth)
                logging.warning("Test \"" + cPath + "\" result has changed.\n" +
                                "Test \"" + cPath + "\" from \"" + oldTestName + "\":\n\n" +
                                str(oldTest) + '\n' +
                                "Test \"" + cPath + "\" from \"" + newTestName + "\"\n" +
                                str(newTest))
                changedTestCounter+=1
                # change is for the worse
                if oldTest.testPassed:
                    changeIsWorse = True
        else:
            logging.warning("Test \"" + cPath + "\" present in \"" + oldTestName+ "\" was missing from \"" + newTestName + "\"")
            missingTestCounter+=1
            changeIsWorse = True

    logging.info('#'*printBarWidth + '\n')
    #Iterate over just completed tests to notify about newly introduced tests.
    for cPath in newTestDic.keys():
        if cPath not in oldTestDic:
            logging.warning("Test \"" + cPath + "\" was executed in \"" + newTestName + "\" but was not present in \"" + oldTestName + "\"" )
            newTestCounter+=1

    logging.info('#'*printBarWidth + '\n')
    #Print test summary
    logging.info("Summary:\n" +
                 "# of changed tests:" + str(changedTestCounter) + '\n' +
                 "# of missing tests:" + str(missingTestCounter) + '\n' +
                 "The above is number of tests in \"" + oldTestName + "\" that aren't present in \"" + newTestName + "\"\n" +
                 "# of new tests:" + str(newTestCounter) + '\n' +
                 "The above is number of tests in \"" + newTestName + "\" that aren't present in \"" + oldTestName + "\"")

    if changeIsWorse:
        if newTestCounter != 0 or missingTestCounter != 0: 
            # If new tests have been added or some tests are missing
            # and some tests have failed then the is not really 
            # any ordering
            logging.info(oldTestName + " != " + newTestName)
        else:
            logging.info(oldTestName + " > " + newTestName)
        return 1
    elif changedTestCounter == 0 and missingTestCounter == 0 and newTestCounter == 0:
        logging.info(oldTestName + " = " + newTestName)
        return 0
    else:
        # Adding tests is fine.
        logging.info(oldTestName + " < " + newTestName)
        return -1

def summariseTests(tests):
    """
        Iterates through a list of GPUVerifyTestKernel objects and prints out a summary
    """

    OpenCLCounter=0
    CUDACounter=0
    passCounter=0
    failCounter=0
    xfailCounter=0
    skipCounter=0

    for test in tests:
        #Record kernel type
        if test.path.endswith('cl'):
            OpenCLCounter += 1
        else:
            CUDACounter += 1

        if test.hasBeenExecuted():
            #Record if test was pass/fail/xfail
            if test.testPassed:
                if test.returnedCode == GPUVerifyErrorCodes.SUCCESS:
                    passCounter += 1
                else:
                    xfailCounter += 1
            else:
                failCounter += 1
        else:
            skipCounter += 1

    #Print output
    print('#'*printBarWidth)
    print('')
    print('Summary:')
    print('# of tests:{0}'.format(len(tests)))
    print('# OpenCL kernels:{0}'.format(OpenCLCounter))
    print('# CUDA kernels:{0}'.format(CUDACounter))
    print('# of passes:{0}'.format(passCounter))
    print('# of expected failures (xfail):{0}'.format(xfailCounter))
    print('# of unexpected failures:{0}'.format(failCounter))
    print('# of tests skipped:{0}'.format(skipCounter))
    print('')
    print('#'*printBarWidth)

class Worker(threading.Thread):
    def __init__(self,theQueue):
        threading.Thread.__init__(self)
        self.theQueue = theQueue

        #We will be abruptly killed in main thread exits
        self.daemon = True

    def run(self):
        while True:
            test=self.theQueue.get(block=True,timeout=None)
            test.run()
            #Notify the Queue that we finished our task
            self.theQueue.task_done()

class ThreadPool:
    def __init__(self, numberOfThreads):
        self.theQueue = Queue(0);

        #Create the Threads
        self.threads= []
        for tid in range(numberOfThreads):
            self.threads.append( Worker(self.theQueue) )

    def addTest(self, test):
        self.theQueue.put(test)

    def start(self):
        for tid in self.threads:
            tid.start()

    def waitForCompletion(self):
      """ This will wait for the queue to empty.
          We use a polling wait because Queue.join()
          blocks until queue is empty, even if we get
          sent signals like SIGTERM (i.e. We can't
          catch KeyboardInterrupt at the right time)

      """
      try:
        # WARNING: unfinished_tasks is not part of public API
        # so this might break in future. Cannot use Queue.empty()
        # because that will return true before threads complete
        while self.theQueue.unfinished_tasks > 0:
          time.sleep(0.5)
      except KeyboardInterrupt:
        logging.error("Received keyboard interrupt. Clearing queue!")

        while not self.theQueue.empty():
          try:
            self.theQueue.get()
            self.theQueue.task_done()
          except Queue.Empty:
            break #Handle potential race where the queue becomes empty whilst executing try block

        raise

def main(arg):
    parser = argparse.ArgumentParser(description='A simple script to run GPUVerify on CUDA/OpenCL kernels in its test suite.')
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')
    #Add command line options

    #Mutually exclusive behaviour options
    parser.add_argument("directory", help="Directory to search recursively for kernels.")
    parser.add_argument("--list-xfail-codes", nargs=0, action=PrintXfailCodes, help="List the valid error codes to use with //xfail: and exit.")
    parser.add_argument("--read-pickle",type=str, action=dumpTestResultsAction, help="Dump detailed log information to console from a pickle format file and exit.")
    parser.add_argument("-c","--compare-pickles",type=str,nargs=2,action=comparePickleFiles,help="Compare two test runs recorded in pickle files then exit. The first file should be an old test run and the second file should be a newer test run.")

    #General options
    parser.add_argument("--test-filename-regex", "--kernel-regex", type=str, default=r'(^kernel\.(cu|cl)$)|(^.+\.misc$)', help="Regex for test file names (default: \"%(default)s\")")
    parser.add_argument("--from-file", type=str, default=None, action='append', help="File containing relative paths of kernels to test")
    parser.add_argument("--ignore-file", type=str, default=None, action='append', help="File containing relative paths of kernels to ignore")
    parser.add_argument("-l","--log-level",type=str, default="INFO",choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    parser.add_argument("-w","--write-pickle",type=str, default="", help="Write detailed log information in pickle format to a file")
    parser.add_argument("-p","--canonical-path-prefix", type=str, default="testsuite", help="When trying to generate canonical path names for tests, look for this prefix. (default: \"%(default)s\")")
    parser.add_argument("-r,","--compare-run", type=str, default="", help="After performing test runs compare the result of that run with the runs recorded in a pickle file.")
    parser.add_argument("-j","--threads", type=int, default=multiprocessing.cpu_count(), help="Number of tests to run in parallel (default: %(default)s)")
    parser.add_argument("--gvopt=", type=str, default=None, action='append',
                        help="Pass a command line options to GPUVerify for all tests. This option can be specified multiple times.  E.g. --gvopt=--keep-temps --gvopt=--no-benign",
                        metavar='GPUVerifyCmdLineOption')
    parser.add_argument("--time-as-csv", action="store_true", default=False, help="Print timing of each test as CSV")
    parser.add_argument("--csv-file", type=str, default=None, help="Write timing data to a file (Note: requires --time-as-csv to be enabled)")
    parser.add_argument("--stop-on-fail", action="store_true", default=False, help="Stop on first failure")
    parser.add_argument("--shuffle", type=int, default=None, help="Permute the order of tests under evaluation")

    #Mutually exclusive test run options
    runGroup = parser.add_mutually_exclusive_group()
    runGroup.add_argument("--run-only-pass",action="store_true",default=False,help="Run only the tests that are expected to pass (default: \"%(default)s\")")
    runGroup.add_argument("--run-only-xfail",action="store_true",default=False,help="Run only the tests that are expected to fail (xfail) (default: \"%(default)s\")")


    args = parser.parse_args(arg)

    logging.getLogger().setLevel(level=getattr(logging, args.log_level.upper(), None))
    logging.debug("Finished parsing arguments.")

    #Check the user isn't trying something silly
    if(args.write_pickle == args.compare_run and len(args.write_pickle) > 0):
        logging.error("Write log and comparison log cannot be the same.")
        return GPUVerifyTesterErrorCodes.GENERAL_ERROR

    #Check the number of threads isn't stupid
    if args.threads > 64:
        logging.error("The number of threads requested is too high.")
        return GPUVerifyTesterErrorCodes.GENERAL_ERROR

    oldTests=None
    if len(args.compare_run) > 0 :
        oldTests=openPickle(args.compare_run)

    recursionRootPath=os.path.abspath(args.directory)

    if not os.path.isdir(recursionRootPath):
        logging.error("\"{}\" does not refer an existing directory".format(recursionRootPath))
        return GPUVerifyTesterErrorCodes.FILE_SEARCH_ERROR

    cudaCount=0
    openCLCount=0
    miscCount=0
    kernelFiles=[]
    if args.shuffle:
        import random
        random.seed(args.shuffle)
        random.shuffle(kernelFiles)
    if (args.from_file):
      for f in args.from_file:
        kernels = [line.strip() for line in open(f) if not line.startswith('#')]
        for kernel in kernels:
          if kernel.endswith('cu'):
            cudaCount+=1
          elif kernel.endswith('cl'):
            openCLCount+=1
          elif kernel.endswith('misc'):
              miscCount+=1
          else:
            logging.debug("Not a valid kernel:\"{}\"".format(kernel))
            return GPUVerifyErrorCodes.FILE_SEARCH_ERROR
          kernelFiles.append(os.path.join(recursionRootPath,kernel))

    else:
      matcher=re.compile(args.test_filename_regex)
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
                  if f.endswith('misc'):
                      miscCount+=1
                      logging.debug("Found miscellaneous test:\"{}\"".format(f))

                  kernelFiles.append(os.path.join(root,f))

    cudaCountIgnored=0
    openCLCountIgnored=0
    miscCountIgnored=0
    kernelFilesIgnored=[]
    if (args.ignore_file):
      for f in args.ignore_file:
        kernels = [line.strip() for line in open(f) if not line.startswith('#')]
        for kernel in kernels:
          if kernel.endswith('cu'):
            cudaCountIgnored+=1
          elif kernel.endswith('cl'):
            openCLCountIgnored+=1
          elif kernel.endswith('misc'):
            miscCountIgnored+=1
          else:
            logging.debug("Not a valid kernel:\"{}\"".format(kernel))
            return GPUVerifyErrorCodes.FILE_SEARCH_ERROR
          kernelFilesIgnored.append(os.path.join(recursionRootPath,kernel))
        kernelFiles = list(set(kernelFiles) - set(kernelFilesIgnored))

      logging.info("Found    {0} OpenCL kernels, {1} CUDA kernels and {2} miscellaneous tests".format(openCLCount,cudaCount,miscCount))
      logging.info("Ignoring {0} OpenCL kernels, {1} CUDA kernels and {2} miscellaneous tests".format(openCLCountIgnored,cudaCountIgnored,miscCountIgnored))
      logging.info("Running  {0} OpenCL kernels, {1} CUDA kernels and {2} miscellaneous tests".format(openCLCount-openCLCountIgnored,cudaCount-cudaCountIgnored,miscCount-miscCountIgnored))
    else:
      logging.info("Found {0} OpenCL kernels, {1} CUDA kernels and {2} miscellaneous tests".format(openCLCount,cudaCount,miscCount))

    if len(kernelFiles) == 0:
        logging.error("Could not find any OpenCL, CUDA kernels or miscellaneous tests")
        return GPUVerifyTesterErrorCodes.FILE_SEARCH_ERROR

    #Do in place sort of paths so we have a guaranteed order
    kernelFiles.sort()
    tests=[]
    csvFile = open(args.csv_file,"w") if args.csv_file else sys.stdout
    for kernelPath in kernelFiles:
        try:
            tests.append(GPUVerifyTestKernel(kernelPath, args.time_as_csv, csvFile, getattr(args,'gvopt=') ))
        except KernelParseError as e:
            logging.error(e)
            if args.stop_on_fail:
                return GPUVerifyTesterErrorCodes.KERNEL_PARSE_ERROR

    #run tests
    logging.info("Using " + str(args.threads) + " threads")
    threadPool = ThreadPool(args.threads)

    logging.info("Running tests...")

    if args.time_as_csv:
        print("kernel,status,clang,opt,bugle,vcgen,cruncher,boogiedriver,total", file=csvFile)

    start = time.time()
    for test in tests:
        if args.run_only_pass and test.expectedReturnCode != GPUVerifyErrorCodes.SUCCESS :
            logging.warning("Skipping xfail test:{0}".format(test.path))
            continue

        if args.run_only_xfail and test.expectedReturnCode == GPUVerifyErrorCodes.SUCCESS :
            logging.warning("Skipping pass test:{0}".format(test.path))
            continue

        threadPool.addTest(test)

    #Start tests
    threadPool.start()
    try:
      threadPool.waitForCompletion()
    except KeyboardInterrupt:
      sys.exit(GPUVerifyTesterErrorCodes.GENERAL_ERROR)

    end = time.time()
    logging.info("Finished running tests.")

    if logging.getLogger().getEffectiveLevel() != logging.CRITICAL:
      summariseTests(tests)

    if len(args.write_pickle) > 0 :
        logging.info("Writing run information to pickle file \"" + args.write_pickle + "\"")
        with open(args.write_pickle,"wb") as output:
            pickle.dump(tests, output, protocol=2, **getPickleOptions())

    if oldTests!=None:
        doComparison(oldTests,args.compare_run,tests,"Newly completed tests", args.canonical_path_prefix)

    if logging.getLogger().getEffectiveLevel() != logging.CRITICAL:
        print("Time taken to run tests: " + str((end - start)) )

    return GPUVerifyTesterErrorCodes.SUCCESS


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
