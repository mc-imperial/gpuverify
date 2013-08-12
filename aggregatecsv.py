#!/usr/bin/env python2.7

import getopt
import numpy
import sys

csvInHeader  = "kernel, status, clang, opt, bugle, vcgen, boogiedriver, total"
cvsOutHeader = "kernel, status, population, clang mean, clang error, " \
               + "opt mean, opt error, bugle mean, bugle error, vcgen mean, " \
               + "vcgen error, boogiedriver mean, boogiedriver error, " \
               + "total mean, total error"

# http://en.wikipedia.org/wiki/Student%27s_t-distribution#Table_of_selected_values
tTable95 = [None,  # 0
            12.71, # 1
            4.303, # 2
            3.182, # 3
            2.776, # 4
            2.571, # 5
            2.447, # 6
            2.365, # 7
            2.306, # 8
            2.262, # 9
            2.228, # 10
            2.201, # 11
            2.179, # 12
            2.160, # 13
            2.145, # 14
            2.131] # 15

class CsvError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class CsvData:
    def __init__(self, kernelName, exitStatus):
        self.kernelName = kernelName
        self.exitStatus = exitStatus

        self.clangTime  = []
        self.optTime    = []
        self.bugleTime  = []
        self.vcgenTime  = []
        self.boogieTime = []
        self.totalTime  = []

        self.numTimes = 0
        self.addCount = 0

    def addTimes(self, data):
        length = len(data)

        if length < 3:
            raise CsvError("Row with " + str(length) + " entries found\n" \
                           + "\n  Expected at least 3 entries")
        elif length > 8:
            raise CsvError("Row with " + str(length) + " entries found\n" \
                           + "\n  Expected at at most 8 entries")

        kernelName = data.pop(0)

        if self.kernelName != kernelName:
            raise CsvError("Row with wrong kernel name found\n" \
                           + "  Expected: " + self.kernelName + "\n" \
                           + "  Was     : " + kernelName)

        exitStatus = data.pop(0)

        if self.exitStatus != exitStatus:
            raise CsvError("Exit status for " + self.kernelName \
                           + " not consistent\n  Found " \
                           + self.exitStatus + " and " + exitStatus)

        try:
            numTimes = 0

            if len(data) > 1:
                self.clangTime.append(float(data.pop(0)))
                numTimes += 1

            if len(data) > 1:
                self.optTime.append(float(data.pop(0)))
                numTimes += 1

            if len(data) > 1:
                self.bugleTime.append(float(data.pop(0)))
                numTimes += 1

            if len(data) > 1:
                self.vcgenTime.append(float(data.pop(0)))
                numTimes += 1

            if len(data) > 1:
                self.boogieTime.append(float(data.pop(0)))
                numTimes += 1

            self.totalTime.append(float(data.pop(0)))
            numTimes += 1
        except ValueError as error:
            raise CsvError("Found non-numerical entry for " + self.kernelName)

        if self.numTimes == 0:
            # This is the first time times are added
            self.numTimes = numTimes
        elif numTimes != self.numTimes:
            raise CsvError("Number of times for " + self.kernelName \
                           + " not consistent\n  Found " \
                           + str(self.numTimes) + " and " + str(numTimes))

        self.addCount += 1

    def checkNumberOfEntries(self, number):
        if self.addCount != number:
            raise CsvError("Number of entries for " + self.kernelName \
                           + " not consistent\n" \
                           + "  Expected: " + str(number) + "\n" \
                           + "  Found   : " + str(self.addCount))

    def computeStatistics(self):
        def compute(times):
            mean = numpy.mean(times)
            population = len(times)

            if population < 2:
                raise CsvError("Population size of at least 2 expected")

            # Set ddof to 1, as we do not sample over the complete population
            std  = numpy.std(times, ddof = 1)
            # Compute confidence interval
            se = numpy.divide(std, numpy.sqrt(population))

            if len(tTable95) >= population:
                error = numpy.multiply(tTable95[population - 1], se)
            else:
                raise CsvError("tTable95 has insufficent entries")

            return mean, error
        # end of compute

        if len(self.optTime) > 0:
            self.clangMean, self.clangError = compute(self.clangTime)

        if len(self.optTime) > 0:
            self.optMean, self.optError = compute(self.optTime)

        if len(self.bugleTime) > 0:
            self.bugleMean, self.bugleError = compute(self.bugleTime)

        if len(self.vcgenTime) > 0:
            self.vcgenMean, self.vcgenError = compute(self.vcgenTime)

        if len(self.boogieTime):
            self.boogieMean, self.boogieError = compute(self.boogieTime)

        self.totalMean, self.totalError = compute(self.totalTime)

    def generateMeanCsvEntry(self, alignColumns):
        entry  = self.kernelName \
                 + ", " + self.exitStatus \
                 + ", " + str(self.addCount)

        if len(self.clangTime) > 0:
            entry += ', %.3f' % self.clangMean \
                     + ', %.3f' % self.clangError
        elif alignColumns:
            entry += ", -, -"

        if len(self.optTime) > 0:
            entry += ', %.3f' % self.optMean \
                     + ', %.3f' % self.optError
        elif alignColumns:
            entry += ", -, -"

        if len(self.bugleTime) > 0:
            entry += ', %.3f' % self.bugleMean \
                     + ', %.3f' % self.bugleError
        elif alignColumns:
            entry += ", -, -"

        if len(self.vcgenTime) > 0:
            entry += ', %.3f' % self.vcgenMean \
                     + ', %.3f' % self.vcgenError
        elif alignColumns:
            entry += ", -, -"

        if len(self.boogieTime):
            entry += ', %.3f' % self.boogieMean \
                     + ', %.3f' % self.boogieError
        elif alignColumns:
            entry += ", -, -"

        entry += ', %.3f' % self.totalMean \
                 + ', %.3f' % self.totalError

        return entry

def processFile(fileName, timeData):
    print >> sys.stderr, "Processing", fileName
    f = open(fileName, 'r')

    firstLine = f.readline().strip()
    if firstLine != csvInHeader:
        raise CsvError("Incorrect CSV header found\n" \
                       + "  Expected: " + csvInHeader + "\n" \
                       + "  Was     : " + firstLine)

    for line in f:
        data   = [cell.strip() for cell in line.split(",")]
        length = len(data)

        if length < 2:
            raise CsvError("Row with " + str(length) + " entries found\n" \
                           + "\n  Expected at least 2 entries")

        kernelName = data[0]
        exitStatus = data[1]

        if kernelName not in timeData:
            csvData = CsvData(kernelName, exitStatus)
            timeData[kernelName] = csvData

        timeData[kernelName].addTimes(data)

def checkFileCount(args):
    if len(args) < 2:
        raise CsvError("At least 2 csv files required")

def doPadding(opts):
    for o, a in opts:
        if o == "--padding" or o == "-p":
            return True

    return False

def showHelp(programName, opts):
    for o, a in opts:
        if o == "--help" or o == "-h":
            print "usage: " + programName + " [-h] [-p] csv-files"
            print ""
            print "Script to aggregate csv files produced by"
            print "  gvtester.py --time-as-csv --log-level=CRITICAL directory"
            print ""
            print "arguments:"
            print "  csv-files      csv files to aggregate"
            print ""
            print "optional arguments:"
            print "-h, --help       show this help message and exit"
            print "-p, --padding    padding in csv data output"

            return True

    return False

def main(argv):
    try:
        opts, args = getopt.gnu_getopt(argv[1:], 'hp',
                                       ['help', 'padding'])
    except getopt.GetoptError as getoptError:
        print >> sys.stderr, getoptError.msg, "\nTry --help for option list"
        return 1

    if showHelp(argv[0], opts):
        return 0

    padding = doPadding(opts)

    try:
        timeData  = {}

        checkFileCount(args)

        for a in args:
            processFile(a, timeData)

        print cvsOutHeader

        kernelNames = sorted(timeData.keys())

        for kernelName in kernelNames:
            timeData[kernelName].checkNumberOfEntries(len(args))
            timeData[kernelName].computeStatistics()
            print timeData[kernelName].generateMeanCsvEntry(padding)

    except CsvError as error:
        print >> sys.stderr, error.value
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
