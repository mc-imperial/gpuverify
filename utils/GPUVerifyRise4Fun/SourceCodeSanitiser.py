# vim: set sw=2 ts=2 softtabstop=2 expandtab:
import inspect
import logging
import re

#Internal logger
_logger = logging.getLogger(__name__)

class SourceCodeSanitiser(object):
  def __init__(self):
    self.warnings = [ ]

  def getWarnings(self):
    return self.warnings

  def removeWarnings(self):
    self.warnings = [ ]

  def sanitise(self, source):
    # Dynamically find all sanitise methods in this object and run them
    methods = inspect.getmembers(self, predicate=inspect.ismethod)

    lastResult=None
    for (name, method) in methods:
      if not name.startswith('sanitise_'):
        continue

      # Call the method
      _logger.debug('Calling {}'.format(name))
      lastResult=method(source)

    return lastResult

  def sanitise_includes(self, source):
    matcher = re.compile(r'^\s*#include')

    splitSource = source.splitlines()
    for (index, line) in enumerate(splitSource): 
      if matcher.match(line):
        lineNumber = index +1
        msg = 'Removing line containing #include on line {line}'.format(line=lineNumber)
        _logger.warning(msg)
        self.warnings.append(msg)

        # Remove the line
        splitSource.pop(index)

    if len(splitSource) == 0:
      return ''
    else:
      return reduce(lambda lineA, lineB: lineA + '\n' + lineB,
                    splitSource)



# For testing
if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)
  s = SourceCodeSanitiser()
  reduced = s.sanitise("#include <iostream>\nint a=5;\nin b=6;\n    #include \"Something\"\n")
  print(s.getWarnings())
  print("Reduced form:")
  print(reduced)

