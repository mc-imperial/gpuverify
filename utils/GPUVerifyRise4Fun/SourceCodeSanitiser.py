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

  def _common_regex_sanitise(self, source, pattern, thingToRemoveAsString):
    """
      This contains the common behaviour for sanitising based on regex
      and removing the entire line matching.
    """
    matcher = re.compile(pattern)

    splitSource = source.splitlines()
    newSource = [ ]
    for (index, line) in enumerate(splitSource): 
      if matcher.search(line):
        lineNumber = index +1
        msg = 'Removing line containing {thing} on line {line}'.format(line=lineNumber,
                                                                       thing=thingToRemoveAsString
                                                                      )
        _logger.warning(msg)
        self.warnings.append(msg)
      else:
        newSource.append(line)

    if len(newSource) == 0:
      return ''
    else:
      return reduce(lambda lineA, lineB: lineA + '\n' + lineB,
                    newSource)

  def sanitise(self, source):
    # Dynamically find all sanitise methods in this object and run them
    methods = inspect.getmembers(self, predicate=inspect.ismethod)

    transformedSource = source
    for (name, method) in methods:
      if not name.startswith('sanitise_'):
        continue

      # Call the method
      _logger.debug('Calling {}'.format(name))
      transformedSource=method(transformedSource)
      #_logger.debug('Transformed source to:\n{}'.format(transformedSource))

    return transformedSource

  def sanitise_includes(self, source):
    pattern = r'#include'
    return self._common_regex_sanitise(source, pattern, '#include')

  def sanitise_includenext(self, source):
    """
      #include_next is a GNU gcc extension to C standard
      http://gcc.gnu.org/onlinedocs/cpp/Wrapper-Headers.html#Wrapper-Headers
    """
    pattern = r'#include_next'
    return self._common_regex_sanitise(source, pattern, '#include_next')

  def sanitise_import(self, source):
    """
      #import is a deprecated feature in gcc, not sure if clang
      supports it but lets just be safe and remove anyway
      http://gcc.gnu.org/onlinedocs/cpp/Alternatives-to-Wrapper-_0023ifndef.html
    """
    pattern = r'#import'
    return self._common_regex_sanitise(source, pattern, '#import')


# For testing
if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)
  s = SourceCodeSanitiser()
  reduced = s.sanitise("""
  #include <iostream>
  int a=5;
  int b=6;
  #include /* HACK */ "Something"
  #include_next "something"
  foo
  #import "something"
  bar
  #include something
  /* HACK */ #include "hahah i got through"
  /* HACK */ #include <hahah i got through.h>
  // XXX
  """)
  print(s.getWarnings())
  print("Reduced form:")
  print(reduced)

