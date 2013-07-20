from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
import sys
import os
import logging

def main():
  """
      This program will run the GPUVerify Rise4Fun web service
      as a production server which uses Tornado as the HTTP
      server.
  """
  # Setup the root logger before importing app so that the loggers used in app
  # and its dependencies have a handler set
  logging.basicConfig(level=logging.INFO)
  from webservice import app

  import argparse
  parser = argparse.ArgumentParser(description=main.__doc__)
  parser.add_argument('-p', '--port', type=int, default=5000, help='Port to use. Default %(default)s')
  parser.add_argument('-f', '--forks', type=int, default=0, help='Number of processes to use. A value of zero will use the number of available cores on the machine. Default %(default)s')

  args = parser.parse_args()

  try:
    print("Starting server on port " + str(args.port))
    http_server = HTTPServer(WSGIContainer(app))
    http_server.bind(args.port)
    http_server.start(args.forks) # Fork multiple sub-processes
    IOLoop.instance().start()
  except KeyboardInterrupt:
    http_server.stop()

  print("Exiting process:" + str(os.getpid()) )

if __name__ == '__main__':
  main()
  sys.exit(0)
