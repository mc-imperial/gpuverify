from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
import sys
import os
import logging
import signal

def main():
  """
      This program will run the GPUVerify Rise4Fun web service
      as a production server which uses Tornado as the HTTP
      server.
  """

  import argparse
  parser = argparse.ArgumentParser(description=main.__doc__)
  parser.add_argument('-p', '--port', type=int, default=5000, help='Port to use. Default %(default)s')
  parser.add_argument('-f', '--forks', type=int, default=0, help='Number of processes to use. A value of zero will use the number of available cores on the machine. Default %(default)s')
  parser.add_argument("-l","--log-level",type=str, default="INFO",choices=['debug','info','warning','error'])
  parser.add_argument("-o","--log-output",type=argparse.FileType(mode='w'), default='-', help='Write logging information to file. Default "%(default)s"')

  args = parser.parse_args()

  # Setup the root logger before importing app so that the loggers used in app
  # and its dependencies have a handler set
  logging.basicConfig(level=getattr(logging,args.log_level.upper(),None), 
                      stream=args.log_output,
                      format='%(asctime)s:%(name)s:%(levelname)s: %(module)s.%(funcName)s() : %(message)s')
  from webservice import app

  # Add signal handler for SIGTERM that will trigger the same exception that SIGINT does
  def terminate(signum,frame):
    logging.info("PID " + str(os.getpid()) + "Received signal " + str(signum))
    raise KeyboardInterrupt()
  signal.signal(signal.SIGTERM,terminate)

  try:
    logging.info("Starting server on port " + str(args.port))
    http_server = HTTPServer(WSGIContainer(app))
    http_server.bind(args.port)
    http_server.start(args.forks) # Fork multiple sub-processes
    IOLoop.instance().start()
  except KeyboardInterrupt:
    http_server.stop()

  logging.info("Exiting process:" + str(os.getpid()) )

if __name__ == '__main__':
  main()
  sys.exit(0)
