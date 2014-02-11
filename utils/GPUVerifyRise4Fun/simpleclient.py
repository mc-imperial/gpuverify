#!/usr/bin/env python2.7
"""
This is a simple program to query a running
GPUVerifyRise4Fun instance with a query.

Note any arguments passed after `kernel` will
be passed to GPUVerify instance running via
GPUVerifyRise4Fun.
"""
import argparse
import clientutil
import sys
import logging
import pprint
import urllib2
import os

def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--port', type=int, default=55000, help='Port to use')
    parser.add_argument('-s', '--server', default='localhost', 
                        help='Server to query. Default "%(default)s"')
    parser.add_argument('--metadata', action='store_true', help="Request server's meta data before executing")
    parser.add_argument('--gvhelp', action='store_true', help="Show GPUVerify help information and exit. You need to pass a kernel filename even though it won't be used")
    parser.add_argument("-l","--log-level",type=str, default="INFO",choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])

    parser.add_argument('kernel', help='Kernel to verify')
    (knownArgs, otherArgs) = parser.parse_known_args(argv)

    logging.basicConfig(level=getattr(logging,knownArgs.log_level))

    if knownArgs.gvhelp:
        logging.debug("Fetching help information")
        try:
            md = clientutil.getJSON( clientutil.genURL(knownArgs.server, knownArgs.port, None , 'help') )
        except urllib2.HTTPError as e:
            logging.error("HTTPError: {0}".format(e))
            return 1
        except urllib2.URLError as e:
            logging.error("URLError: {0}".format(e))
            return 1

        if 'help' in md:
            print(md['help'])
            return 0
        else:
            logging.error('Recieved invalid data.')
            return 1
        

    if len(otherArgs) < 2:
        logging.error("There must be at least two arguments passed to GPUVerifyRise4Fun (pass after the kernel filename)")
        return 1
    else:
        logging.info("Passing the following arguments to the GPUVerify executable" +
                     " running on GPUVerifyRise4Fun:\n{}".format(pprint.pformat(otherArgs))
                    )

    langStr=None
    # Work out the language we are using
    if knownArgs.kernel.endswith('.cl'): langStr='opencl'
    if knownArgs.kernel.endswith('.cu'): langStr='cuda'


    if not langStr: 
        logging.error("Can't determine language version from file extension")
        return 1
    logging.debug("Assuming kernel language is {0}".format(langStr))

    if not ( os.path.exists(knownArgs.kernel) and 
             ( os.path.isfile(knownArgs.kernel) or
               os.path.islink(knownArgs.kernel) 
             ) 
           ):
        logging.error("{0} does not exist".format(knownArgs.kernel))
        return 1

    server = knownArgs.server
    port = knownArgs.port


    # Get metadata if necessary
    if knownArgs.metadata:
        logging.debug("Fetching metadata")
        try:
            md = clientutil.getJSON( clientutil.genURL(server, port, langStr, 'metadata') )
        except urllib2.HTTPError as e:
            logging.error("HTTPError: {0}".format(e))
            return 1
        except urllib2.URLError as e:
            logging.error("URLError: {0}".format(e))
            return 1

        logging.info("Fetched meta data:\n{0}".format(pprint.pformat(md)))

    # Build source file to send to GPUVerifyRise4Fun
    
    # Build first line which has GPUVerify arguments on it
    source = "//" + ' '.join(otherArgs) + '\n'

    # Read the source kernel
    with open(knownArgs.kernel, 'r') as f:
        for line in f.readlines():
            source += line

    logging.debug("Sending the following to GPUVerifyRise4Fun:\n{0}".format(source))

    # Build JSON Request
    request = {'Version': 'ignored', # Only Rise4Fun cares about this, we're by-pass thins
               'Source' : source
              }

    logging.info("Please be patient. This can be quite slow...")
    try:
        responce = clientutil.postJSON( clientutil.genURL(server, port, langStr, "run"), request)
    except urllib2.HTTPError as e:
        logging.error("HTTPError: {0}".format(e))
        return 1
    except urllib2.URLError as e:
        logging.error("URLError: {0}".format(e))
        return 1

    logging.info("Response:\n{0}".format(responce["Outputs"][0]["Value"]))
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
