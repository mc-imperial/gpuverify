#!/usr/bin/env python2.7
"""
This script automatically tests the GPUVerifyRise4Fun webservice
on the local machine. To use it start the service on the local machine
and then run this script.

"""
import urllib2
import json
import pprint
import logging
from socket import gethostname
from time import time

width=80

def getJSON(url):
    start = time()
    logging.info('Grabbing url:' + url)
    response = urllib2.urlopen(url,None,20)
    end = time()
    rj = json.loads(response.read())
    logging.info('Run time {0} seconds'.format(end - start))
    logging.debug('Responce:' + pprint.pformat(rj))
    return rj

def postJSON(url, data):
    logging.info('Posting to url:' + url)
    logging.debug('Data to post:' + pprint.pformat(data))
    start = time()
    request = urllib2.Request(url, json.dumps(data),{'Content-Type': 'application/json'} ) 
    response = urllib2.urlopen(request)
    end = time()

    rj = json.loads(response.read())
    print(end - start)
    logging.info('Run time {0} seconds'.format(end - start))
    logging.debug('Responce:' + pprint.pformat(rj))
    return rj

def genURL(server, port,lang,query):
    return 'http://'+ server  + ':' + str(port) + '/' + lang + '/' + query

def test(lang, server, port):
    # Grab meta-data
    md = getJSON( genURL(server, port, lang, 'metadata') )

   # Loop over examples
    for sample in md['Samples']:
        logging.info('Loading file:{}'.format(sample['Name']) )

        # Build JSON Request to process the example
        request = { 'Version' : 'ignored',
                    'Source'  : sample['Source']
                  }

        response = postJSON( genURL(server, port, lang, 'run') , request )
        logging.info('#' * width)
        logging.info('Received response:\n{0}'.format(response['Outputs'][0]['Value']))
        logging.info('#' * width + '\n')
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--port', type=int, default=55000, help='Port to use')
    parser.add_argument('-s', '--server', default='localhost', 
                        help='Server to query. Default "%(default)s"')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Test the languages
    for lang in ['opencl', 'cuda']:
        test(lang, args.server, args.port)

    logging.info('Done')


    
