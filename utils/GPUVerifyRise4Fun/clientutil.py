"""
This module contains code useful for building a GPUVerifyRise4Fun client
"""
import urllib2
import json
import pprint
import logging
from time import time

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
    logging.info('Run time {0} seconds'.format(end - start))
    logging.debug('Responce:' + pprint.pformat(rj))
    return rj


def genURL(server, port,lang,query):
    url='http://'+ server
    
    if port != None:
        url+=':' + str(port) + '/' 
    else:
        url += '/'
        
    if lang != None:
        url +=  lang + '/' 
        
    url += query
    return url
