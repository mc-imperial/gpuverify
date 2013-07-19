from meta_data import *
from flask import Flask, jsonify, url_for, request
from socket import gethostname
import logging
import pprint
import traceback

app = Flask(__name__)

#Load configuration from config.py
app.config.from_object('config')

#Internal logger
_logging = logging.getLogger(__name__)

cudaMetaData = {}
openclMetaData = {} 
def initMetaData():
  # Pre-compute metadata information
  global cudaMetaData , openclMetaData 
  cudaMetaData = CUDAMetaData(app.config['SRC_ROOT'])
  openclMetaData = OpenCLMetaData(app.config['SRC_ROOT'])

#Setup routing
@app.errorhandler(404)
def notFound(error):
  return jsonify({ 'Error': error.description} ), 404
  

@app.route('/<lang>/metadata', methods=['GET'])
def getToolInfo(lang):
  if not checkLang(lang):
    abort(400)


  if lang == CUDAMetaData.folderName:
    metaData = cudaMetaData.metadata
  else:
    metaData = openclMetaData.metadata

  # Patch meta data with image URL
  metaData['InstitutionImageUrl'] = url_for('static', filename='imperial-college-logo.png', _external=True)

  # Patch meta data with privacy policy URL
  metaData['PrivacyUrl'] = url_for('static', filename='privacy-policy.html', _external=True)

  # Patch meta data with terms of use URL
  metaData['TermsOfUseUrl'] = url_for('static', filename='terms-of-use.html', _external=True)
  return jsonify(metaData)

@app.route('/<lang>/language', methods=['GET'])
def getLanguageSyntaxDefinition(lang):
  if not checkLang(lang):
    abort(400)

  if lang == CUDAMetaData.folderName:
    metaData = cudaMetaData
  else:
    metaData = openclMetaData

  if metaData.languageSyntax != None:
    return jsonify(metaData.languageSyntax)
  else:
    return jsonify({'Error':'Language syntax definition is not availabe for ' + metaData.folderName})

@app.route('/<lang>/run', methods=['POST'])
def runGpuverify(lang):
  if not checkLang(lang):
    abort(400)
  
  if not request.json:
    abort(400)

  #FIXME: Sanitise source code
  source = request.json['Source']
  _logging.debug('Received request:\n' + pprint.pformat(request.json))


  tool = gvapi.GPUVerifyTool()
  returnMessage = { 'Version':None, 
                    'Outputs': [
                                 { "MimeType":"text/plain",
                                   "Value":None
                                 }
                               ]   
                  }
  returnCode=None
  toolMessage=None
  dimMessage=""
  safeArgs=[]
  try:
    tool.extractOtherCmdArgs(source,safeArgs)

    if lang == CUDAMetaData.folderName:
      blockDim=[]
      gridDim=[]
      tool.extractGridSizeFromSource(source, blockDim, gridDim)

      returnMessage['Version'] = cudaMetaData.metadata['Version']
      (returnCode, toolMessage) = tool.runCUDA(source, blockDim=blockDim , gridDim=gridDim, extraCmdLineArgs=safeArgs)

    else:
      returnMessage['Version'] = openclMetaData.metadata['Version']

      localSize=[]
      numOfGroups=[]
      tool.extractNDRangeFromSource(source,localSize, numOfGroups)

      (returnCode, toolMessage) = tool.runOpenCL(source, localSize=localSize , numOfGroups=numOfGroups, extraCmdLineArgs=safeArgs)

    # We might have an extra message to show.
    extraHelpMessage=""
    if gvapi.helpMessage[returnCode] != "":
      extraHelpMessage = gvapi.helpMessage[returnCode] + '\n' 

    # Strip out any leading new lines from tool output.
    for (index,char) in enumerate(toolMessage):
      if char != '\n':
        toolMessage=toolMessage[index:]
        break
    
    returnMessage['Outputs'][0]['Value'] = (extraHelpMessage + toolMessage).decode('utf8')

  except Exception as e:
    returnMessage['Outputs'][0]['Value'] = 'Web service error:' + str(e)
    _logging.error('Exception occured:\n' + traceback.format_exc())

  _logging.debug('Sending responce:\n' + pprint.pformat(returnMessage))
  return jsonify(returnMessage)
    

def checkLang(lang):
  if lang in BasicMetaData.registeredLanguage:
    return True
  else:
    return False


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  import argparse
  parser = argparse.ArgumentParser(description='Run Development version of GPUVerifyRise4Fun web service')
  parser.add_argument('-p', '--port', type=int, default=55000, help='Port to use. Default %(default)s')
  parser.add_argument('-d', '--debug', action='store_true',default=False, help='Use Flask Debug mode. Default %(default)s')
  parser.add_argument('--public', action='store_true', default=False, help='Make publically accessible. Default %(default)s')
  parser.add_argument('-s','--server-name', type=str, default=gethostname() , help='Set server hostname. Default "%(default)s"')

  args = parser.parse_args()

  app.config['SERVER_NAME'] = args.server_name + ':' + str(args.port)

  print("Setting SERVER_NAME to " + app.config['SERVER_NAME'])

  if args.debug:
    print("Using Debug mode")
    logging.getLogger().setLevel(logging.DEBUG)
  
  # extra options
  options = { }
  if args.public:
    options['host'] = '0.0.0.0'

  initMetaData() #Delay so debug output may be seen
  app.run(debug=args.debug, port=args.port, **options)
else:
  initMetaData()
