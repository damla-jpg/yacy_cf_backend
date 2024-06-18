'''
This file contains the server implementation for the Birbs project.
'''

# Default imports
import os
import logging

# Third-party imports
import flask as fl
import requests as req
from flask import Flask, jsonify
from flask_cors import CORS
from flask_httpauth import HTTPDigestAuth

# Initialize the Flask app
app = Flask(__name__)
CORS(app)
auth = HTTPDigestAuth()

# Constants
MY_DIGEST_USERNAME = None
MY_DIGEST_PASSWORD = None
USERPORT = None

server_logger = logging.getLogger("Server")

@app.route('/')
def home():
    return jsonify(message="Hello, World!")

@app.route('/api/send_message', methods=['POST'])
def send_message():
    hash = fl.request.args.get('hash')
    subject = fl.request.args.get('subject')
    message = fl.request.args.get('message')
    url = f'http://localhost:{USERPORT}/MessageSend_p.html?hash={hash}&subject={subject}&message={message}'
    print('url:', url)
    response = req.post(url, auth=req.auth.HTTPDigestAuth(MY_DIGEST_USERNAME, MY_DIGEST_PASSWORD))
    if response.status_code == 200:
        return jsonify(response.text)
    else:
        return jsonify(error='Failed to send message')

# turn this function to python
@app.route('/getPeers', methods=['GET'])
def getPeers():
    url = f'http://localhost:{USERPORT}/yacy/seedlist.json'
    response = req.get(url)
    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify(error='Failed to get peers')

@app.route('/profile', methods=['GET'])
def get_profile():
    url = f'http://localhost:{USERPORT}/Network.xml'
    response = req.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return jsonify(error='Failed to get profile')
    

@app.route('/search', methods=['GET'])
def search():
    query = fl.request.args.get('query')
    startRecord = fl.request.args.get('startRecord')
    url = f'http://localhost:{USERPORT}/yacysearch.json?query={query}&resource=global&urlmaskfilter=.*&prefermaskfilter=&nav=all&startRecord={startRecord}'
    response = req.get(url)
    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify(error='Failed to search')


@app.route('/api/get_contact_list', methods=['GET'])
def get_contact_list():
    url = f'http://localhost:{USERPORT}/Messages_p.html'
    response = req.get(url, auth=req.auth.HTTPDigestAuth(MY_DIGEST_USERNAME, MY_DIGEST_PASSWORD))
    if response.status_code == 200:
        return response.text
    else:
        return jsonify(error='Failed to get contact list')
    

@app.route('/api/retrieve_message_ids', methods=['GET'])
def retrieve_message_ids():
    url = f'http://localhost:{USERPORT}/Messages_p.xml'
    response = req.get(url, auth=req.auth.HTTPDigestAuth(MY_DIGEST_USERNAME, MY_DIGEST_PASSWORD))
    if response.status_code == 200:
        return response.text
    else:
        return jsonify(error='Failed to retrieve message ids')

@app.route('/api/get_message_contents', methods=['GET'])
def retrieve_message_contents():
    messageId = fl.request.args.get('messageId')
    url = f'http://localhost:{USERPORT}/Messages_p.html?action=view&object={messageId}'
    response = req.get(url, auth=req.auth.HTTPDigestAuth(MY_DIGEST_USERNAME, MY_DIGEST_PASSWORD))
    if response.status_code == 200:
        return response.text
    else:
        return jsonify(error='Failed to retrieve message contents')

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in fl.request.files:
        return jsonify(error='No files were uploaded')
    file = fl.request.files['files']
    if file.filename == '':
        return jsonify(error='No files were uploaded')
    if file:
        if not os.path.exists('history'):
            os.makedirs('history')
        file.save(os.path.join('history', file.filename))
        return jsonify(message='File uploaded')
    else:
        return jsonify(error='Error uploading file')

def start_server(yacy_settings: dict, host : str, port : int):
    '''
    This function starts the server.
    '''

    # Check if the settings are valid
    if not yacy_settings or not host or not port:
        server_logger.error("Invalid settings.")
        return

    # Set the global variables
    global MY_DIGEST_USERNAME, MY_DIGEST_PASSWORD, USERPORT
    MY_DIGEST_USERNAME = yacy_settings['username']
    MY_DIGEST_PASSWORD = yacy_settings['password']
    USERPORT = yacy_settings['port']

    try:
        app.run(host=host, port=port)
    except Exception as e:
        server_logger.error(f"An error occurred during starting the server: {e}")
    except KeyboardInterrupt:
        server_logger.info("Server stopped.")