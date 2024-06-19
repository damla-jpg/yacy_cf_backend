'''
This file contains the server implementation for the Birbs project.
'''

# Default imports
import os
import logging

# Custom imports
from birbs.communication import send_message as send_socket_message

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
    '''
    Home route, returns a simple message.
    '''
    return jsonify(message="Hello, World!")

@app.route('/getPeers', methods=['GET'])
def getPeers():
    '''
    This function returns the list of peers in the network.
    '''

    url = f'http://localhost:{USERPORT}/yacy/seedlist.json'
    response = req.get(url)
    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify(error='Failed to get peers')

@app.route('/profile', methods=['GET'])
def get_profile():
    '''
    This function returns the profile of the user.
    '''
    
    url = f'http://localhost:{USERPORT}/Network.xml'
    response = req.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return jsonify(error='Failed to get profile')
    

@app.route('/search', methods=['GET'])
def search():
    '''
    This function searches for a query in the network.
    '''

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
    '''
    This function returns the contact list of the user.
    '''

    url = f'http://localhost:{USERPORT}/Messages_p.html'
    response = req.get(url, auth=req.auth.HTTPDigestAuth(MY_DIGEST_USERNAME, MY_DIGEST_PASSWORD))
    if response.status_code == 200:
        return response.text
    else:
        return jsonify(error='Failed to get contact list')
    

@app.route('/api/retrieve_message_ids', methods=['GET'])
def retrieve_message_ids():
    '''
    This function retrieves the message ids.
    '''

    url = f'http://localhost:{USERPORT}/Messages_p.xml'
    response = req.get(url, auth=req.auth.HTTPDigestAuth(MY_DIGEST_USERNAME, MY_DIGEST_PASSWORD))
    if response.status_code == 200:
        return response.text
    else:
        return jsonify(error='Failed to retrieve message ids')

@app.route('/api/get_message_contents', methods=['GET'])
def retrieve_message_contents():
    '''
    This function retrieves the message contents.
    '''

    messageId = fl.request.args.get('messageId')
    url = f'http://localhost:{USERPORT}/Messages_p.html?action=view&object={messageId}'
    response = req.get(url, auth=req.auth.HTTPDigestAuth(MY_DIGEST_USERNAME, MY_DIGEST_PASSWORD))
    if response.status_code == 200:
        return response.text
    else:
        return jsonify(error='Failed to retrieve message contents')

@app.route('/api/send_message', methods=['POST'])
def send_message():
    '''
    This function sends a message to a user.
    '''

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
    
@app.route('/api/share_history', methods=['POST'])
def share_history():
    '''
    This function shares the history of the user.
    '''
    server_logger.info("/api/share_history: sharing history...")

    # Get the ip and port from the headers
    ip = fl.request.headers.get('ip')
    port = fl.request.headers.get('port')

    # Check if the ip and port are valid
    if not ip or not port:
        server_logger.error("/api/share_history: Invalid IP or port.")
        return jsonify(error='Invalid IP or port')

    # Get the message from the parameters
    message = fl.request.args.get('message')

    server_logger.info(f"/api/share_history: Sending message to {ip}:{port} with message: {message}")

    # Send the message to socket listener
    try:
        send_socket_message(ip, int(port), message)
        server_logger.info("/api/share_history: Message sent.")

        return jsonify(message='Message sent')
    except Exception as e:
        server_logger.error(f"/api/share_history: An error occurred: {e}")
        
        return jsonify(error='Failed to send message')
    
@app.route('/upload', methods=['POST'])
def upload():
    '''
    This function uploads a file to the server.
    '''

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