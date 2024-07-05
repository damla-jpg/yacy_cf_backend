'''
This module contains the server integration for the COL.
'''

# Default imports
import json
import pickle

# Custom imports
from birbs.col_filtering import COL
import birbs.server.server as server

def get_my_peer_info():
    '''
    This function is for testing purposes.
    '''
    response = server.get_peers()
    peers = json.loads(response.data)
    my_peer_ip = peers["peers"][0]["IP"]
    my_peer_port = peers["peers"][0]["Port"]
    my_peer_hash = peers["peers"][0]["Hash"]

    col = COL(my_peer_ip, my_peer_port, my_peer_hash)
    col.start()

def handle_received_message(message):
    '''
    This function handles the received message.
    '''
    # Parse the message
    message = pickle.loads(message)

    # Fetch the message type and data
    message_type = message["msg"]
    data = message["data"]

    # Handle the message based on the message type
    if message_type == "SEND_MODEL":
        # ...
        pass
    elif message_type == "SEND_DATA":
        # ...
        pass
    elif message_type == "NODE_JOINED":
        # ...
        pass
    elif message_type == "NODE_LEFT":
        # ...
        pass
    else:
        # ...
        pass