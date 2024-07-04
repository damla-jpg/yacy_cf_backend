'''
This module is responsible for sending messages to the server.
'''

# Third party imports
import pickle

# Default imports
import socket
import logging

sender_logger = logging.getLogger("Sender")

def send_message(ip : str, port : int, message : str | dict):
    '''
    This function sends a message to the server.
    '''

    # Initialize the variables
    client_socket = None
    response = None

    # Log the message
    sender_logger.info(f"Trying to send message to {ip}:{port} with message: {message}")
    

    # Send the message
    try:
        # Create a socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sender_logger.info(f"Connecting to {ip}:{port}")

        # Connect to the server
        client_socket.connect((ip, port))

        sender_logger.info(f"Connected to {ip}:{port}")
        sender_logger.info(f"Sending message")
        
        # Convert the message to bytes
        message = pickle.dumps(message)

        # Send the message
        client_socket.send(message)

        sender_logger.info(f"Message sent")

        # Receive the response
        response = pickle.loads(client_socket.recv(1024))

        # print(f"Received response: {response}")

        # TODO: Implement the response handling logic here, this is a TCP connection so 
        # we need to handle the response

        # Close the socket
        client_socket.close()
    except Exception as e:
        sender_logger.error(f"An error occurred while sending message: {e}")
    
    finally:
        if client_socket:
            client_socket.close()

    return response