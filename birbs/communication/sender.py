'''
This module is responsible for sending messages to the server.
'''

# Default imports
import socket
import logging
2
sender_logger = logging.getLogger("Sender")

def send_message(ip : str, port : int, message : str):
    '''
    This function sends a message to the server.
    '''

    client_socket = None

    sender_logger.info(f"Trying to send message to {ip}:{port} with message: {message}")
    try:
        # Create a socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sender_logger.info(f"Connecting to {ip}:{port}")

        # Connect to the server
        client_socket.connect((ip, port))

        sender_logger.info(f"Connected to {ip}:{port}")
        sender_logger.info(f"Sending message: {message}")
        
        # Convert the message to bytes
        message = message.encode('utf-8')

        # Send the message
        client_socket.send(message)

        sender_logger.info(f"Message sent")

        # Receive the response
        # response = client_socket.recv(1024).decode('utf-8')

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