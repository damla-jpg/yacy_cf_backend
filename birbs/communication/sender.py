"""
This module is responsible for sending messages to the server.
"""
# pylint: disable=W0703

# Third party imports
import pickle

# Default imports
import socket
import logging
sender_logger = logging.getLogger("Sender")


def send_message(ip: str, port: int, message: str | dict):
    """
    This function sends a message to the server.
    """

    # Initialize the variables
    client_socket = None
    response = None

    # Log the message
    sender_logger.info("Trying to send message to %s:%s", ip, port)

    # Send the message
    try:
        # Create a socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sender_logger.info("Connecting to %s:%s", ip, port)

        # Connect to the server
        client_socket.connect((ip, port))

        sender_logger.info("Connected to %s:%s", ip, port)
        sender_logger.info("Sending message")

        # Convert the message to bytes
        message = pickle.dumps(message)

        # Send the message chunk by chunk
        for i in range(0, len(message), 1024):
            chunk = message[i : i + 1024]
            client_socket.send(chunk)

        sender_logger.info("Message sent")

        # Close the socket
        client_socket.close()
    except Exception as e:
        sender_logger.error("An error occurred while sending message: %s", e)

    finally:
        if client_socket:
            client_socket.close()

    return response

if __name__ == "__main__":
    # Test the send_message function
    IP = "80.57.135.211"
    PORT = 8191
    message = {"msg": "test", "data": "test"}

    response = send_message(IP, PORT, message)