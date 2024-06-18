'''
This module is responsible for sending messages to the server.
'''

# Default imports
import socket

def send_message(ip : str, port : int, message : str):
    '''
    This function sends a message to the server.
    '''

    client_socket = None

    try:
        # Create a socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the server
        client_socket.connect((ip, port))

        # Send the message
        client_socket.send(message.encode('utf-8'))

        # Receive the response
        response = client_socket.recv(1024).decode('utf-8')

        print(f"Received response: {response}")

        # Close the socket
        client_socket.close()
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        if client_socket:
            client_socket.close()