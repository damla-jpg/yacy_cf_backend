'''
This module is responsible for handling the communication between the peers in the network.
'''

# Default imports
import socket
import threading
import logging
import requests

# Third party imports
import pickle

# Custom imports
from birbs.config import ConfigLoader

NUM_CONNECTIONS = 5
com_logger = logging.getLogger("Communication")

class Listener:
    '''
    This class listens for incoming messages from the network.
    '''

    def __init__(self, ip : str, port : int):
        self.ip = ip
        self.port = port
        self.server_socket = None
        self.stop_signal = threading.Event()

        # Initialize the configuration
        self.config_loader = ConfigLoader()

    def handle_client(self, client_socket : socket.socket):
        '''
        This function handles the incoming messages from the network.
        '''
        
        # Initialize the variables
        message : bytes = b""

        com_logger.info("Handling the client...")

        try:
            while not self.stop_signal.is_set():
                # Receive the message
                chunk = client_socket.recv(1024)
                
                # If the message is empty, break the loop
                if not chunk:
                    break
                
                com_logger.info(f"Received chunk: {chunk}")

                # Append the chunk to the message
                message += chunk

        finally:
            # Send confirmation
            client_socket.send(pickle.dumps("Message received."))
            
            # Close the socket
            client_socket.close()

        message = pickle.loads(message)   

        # Send the message to the backend
        # Get the flask server IP and port
        ip = self.config_loader.flask_settings['host']
        port = self.config_loader.flask_settings['port']

        # Send the message to the server
        try:
            response = requests.post(f"http://{ip}:{port}/api/receive_model", data=message)
            com_logger.info(f"Message sent to the backend. Response: {response}")
        except Exception as e:
            com_logger.error(f"An error occurred while sending the message to the server: {e}")
            
        # Log the message
        com_logger.info(f"Received message: {message}")

        com_logger.info("Client handled.")

    def stop(self):
        '''
        This function stops the socket listener.
        '''

        self.stop_signal.set()

        if self.server_socket:
            self.server_socket.close()

    def start(self):
        '''
        This function starts the socket listener.
        '''

        # Create a new thread for listening
        listener_thread = threading.Thread(target=self.listen, args=(self.ip, self.port))
        listener_thread.start()

    def listen(self, ip : str, port : int):
        '''
        This function listens for incoming messages from the network.
        '''

        if not ip or not port:
            com_logger.error("Invalid IP or port.")
            return

        try:
            com_logger.info(f"Trying to listen on {ip}:{port}...")

            # Create a socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            com_logger.info("Socket created successfully.")
            com_logger.info("Binding the socket...")

            # Bind the socket to the port
            self.server_socket.bind((ip, port))

            com_logger.info("Socket bound successfully.")
            com_logger.info("Listening for incoming connections...")

            # Listen for incoming connections
            self.server_socket.listen(NUM_CONNECTIONS)
        except Exception as e:
            com_logger.error(f"An error occurred during initializing the socket: {e}")
            if self.server_socket:
                self.server_socket.close()
            
            com_logger.info("Quitting the socket listener...")
            return
        
        # Accept incoming connections, main thread loop
        while not self.stop_signal.is_set():
            try:
                # Accept the connection (blocking call, waits for incoming connections
                # if there are not connections the program will be blocked here till a 
                # connection is established)
                client_socket, client_address = self.server_socket.accept()
                
                com_logger.info(f"Connection from {client_address} has been established.")

                # Start a new thread to handle the connection
                thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                thread.start()
            except Exception as e:
                com_logger.error(f"An error occurred during accepting the connection: {e}")
                break
            except KeyboardInterrupt:
                com_logger.info("Listener stopped.")
                break
