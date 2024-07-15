"""
This module is responsible for handling the communication between the peers in the network.
"""

# pylint: disable=W0718

# Default imports
import socket
import threading
import logging
import pickle
import time

# Third party imports
import requests

# Custom imports
from birbs.config import ConfigLoader
from birbs.col_filtering.evalutation import evaluate_receiving as eval_sys

NUM_CONNECTIONS = 5
com_logger = logging.getLogger("Communication")


class Listener:
    """
    This class listens for incoming messages from the network.
    """

    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.server_socket = None
        self.stop_signal = threading.Event()

        # Initialize the configuration
        self.config_loader = ConfigLoader()

    def handle_client(self, client_socket: socket.socket):
        """
        This function handles the incoming messages from the network.
        """
        start_time = time.time()
        # Initialize the variables
        message: bytes = b""

        com_logger.info("Handling the client...")

        try:
            while not self.stop_signal.is_set():
                # Receive the message
                chunk = client_socket.recv(1024)

                # If the message is empty, break the loop
                if not chunk:
                    break

                # Append the chunk to the message
                message += chunk

        finally:
            # Close the socket
            client_socket.close()

        end_time = time.time()
        
        message = pickle.loads(message)

        # Send the message to the backend
        # Get the flask server IP and port
        ip = self.config_loader.flask_settings["host"]
        port = self.config_loader.flask_settings["port"]

        if message["msg"] == "SEND_MODEL":
            message_to = str(str(message["data"][1]) + ":" + str(message["data"][2]))
            message_from = str(str(message["data"][4][1]) + ":" + str(message["data"][4][2]))
        else:
            #TODO: Change the to
            message_to = None
            message_from = str(str(message["data"]["ip"]) + ":" + str(message["data"]["port"]))
            
        # Evaluate the system
        eval_sys(start_time, len(pickle.dumps(message)), end_time, message_from, message_to)

        # Send the message to the server
        try:
            if not message:
                com_logger.error("Empty message received.")
            else:
                temp_message = pickle.dumps(message)

                response = requests.post(
                    f"http://{ip}:{port}/api/receive_model", data=temp_message, timeout=60
                )
                com_logger.info("Message sent to the backend. Response: %s", response)
        except Exception as e:
            com_logger.error(
                "An error occurred while sending the message to the server: %s", e
            )

        
        # Log the message
        com_logger.info("Received message: %s", message)

        com_logger.info("Client handled.")

    def stop(self):
        """
        This function stops the socket listener.
        """

        self.stop_signal.set()

        if self.server_socket:
            self.server_socket.close()

    def start(self):
        """
        This function starts the socket listener.
        """

        # Create a new thread for listening
        listener_thread = threading.Thread(
            target=self.listen, args=(self.ip, self.port)
        )
        listener_thread.start()

    def listen(self, ip: str, port: int):
        """
        This function listens for incoming messages from the network.
        """

        if not ip or not port:
            com_logger.error("Invalid IP or port.")
            return

        try:
            com_logger.info("Trying to listen on %s:%s", ip, port)

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
            com_logger.error("An error occurred during initializing the socket: %s", e)
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

                com_logger.info(
                    "Connection from %s has been established.", client_address
                )

                # Start a new thread to handle the connection
                thread = threading.Thread(
                    target=self.handle_client, args=(client_socket,)
                )
                thread.start()
            except Exception as e:
                com_logger.error(
                    "An error occurred during accepting the connection: %s", {e}
                )
                break
            except KeyboardInterrupt:
                com_logger.info("Listener stopped.")
                break
