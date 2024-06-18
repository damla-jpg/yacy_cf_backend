'''
This is the main file of the project. It is the entry point of the project.
'''

# Default imports
import os
import threading
import logging
import time

# Custom imports
from birbs.server import start_server
from birbs.communication import Listener

# Constants
FLASK_PORT = 3001
HOST = 'localhost'
SOCKET_PORT = 3002

class Birbs:
    '''
    This class is the main class of the project. Handles the server and the socket listener.
    '''

    def __init__(self, run_socket_listener : bool = True):
        self.listener = None
        self.logger = None

        # Initialize the logger
        self.init_logger()

        # Run the server
        self.run(run_socket_listener)
        
    def init_logger(self):
        '''
        This function initializes the logger.
        '''

        # Get the current time
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S")

        # Set the logging level
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
            filename=f"./resources/logs/server_log_{current_time}.log"
        )
    
        # Initialize the logger
        self.logger = logging.getLogger("Main")

    def run(self, run_socket_listener : bool = True):
        '''
        This function runs the server and the socket listener.
        '''

        # Try to start the socket listener
        try:
            if run_socket_listener:
                self.logger.info("Starting the socket listener...")
            
                # Start the socket listener
                self.listener = Listener(HOST, SOCKET_PORT)
                self.listener.start(HOST, SOCKET_PORT)

        except Exception as e:
            self.logger.error(f"An error occurred during initializing the socket thread: {e}")
            self.stop()

        # Try to start the server
        try:
            self.logger.info("Starting the server...")

            # Start the server, this is a blocking call (technically main thread loop)
            start_server(HOST, FLASK_PORT)
            
            self.logger.info("Quitting the server...")
        except Exception as e:
            self.logger.error(f"An error occurred during starting the server: {e}")

        self.logger.info("Quitting program...")
        self.stop()

    def stop(self):
        '''
        This function stops the server and the socket listener.
        '''

        if self.listener:
            self.listener.stop()

        os._exit(0)


if __name__ == '__main__':
    # Initialize the Birbs class
    birbs = Birbs()

    # Run the program
    # !! Pass False to the constructor to disable the socket listener !!
    birbs.run()