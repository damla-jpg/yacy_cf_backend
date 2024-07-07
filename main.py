"""
This is the main file of the project. It is the entry point of the project.
"""
# pylint: disable=W0718, C0301

# Default imports
import os
import sys
import logging
import time
import json

# Custom imports
from birbs.server import start_server
from birbs.communication import Listener
from birbs.config import ConfigLoader
import birbs.server.server as server
import birbs.server.col_server_integration as col_server_integration


class Birbs:
    """
    This class is the main class of the project. Handles the server and the socket listener.
    """

    def __init__(self, run_socket_listener=True):
        """
        Constructor for the Birbs class.
        """

        # Initialize the variables
        self.listener = None
        self.logger = None
        self.run_socket_listener = run_socket_listener
        self.col_integration = None

        # Initialize the logger
        self.logger_rpath = self.init_logger()

        # Initialize the configuration
        self.config_loader = ConfigLoader()

    def init_logger(self):
        """
        This function initializes the logger.
        """

        # Get the current time
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S")

        # Set the logging level
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=f"./resources/logs/server_log_{current_time}.log",
        )

        # Initialize the logger
        self.logger = logging.getLogger("Main")

        # Logging handler for stdout
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        return f"./resources/logs/server_log_{current_time}.log"

    def start_socket_listener(self):
        '''
        This function starts the socket listener.
        '''

        # Try to start the socket listener
        try:
            if self.run_socket_listener:
                self.logger.info("Starting the socket listener...")

                # Start the socket listener
                self.listener = Listener(
                    self.config_loader.socket_settings["host"],
                    self.config_loader.socket_settings["port"],
                )

                self.listener.start()

        except Exception as e:
            self.logger.error("An error occurred during initializing the socket thread: %s", e)

            # Stop the server and the socket listener
            self.stop()

    def start_col_server_integration(self):
        '''
        This function starts the COL server integration.
        '''

        # Try to start the COL server integration
        try:
            self.col_integration = col_server_integration.COLServerIntegration()
            self.col_integration.start()
        except Exception as e:
            self.logger.error(
                "An error occurred during initializing the COL server integration: %s", e
            )

            # Stop the server and the socket listener
            self.stop()

    def start_server(self):
        '''
        This function starts the server.
        '''

        # Try to start the server
        try:
            self.logger.info("Starting the server...")

            # Start the server, this is a blocking call (technically main thread loop)
            start_server(
                self.config_loader.yacy_settings,
                self.config_loader.flask_settings["host"],
                self.config_loader.flask_settings["port"],
                self.col_integration,
            )

            self.logger.info("Quitting the server...")
        except Exception as e:
            self.logger.error("An error occurred during starting the server: %s", e)

    def start(self):
        """
        This function runs the server and the socket listener.
        """

        # Clear the logs
        choice = input("Do you want to clear the logs? (y/n): ")

        if choice.lower() == "y":
            self.clear_logs()

        # Start the socket listener
        self.start_socket_listener()

        # Start the COL server integration
        self.start_col_server_integration()

        # Start the server
        self.start_server()

        self.logger.info("Quitting program...")

        # Stop the server and the socket listener
        self.stop()

    def stop(self):
        """
        This function stops the server and the socket listener.
        """

        # Stop the socket listener
        if self.listener:
            self.listener.stop()

        # Exit the program
        os._exit(0)

    def clear_logs(self):
        """
        This function clears the logs.
        """

        # Delete the logs in the logs directory
        for file in os.listdir("./resources/logs"):
            if file.endswith(".log"):
                try:
                    path = f"./resources/logs/{file}"
                    if path != self.logger_rpath:
                        os.remove(path)
                except PermissionError:
                    # If the file is in use, skip it
                    pass
                except Exception as e:
                    self.logger.error("An error occurred during deleting the log file: %s", e)

if __name__ == "__main__":
    # Initialize the Birbs class
    birbs = Birbs(run_socket_listener=True)

    # Run the program
    # !! Pass False to the constructor to disable the socket listener !!
    birbs.start()
