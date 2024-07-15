"""
This module contains the server integration for the COL.
"""

# pylint: disable=W0719

# Default imports
import logging

# Third party imports
import requests as req

# Custom imports
from birbs.col_filtering import COL
import birbs.config.config_loader as config_loader

col_integration_logger = logging.getLogger("COLServerIntegration")


class COLServerIntegration:
    """
    This class is responsible for the server integration of the COL.
    """

    def __init__(self):
        # Initialize the variables
        self.col : COL = None
        self.peers : dict = {}
        self.yacy_info : dict= {}

        # Initialize the configuration
        self.config_loader = config_loader.ConfigLoader()

    def fetch_peers(self):
        """
        Fetch the peers list from yacy's api
        """

        # Fetch the yacy info from the config
        yacy_info = self.config_loader.yacy_settings

        # Fetch the peers
        try:
            url = f"http://localhost:{yacy_info['port']}/yacy/seedlist.json"
            response = req.get(url, timeout=60)

            if response.status_code == 200:
                return response.json()
            
            raise Exception(f"Failed to fetch peers, status code: {response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to fetch peers: {e}") from e

    def start(self):
        """
        This function is for testing purposes.
        """

        col_integration_logger.info("Starting the COL server integration...")
        col_integration_logger.info("Fetching the peers...")

        # Fetch the peers
        peers = self.fetch_peers()

        # Fetch the Yacy info for the active peer
        self.yacy_info = {
            "ip": peers["peers"][0]["IP"],
            "port": peers["peers"][0]["Port"],
            "hash": peers["peers"][0]["Hash"],
        }

        col_integration_logger.info("Fetched the Yacy info: %s", self.yacy_info)

        # Initialize the COL
        self.col = COL(
            self.yacy_info["ip"], self.yacy_info["port"], self.yacy_info["hash"]
        )

        # Start the COL
        self.col.start()

        col_integration_logger.info("COL server integration started")

    def handle_received_message(self, message):
        """
        This function handles the received message.
        """

        # Fetch the message type and data
        message_type = message["msg"]
        data = message["data"]

        # Check if col is initialized
        if self.col is None:
            col_integration_logger.error("COL is not initialized")
            return

        col_integration_logger.info("Message type: %s", message_type)

        # Handle the message based on the message type
        if message_type == "SEND_MODEL":
            self.col.queue.put(data)
            col_integration_logger.info("Model is added to the queue")

        elif message_type == "SEND_DATA":
            # ...
            pass
        elif message_type == "NODE_JOINED":
            # Make a api call to /api/update_whitelist in the flask server and pass the data

            hash_peer = data["hash"]
            ip = data["ip"]
            port = data["port"]

            # Update the whitelist
            if not hash_peer or not ip or not port:
                col_integration_logger.error("Invalid data received for NODE_JOINED: %s", data)
                return

            with open("resources/whitelist/whitelist.json", "a", encoding="utf-8") as f:
                f.write(f'{{"hash": "{hash_peer}", "ip": "{ip}", "port": "{port}"}}\n')

            return
        elif message_type == "NODE_LEFT":
            # ...
            pass
        else:
            # ...
            pass
    
    def fetch_predictions(self):
        """
        This function fetches the predictions from the COL.
        """

        # Check if col is initialized
        if self.col is None:
            col_integration_logger.error("COL is not initialized")
            return

        # Fetch the predictions
        
        query_predictions = self.col.p
        link_predictions = self.col.links

        # Prediction dictionary
        result = {}

        for index, key in enumerate(query_predictions):
            result[key] = link_predictions[index]

        # Format it as a json response
        response = {
            "predictions": result
        }

        return response