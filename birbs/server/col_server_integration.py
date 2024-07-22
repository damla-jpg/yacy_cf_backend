"""
This module contains the server integration for the COL.
"""

# pylint: disable=W0719

# Default imports
import logging
import json
import os

# Third party imports
import requests as req

# Custom imports
from birbs.col_filtering import COL

# import birbs.config.config_loader as config_loader

col_integration_logger = logging.getLogger("COLServerIntegration")


class COLServerIntegration:
    """
    This class is responsible for the server integration of the COL.
    """

    def __init__(self):
        # Initialize the variables
        self.col: COL = None
        self.peers: dict = {}
        self.yacy_info: dict = {}
        self.yacy_service = os.getenv("YACY_SERVICE", "localhost")
        self.yacy_port = os.getenv("YACY_PORT", "8090")

        # Initialize the configuration
        # self.config_loader = config_loader.ConfigLoader()

    def fetch_peers(self):
        """
        Fetch the peers list from yacy's api
        """

        # Fetch the yacy info from the config
        # yacy_info = self.config_loader.yacy_settings

        # Fetch the peers
        try:
            url = f"http://{self.yacy_service}:{self.yacy_port}/yacy/seedlist.json"
            response = req.get(url, timeout=60)

            if response.status_code == 200:
                return response.json()

            raise Exception(
                f"Failed to fetch peers, status code: {response.status_code}"
            )
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

    def add_to_whitelist(self, data):
        """
        This function updates the whitelist with the new peer.
        """

        hash_peer = data["hash"]
        ip = data["ip"]
        port = data["port"]
        peer_dict = {"whitelist": []}

        # Update the whitelist
        if not hash_peer or not ip or not port:
            col_integration_logger.error(
                "Invalid data received for NODE_JOINED: %s", data
            )
            return

        try:
            with open("resources/whitelist/whitelist.json", "r", encoding="utf-8") as f:
                peer_dict = json.load(f)
        except FileNotFoundError:
            col_integration_logger.error("Whitelist file not found")
        except json.JSONDecodeError:
            col_integration_logger.error("Error decoding JSON")

        for peer in peer_dict["whitelist"]:
            if peer["hash"] == hash_peer:
                col_integration_logger.error(
                    "Peer already exists in the whitelist: %s", data
                )
                return

        # Append the new entry to the existing data
        peer_dict["whitelist"].append({"hash": hash_peer, "ip": ip, "port": port})

        col_integration_logger.info("Updated whitelist with added peer: %s", peer_dict)

        with open("resources/whitelist/whitelist.json", "w", encoding="utf-8") as f:
            json.dump(peer_dict, f)

        return

    def remove_from_whitelist(self, data):
        """
        This function removes the peer from the whitelist.
        """

        hash_peer = data["hash"]

        if not hash_peer:
            col_integration_logger.error(
                "Invalid data received for NODE_LEFT: %s", data
            )
            return

        try:
            with open("resources/whitelist/whitelist.json", "r", encoding="utf-8") as f:
                peer_dict = json.load(f)
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            print(f)
            print("Error decoding JSON")

        # Remove the entry from the whitelist
        peer_dict["whitelist"] = [
            peer for peer in peer_dict["whitelist"] if peer["hash"] != hash_peer
        ]

        col_integration_logger.info(
            "Updated whitelist with removed peer: %s", peer_dict
        )

        with open("resources/whitelist/whitelist.json", "w", encoding="utf-8") as f:
            json.dump(peer_dict, f)

        return

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

        elif message_type == "AUTO_NODE_JOINED":
            # Update the whitelist
            col_integration_logger.info("AUTO_NODE_JOINED: %s", data)
            self.add_to_whitelist(data)
            col_integration_logger.info("AUTO_NODE_JOINED and Finished")

        elif message_type == "NODE_JOINED":
            # Update the whitelist
            self.add_to_whitelist(data)
        elif message_type == "NODE_LEFT":
            # Remove the peer from the whitelist
            self.remove_from_whitelist(data)
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
        response = {"predictions": result}

        return response
