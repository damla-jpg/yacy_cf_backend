"""
Collaborative Filtering Implementation
This code in this section was taken from Krishna Shukla from the following repository:
https://gitlab.com/ucbooks/dht/-/tree/main?ref_type=heads
The code was modified to fit this project.
"""

# pylint: disable= C0301

# Default Libraries
import os
import math
import hashlib
import socket
import random
import errno
import selectors
import pickle
import json
import time
import threading


# Third Party Libraries
import requests
import numpy as np
import geocoder
from sklearn.linear_model import LinearRegression

# Custom Libraries
from birbs.col_filtering.helpers import (
    cosine_similarity,
    hex_distance,
    hex_different_index,
    hex_compare,
    generate_node_id_from_ip,
)
from birbs.col_filtering import b, hash_size, N, hex_map

# Numpy error handling for debugging
np.seterr(over="raise")

# Constants
MODE = "production"
METHOD = "biases"

JOIN_MESSAGE = "JOIN_PASTRY"
ADD_MESSAGE = "ADD_MESSAGE"
FAILED_NODE = "FAILED_NODE"
SEND_MODEL = "SEND_MODEL"


class COL:
    """
    Collaborative Filtering Class
    """

    def __init__(self, ip_address, port, position):
        # TODO: Most of these variables are not needed without the Pastry implementation
        # Initialize the variables
        self.node_id = generate_node_id_from_ip(
            str(position[0]) + "+" + str(position[1])
        )
        self.ip_address = ip_address
        self.port = int(port)
        self.position = position
        self.node_state = tuple(
            (self.position, self.node_id, self.ip_address, self.port)
        )
        self.r = [[None for j in range(pow(2, b))] for i in range(hash_size)]
        self.m = [None for x in range(pow(2, b + 1))]
        self.l_min = [None for x in range(pow(2, b - 1))]
        self.l_max = [None for x in range(pow(2, b - 1))]
        self.main_node = {"ip_address": "localhost", "port": 9090}
        self.overlay_interval = 20
        self.heartbeat_interval = 15
        self.pastry_nodes = []
        self.search_queries = []
        self.ht = {}
        self.delta = 15
        self.y = None  # The base model of the node
        self.received_y = []  # Storage for the received models
        self.xi = None  # Node latent factors
        self.bi = None  # Node biases
        self.k = 20
        self.lr = 0.001
        self.rp = 1
        self.p = []
        self.max_rec = 40
        self.method = METHOD
        self.lock = threading.Lock()

    def generate_hashes(self, qi):
        """
        Input: List of Dictionary with keys, query, hash and frequency
        Output: ai: Dictionary with the key being the hash and the value a rating
        """

        urls = {}
        for instance in qi:
            url = instance["query"]
            if url in urls:
                urls[url] += 1
            else:
                urls[url] = 0

        ai = {}
        for key, value in urls.items():
            search_hash = key.encode("utf-8").hex()
            ai[search_hash] = value

        return ai

    def normalize_ratings(self, ai: dict):
        """
        Function to Z score normalize the ratings.
        Input: ai: url hash to ratings
        Output: ai: url hash to normalized ratings
        """

        # Get the ratings and hashes
        ratings = list(ai.values())
        hashes = list(ai.keys())

        # Get the mean and standard deviation of the ratings
        mean_ratings = np.mean(ratings)
        std_ratings = np.std(ratings)

        # Normalize the ratings
        if std_ratings == 0:
            normalized_ratings = (ratings - mean_ratings) / std_ratings
        else:
            normalized_ratings = ratings

        # Initialize the new dictionary
        new_ai = {}

        # Update the dictionary with the normalized ratings
        for i, h in enumerate(hashes):
            # Update the dictionary with the normalized ratings
            new_ai[h] = normalized_ratings[i]

        # Return the new dictionary
        return new_ai

    def initialize_xibi(self, k: int):
        """
        Initialize the latent factors and biases
        """

        xi = np.random.normal(loc=0.0, scale=1.0, size=k).astype("float64")
        bi = np.random.normal(loc=0.0, scale=1.0)

        return xi, bi

    def initialize_model(self, ai: dict, k: int):
        """
        Input: ai: Dictionary with the key being the hash and the url a rating
        Output: Model of the format:
            {
                Y: {
                   hash_1: { w: -, age: -, ci: - },
                   ...
                   hash_n: { w: -, age: -, ci: - }
                },
                history: []
            }
        For more information about what these symbols mean, please read the paper.
        """

        temp_y = {}
        history = []

        # For every hash in the ai dictionary create a weight vector, age and confidence interval
        for h, _ in ai.items():
            temp_y[h] = {
                "w": np.random.normal(loc=0.0, scale=1.0, size=k),
                "ci": np.random.normal(loc=0.0, scale=1.0),
                "age": 0,
            }

        # return the parameters and clean history
        return (temp_y, history)

    def select_peer(self):
        """
        Random normally select a peer to send a model to
        Input: None
        Output: node: { "ip_address": ip_address, "port": port, "id_": id, "position": position }
        """

        # TODO: Fetch the list of peers in the WHITELIST and select a random peer
        return {}

    def lrmf_loop(self):
        """
        This function runs the LRMF algorithm.
        """

        # Initialize the search queries
        qi = self.search_queries

        # Generate the hashes and normalize the ratings
        ai = self.generate_hashes(qi)
        ai = self.normalize_ratings(ai)

        # Initialize the latent factors and biases
        self.xi, self.bi = self.initialize_xibi(self.k)
        self.y = self.initialize_model(ai, k=self.k)
        self.received_y.append(self.y)

        quiet = 0

        # Update the model and send it to a peer every delta seconds
        while True:
            # Wait for delta seconds
            time.sleep(self.delta)

            # If no models have been received, increment the quiet counter
            if len(self.received_y) == 0:
                quiet += 1

                # If the model has been quiet for 5 cycles, select a peer and send the model
                if quiet >= 5:
                    # Select a peer
                    p = self.select_peer()

                    # If a peer is selected, send the model
                    if p:
                        self.forward(
                            "SEND_MODEL",
                            (
                                (p["position"][0], p["position"][1]),
                                p["id_"],
                                p["ip_address"],
                                p["port"],
                                self.y,
                                self.node_state,
                            ),
                        )

            # If models have been received, update the model and send it to a peer
            else:
                # Reset the quiet counter
                quiet = 0

                # Update the model with the received models
                while len(self.received_y) > 0:
                    # Get the received model
                    y_hat = self.received_y[0]

                    # Update the model
                    self.received_y = self.received_y[1:]

                    # Select a peer to send the model to
                    p = self.select_peer()

                    # If a peer is selected, send the model
                    if p:
                        self.forward(
                            "SEND_MODEL",
                            (
                                (p["position"][0], p["position"][1]),
                                p["id_"],
                                p["ip_address"],
                                p["port"],
                                y_hat,
                                self.node_state,
                            ),
                        )

    def lrmf_run(self):
        """
        This function runs the LRMF thread
        """

        # Start the LRMF thread
        lrmf_thread = threading.Thread(target=self.lrmf_loop)

        # Set the thread as a daemon (will close when the main thread closes)
        lrmf_thread.daemon = True

        # Start the LRMF thread
        lrmf_thread.start()

    def forward(self, msg, data):
        """
        This function forwards messages to other nodes.
        Input:
            msg: The message to be sent (in this case command)
            data format:
                [0]:  position,
                [1]:  id_,
                [2]:  ip_address,
                [3]:  port,
                [4]:  Y,
                [5]:  nodeState
        """

        # TODO: This part might not be needed
        # If the message is a JOIN_MESSAGE, that means a new node is joining the network
        if msg == JOIN_MESSAGE:
            # Calculate the distance between the current node and the target node
            x = hex_different_index(data[1], self.node_id)

            # Create a socket connection
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Connect to the target node
            client_socket.connect((data[2], data[3]))

            # Form the message
            message = pickle.dumps({"message": "UPDATE_R", "x": x, "R[x]": self.r[x]})

            # Send the message
            for i in range(0, len(message), 1024):
                chunk = message[i : i + 1024]
                try:
                    # Send the message, blocks the thread until the message is sent
                    client_socket.sendall(chunk)
                except Exception as e:
                    print("forward error: ", e)
                    raise

            client_socket.close()

        # Call the __forward__ extension function
        self.__forward__(msg, data)

    def __forward__(self, msg, data):
        """
        Extension function for forwarding messages to other nodes.
        """

        # TODO: Ip address and port should be the target node's ip address and port
        ip = None
        port = None

        # Create a socket connection
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the target node
        client_socket.connect(ip, port)

        # TODO: Don't have to use pickle for this but sure
        # Form the message
        message = pickle.dumps({"message": "FORWARD", "msg": msg, "data": data})

        # Send the message
        for i in range(0, len(message), 1024):
            # Split the message into chunks of 1024 bytes
            chunk = message[i : i + 1024]

            try:
                # Send the message, blocks the thread until the message is sent
                client_socket.sendall(chunk)
            except Exception as e:
                print("__forward__ error: ", e)
                raise

        client_socket.close()
