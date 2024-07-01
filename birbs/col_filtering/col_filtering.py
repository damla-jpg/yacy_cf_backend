"""
Collaborative Filtering Implementation
This code in this section was taken from Krishna Shukla from the following repository:
https://gitlab.com/ucbooks/dht/-/tree/main?ref_type=heads
The code was modified to fit this project.
"""

# pylint: disable= C0301

# Default Libraries
import os
import sys
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
import birbs.col_filtering.helpers as helpers
import birbs.server.server as server
from birbs.config import ConfigLoader

# Numpy error handling for debugging
np.seterr(over="raise")

# Constants
MODE = "production"
METHOD = "biases"

JOIN_MESSAGE = "JOIN_PASTRY"
ADD_MESSAGE = "ADD_MESSAGE"
FAILED_NODE = "FAILED_NODE"
SEND_MODEL = "SEND_MODEL"

b = 4
N = 50
hash_size = 16
hex_map = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'a': 10, 'b': 11, 'c': 12, 'd': 13,
    'e': 14, 'f': 15
}


class COL:
    """
    Collaborative Filtering Class
    """

    def __init__(self, ip_address, port, hash_):
        
        # Initialize the configuration
        self.config_loader = ConfigLoader()

        # TODO: Most of these variables are not needed without the Pastry implementation
        # Initialize the variables
        self.node_id = hash_
        self.ip_address = ip_address
        self.port = int(port)
        self.node_state = tuple((self.node_id, self.ip_address, self.port))
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
        self.delta = 5
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

    def load_history(self):
        """
        Load the search history of the node
        """

        try: 
            with open("resources/history/history.json", "r", encoding="utf-8") as handler:
                history = json.load(handler)
                self.search_queries = history["history"]
        except FileNotFoundError:
            print("No search history found")
            return
        

        self.search_queries = history["history"]

    def generate_hashes(self, qi):
        """
        Input: List of Dictionary with keys, query, hash and frequency
        Output: ai: Dictionary with the key being the hash and the value a rating
        """

        urls = {}
        for instance in qi:
            urls[instance["query"]] = instance["total_frequency"]

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
        Output: node: { "ip_address": ip_address, "port": port, "hash": id}
        """
        
        with open("resources/whitelist/whitelist.json", "r", encoding="utf-8") as handler:
            whitelist = json.load(handler)
        
        if whitelist:
            peer = random.choice(whitelist["whitelist"])
            return peer
        
        print("No peers in the whitelist")
        return None

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

                    print(f"Selected peer: {p}")

                    # If a peer is selected, send the model
                    if p:
                        self.forward(
                            "SEND_MODEL",
                            (
                                p["hash"],
                                p["ip"],
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
                            [
                                p["hash"],
                                p["ip"],
                                p["port"],
                                y_hat,
                                self.node_state
                            ]
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
                [1]:  hash,
                [2]:  ip_address,
                [3]:  port,
                [4]:  Y,
                [5]:  nodeState
        """

        # Form the message
        message_pure = {"msg": msg, "data": data}
        query = f"http://localhost:{self.config_loader.flask_settings['port']}/api/send_message?hash={data[0]}&subject={message_pure["msg"]}&message={message_pure["data"]}"

        print(f"Forwarding message: {query}")
        print(f"Data: {data}")
        # convert bytes to megabytes
        print(f"Data size: {sys.getsizeof(data) / 1024 / 1024} MB")
        try:
            response = requests.post(query)
            print(response)
        except requests.exceptions.RequestException as e:
            print(e)


    def drift_handler(self, y, xi, ai, bi, k, rp):
        """
        Function to check whether concept drift occured.
        Input: Y: model, xi: user latent factors, ai: website ratings, bi: user bias
        Output: Model (after checking for concept drift
        """

        e_hat = self.error_bias(y[0], xi, ai, bi, k, rp)
        # print("The error is: ", e_hat)

        y[1].append(e_hat)

        if len(y[1]) > 0 and self.drift_occured(y[1]):
            # print("drift occured initializing new model")
            # print(Y[1])
            y = self.initialize_model(ai, k)

        return y

    def drift_occured(self, model_history):
        """
        Function to check whether concept drift occured.
        Input: model_history: an array of errors
        Output: Boolean
        """

        # Smooth the error rates by applying sliding window based averaging
        window_size = max(
            len(model_history) // 2, 1
        )  # Ensure window size is at least 1
        # Perform sliding window-based averaging
        if len(model_history) > 1:
            for i in range(len(model_history) - window_size + 1):
                window = model_history[i : i + window_size]
                average_value = sum(window) / window_size
                # Replace the values in the window with the calculated average
                model_history[i : i + window_size] = [average_value] * window_size

        # We then apply linear regression
        model_history = np.array(model_history)
        X = model_history.reshape(-1, 1)
        model_ = LinearRegression()
        model_.fit(X, model_history)
        slope = model_.coef_[0]  # we use this slope
        intercept = model_.intercept_

        random_value = random.uniform(0, 1)
        sigmoid_value = 1 / (1 + np.exp(-20 * (slope - 0.5)))

        if random_value < sigmoid_value:
            return True

        return False

    def predict(self, xi, ai, y, bi):
        """
        An array of predictions
        """
        local_hashes = list(ai.keys())

        url_hash = []
        weights = []
        cis = []
        for hash_ in y[0]:
            url_hash.append(hash_)
            weights.append(y[0][hash_]["w"])
            cis.append(y[0][hash_]["ci"])

        predictions = np.matmul(xi, np.array(weights).T) - bi - cis
        indexes = np.argsort(predictions)[::-1]
        p = []
        for index in indexes:
            p.append(url_hash[index])

        nonlocal_hashes = []
        for each_hash in p:
            if each_hash not in local_hashes:
                nonlocal_hashes.append(bytes.fromhex(each_hash).decode("utf-8"))
        if len(nonlocal_hashes) > self.max_rec:
            nonlocal_hashes = nonlocal_hashes[0 : self.max_rec]

        return nonlocal_hashes
    
    def update_model_with_bias(self, y, xi, ai, bi, k, lr, rp):

        """
        Update the latent factors using the bias update rule.
        We update Y[0] (the model), ci in the model, xi and bi

        ai: Dictionary with the key being the hash and the url a rating.
        Y: Contains element to item latent factors each of size k.
        xi: User latent factors
        """

        all_ai = list(ai.keys())

        ratings = []
        g = []
        cis = []
        for each_hash in all_ai:
            ratings.append(ai[each_hash])
            if each_hash in y[0]:
                g.append(y[0][each_hash]['w'])
                cis.append(y[0][each_hash]['ci'])
            else:
                g.append(np.random.normal(loc=0.0, scale=1.0, size=k))
                cis.append(np.random.normal(loc=0.0, scale=1.0))

        ratings = np.array(ratings, dtype="float64")
        g = np.array(g, dtype="float64")
        cis = np.array(cis, dtype="float64")

        xprime = xi.copy()
        gprime = g.copy()
        biprime = bi
        cisprime = cis.copy()

        for j in range(0, len(all_ai)):
            rating = ratings[j]
            eij = rating - np.matmul(xi, g[j].T) - bi - cis[j]
            xprime = (1-lr*rp) * xprime + lr * eij * g[j]
            gprime[j] = (1-lr*rp) * gprime[j] + lr * eij * xi
            biprime = (1-lr*rp) * biprime + lr * eij
            cisprime[j] = (1-lr*rp) * cisprime[j] + lr * eij

        # Normalize xprime, biprime by the number of ratings
        # xprime = xprime / len(all_ai)
        # biprime = biprime / len(all_ai)

        for each_hash in all_ai:
            y[0][each_hash]['age'] += 1
            y[0][each_hash]['w'] = gprime[all_ai.index(each_hash)]
            y[0][each_hash]['ci'] = cisprime[all_ai.index(each_hash)]

        return y, xprime, biprime

    def on_receive_model(self, model):
        """
        Function to execute when the model is received.
        Input: model: Model of the format:
            {
                Y: {
                   hash_1: { w: -, age: -, ci: - },
                   ...
                   hash_n: { w: -, age: -, ci: - }
                },
                history: []
            }
        """

        qi = self.search_queries
        ai = self.generate_hashes(qi)
        ai = self.normalize_ratings(ai)
        model_keys = list(self.y[0].keys())
        new_keys = {}

        for key, value in ai.items():
            if key not in model_keys:
                new_keys[key] = ai[key]

        updated_y = self.initialize_model(new_keys, self.k)

        ############ WARNING: THIS MIGHT BE A BUG ##############
        self.merge_models(self.merge_models(self.y, updated_y), model)

        y, xi, bi = self.update_model_with_bias(
            self.y, self.xi, ai, self.bi, self.k, self.lr, self.rp
        )

        # Normalize the latent factors by the length of the ratings
        # This is because in update step we repeat the multiplication len(ai) times
        xi = xi / len(list(ai.keys()))
        bi = bi / len(list(ai.keys()))

        # How do you handle predictions?
        predictions = self.predict(xi, ai, y, bi)
        if predictions:
            self.p = predictions

        self.y = y
        self.xi = xi
        self.bi = bi
        self.received_y.append(y)
        error = self.error_bias(y, xi, ai, bi, self.k, self.rp)

        print("The error is: ", error)

        RESULTS_PATH = os.path.join("results", self.node_id + ".json")

        with self.lock:
            if not os.path.exists(RESULTS_PATH):
                with open(RESULTS_PATH, "w+") as handler:
                    handler.write(json.dumps({}))

            with open(RESULTS_PATH, "r") as handler:

                file_content = handler.read().strip()

                node_results = json.loads(file_content)

            if self.method in node_results:
                node_results[self.method].append(error)
            else:
                node_results[self.method] = [error]

            with open(RESULTS_PATH, "w+") as handler:
                handler.write(json.dumps(node_results))
                handler.flush()

    def error_bias(self, y, xi, ai, bi, k, rp):
        """
        Calculate the error of a model.
        Input: Y: model, xi: user latent factors, ai: ratings, bi:
        """

        all_ai = list(ai.keys())

        ratings = []
        g = []
        cis = []
        for each_hash in all_ai:
            ratings.append(ai[each_hash])
            if each_hash in y[0]:
                g.append(y[0][each_hash]["w"])
                cis.append(y[0][each_hash]["ci"])
            else:
                g.append(np.random.normal(loc=0.0, scale=1.0, size=k))
                cis.append(np.random.normal(loc=0.0, scale=1.0))

        ratings = np.array(ratings, dtype="float64")
        g = np.array(g, dtype="float64")
        cis = np.array(cis, dtype="float64")

        predictions = np.matmul(xi, g.T) - bi - cis

        data_fit_term = 0.5 * np.sum((ratings - predictions) ** 2)

        regularization_term = (
            0.5
            * rp
            * (
                np.linalg.norm(xi) ** 2
                + np.linalg.norm(g, "fro") ** 2
                + abs(bi) ** 2
                + np.linalg.norm(cis) ** 2
            )
        )

        error = data_fit_term + regularization_term

        return error

    def merge_models(self, y, y_hat):
        """
        Function to merge the incoming model into the one that is already present.
        y: local model
        y_hat: incoming model
        """

        u = list(set(list(y[0].keys()) + list(y_hat[0].keys())))

        y_k = {}
        for element in u:
            y_k[element] = {"age": 0, "w": None, "ci": None}

        for j in u:
            if j in y[0] and j in y_hat[0]:
                if y_hat[0][j]["age"] != 0:
                    w = y_hat[0][j]["age"] / (y[0][j]["age"] + y_hat[0][j]["age"])
                    y_k[j]["age"] = max(y_hat[0][j]["age"], y[0][j]["age"])
                    y_k[j]["w"] = (1 - w) * y[0][j]["w"] + w * y_hat[0][j]["w"]
                    y_k[j]["ci"] = (1 - w) * y[0][j]["ci"] + w * y_hat[0][j]["ci"]
                else:
                    y_k[j]["age"] = y[0][j]["age"]
                    y_k[j]["w"] = y[0][j]["w"]
                    y_k[j]["ci"] = y[0][j]["ci"]
            else:
                if j in y[0]:
                    y_k[j]["age"] = y[0][j]["age"]
                    y_k[j]["w"] = y[0][j]["w"]
                    y_k[j]["ci"] = y[0][j]["ci"]
                else:
                    y_k[j]["age"] = y_hat[0][j]["age"]
                    y_k[j]["w"] = y_hat[0][j]["w"]
                    y_k[j]["ci"] = y_hat[0][j]["ci"]

    #start function
    def start(self):
        """
        Function to start the Collaborative Filtering node.
        """

        # Initialize the Collaborative Filtering node
        self.load_history()

        self.lrmf_run()