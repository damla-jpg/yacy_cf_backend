"""
Collaborative Filtering Implementation
This code in this section was taken from Krishna Shukla from the following repository:
https://gitlab.com/ucbooks/dht/-/tree/main?ref_type=heads

The code was modified to fit this project.
"""

# pylint: disable= C0301, W0201, W0718, R0902, W0719

# Default Libraries
import os
import random
import pickle
import json
import time
import threading
import xml.etree.ElementTree as ET
import logging
import queue

# Third Party Libraries
import requests
from bs4 import BeautifulSoup
import numpy as np

# Custom Libraries
from birbs.communication import send_message as send_socket_message
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

B = 4
N = 50
HASH_SIZE = 16

# Initialize the logger
col_logger = logging.getLogger("COL")


class COL:
    """
    Collaborative Filtering Class
    """

    def __init__(self, ip_address, port, hash_):

        # Initialize the configuration
        self.config_loader = ConfigLoader()

        # Initialize the variables
        self.initialize_variables(ip_address, port, hash_)

    def initialize_variables(self, ip_address, port, hash_):
        """
        Initialize the variables
        """
        self.lock = threading.RLock()

        with self.lock:
            # Initialize the variables
            self.node_id = hash_
            self.ip_address = ip_address
            self.port = int(port)
            self.node_state = tuple((self.node_id, self.ip_address, self.port))
            self.r = [[None for j in range(pow(2, B))] for i in range(HASH_SIZE)]
            self.m = [None for x in range(pow(2, B + 1))]
            self.l_min = [None for x in range(pow(2, B - 1))]
            self.l_max = [None for x in range(pow(2, B - 1))]
            self.main_node = {"ip_address": "localhost", "port": 9090}
            self.overlay_interval = 20
            self.heartbeat_interval = 15
            self.pastry_nodes = []

            self.search_queries = self.load_history()

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

            self.queue = queue.Queue()

    def load_history(self, path="resources/history/history.json"):
        """
        Load the search history of the node
        """

        try:
            with open(path, "r", encoding="utf-8") as handler:
                history = json.load(handler)
        except FileNotFoundError:
            return

        return history["history"]

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

        with open(
            "resources/whitelist/whitelist.json", "r", encoding="utf-8"
        ) as handler:
            whitelist = json.load(handler)

        if whitelist:
            peer = random.choice(whitelist["whitelist"])
            return peer

        return None

    def create_dummy_model(self):
        """
        Create a dummy model for testing purposes.
        """

        with self.lock:
            # Initialize the search queries
            qi = self.load_history(path="resources/history/dummy_search_history_2.json")
            # Generate the hashes and normalize the ratings
            ai = self.generate_hashes(qi)
            ai = self.normalize_ratings(ai)

            # Initialize the latent factors and biases
            y = self.initialize_model(ai, k=self.k)
            return y

    def lrmf_loop(self):
        """
        This function runs the LRMF algorithm.
        """

        with self.lock:
            # Initialize the search queries
            qi = self.search_queries

            # Generate the hashes and normalize the ratings
            ai = self.generate_hashes(qi)
            ai = self.normalize_ratings(ai)

            # Initialize the latent factors and biases
            self.xi, self.bi = self.initialize_xibi(self.k)
            self.y = self.initialize_model(ai, k=self.k)
            self.received_y.append(self.y)

        # Initialize the quiet counter
        quiet = 0

        # Update the model and send it to a peer every delta seconds
        while True:
            # Wait for delta seconds
            time.sleep(self.delta)

            try:
                # Select a peer
                p = self.select_peer()

                # Initialize the payload
                payload = None

                if p is None:
                    col_logger.warning("No peers found")
                    continue

                col_logger.info("Selected peer: %s", p)

                with self.lock:
                    # If no models have been received, increment the quiet counter
                    if len(self.received_y) == 0:
                        # Increment the quiet counter
                        quiet += 1

                        col_logger.info("No models received, incrementing the quiet counter: %s", quiet)

                        # If the model has been quiet for 5 cycles, select a peer and send the model
                        if quiet >= 5:
                            col_logger.info("Sending the model to the peer: %s after %s quiet cycles", p, quiet)

                            # Construct the payload
                            payload = [
                                p["hash"],
                                p["ip"],
                                p["port"],
                                self.y,
                                self.node_state,
                            ]

                    # If models have been received, update the model and send it to a peer
                    else:
                        # Reset the quiet counter
                        quiet = 0

                        col_logger.info("Models received, updating the model")

                        # Update the model with the received models
                        while len(self.received_y) > 0:
                            # Get the received model
                            y_hat = self.received_y[0]

                            # Update the model
                            self.received_y = self.received_y[1:]

                            col_logger.info("Sending the received model to the peer: %s", p)

                            # Construct the payload
                            payload = [
                                p["hash"],
                                p["ip"],
                                p["port"],
                                y_hat,
                                self.node_state,
                            ]

                if payload is not None:
                    col_logger.info("Forwarding the payload")

                    self.forward(
                        "SEND_MODEL",
                        payload,
                    )
                else:
                    col_logger.info("No payload to forward")

            except Exception as e:
                col_logger.error("An error occurred during the LRMF loop: %s", e)

    def lrmf_run(self):
        """
        This function runs the LRMF thread
        """

        # Init the LRMF thread
        lrmf_thread = threading.Thread(target=self.lrmf_loop)

        # Set the thread as a daemon (will close when the main thread closes)
        lrmf_thread.daemon = True

        # Init the receive model thread
        receive_model_thread = threading.Thread(target=self.model_receive_loop)

        # Set the thread as a daemon (will close when the main thread closes)
        receive_model_thread.daemon = True

        only_listen = self.config_loader.debug_settings["only_listen"]
        
        # Start the LRMF thread
        if not only_listen:
            lrmf_thread.start()

        # Start the receive model thread
        receive_model_thread.start()

    def model_receive_loop(self):
        """
        This function calls the on_receive_model function from the queue while locking the model.
        """
        #TODO: REMOVE TIHS
        import traceback

        while True:
            time.sleep(self.delta)

            try:
                # Get the message from the queue
                recv_model = self.queue.get()

                with self.lock:
                    col_logger.info("Received a model from the queue")
                    col_logger.info("Calling the on_receive_model function")
                    col_logger.info("Model with type: %s and data: %s", type(recv_model), recv_model)

                    # Call the on_receive_model function
                    self.on_receive_model(recv_model)
            except Exception as e:
                col_logger.error(
                    "An error occurred during the model receive loop: %s", e
                )
                col_logger.error(traceback.format_exc())

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
        message = {"msg": msg, "data": data}

        # Fetch the ip and port
        ip = data[1]
        # Since the socket is running on a different port than yacy, we need to send it to a different port.
        # Normally, the socket would be running on the yacy, so the port would be the same as the yacy port.
        # However, for testing purposes, we are assuming that the socket is running on yacy port + 100
        port = int(data[2]) + 100

        # Send the message to the peer
        try:
            _ = send_socket_message(ip, int(port), message)
        except Exception as e:
            col_logger.error("An error occurred while sending message: %s", e)

    def predict(self, xi, ai, y, bi):
        """
        An array of predictions
        """
        local_hashes = list(ai.keys())

        # WHEN A NEW SEARCH IS MADE, RUN THE ALGORITHM
        # TAKE THE FIRST PREDICTION AND RETURN ITS LINKS
        # TECHNICALLY SHOULD BE THE LAST AI W

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

        with self.lock:
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
                g.append(y[0][each_hash]["w"])
                cis.append(y[0][each_hash]["ci"])
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
            xprime = (1 - lr * rp) * xprime + lr * eij * g[j]
            gprime[j] = (1 - lr * rp) * gprime[j] + lr * eij * xi
            biprime = (1 - lr * rp) * biprime + lr * eij
            cisprime[j] = (1 - lr * rp) * cisprime[j] + lr * eij

        # Normalize xprime, biprime by the number of ratings
        # xprime = xprime / len(all_ai)
        # biprime = biprime / len(all_ai)

        for each_hash in all_ai:
            y[0][each_hash]["age"] += 1
            y[0][each_hash]["w"] = gprime[all_ai.index(each_hash)]
            y[0][each_hash]["ci"] = cisprime[all_ai.index(each_hash)]

        return y, xprime, biprime

    def receive_models(self):
        """
        Function to retrieve all received models.
        """

        with self.lock:
            # Retrieving message ids
            query = f"http://localhost:{self.config_loader.flask_settings['port']}/api/retrieve_message_ids"
            document = requests.get(query, timeout=60).text

            xml_root = ET.fromstring(document)
            message_ids = [
                {"id": message.get("id")} for message in xml_root.findall(".//message")
            ]

            all_models = {}

            # Retrieving messages
            for message in message_ids:
                query = f"http://localhost:{self.config_loader.flask_settings['port']}/api/get_message_contents?messageId={message['id']}"
                document = BeautifulSoup(
                    requests.get(query, timeout=60).text, "html.parser"
                )
                pairs = document.find(class_="pairs")
                if pairs:
                    message_data = pairs.find_all("dd")
                    from_ = message_data[0].get_text()
                    subject = message_data[3].get_text()
                    message = message_data[4].get_text()

                    # Unpickle the message
                    # message is a string that is pickled
                    # get rid of empty spaces and new lines
                    message = message.replace(" ", "").replace("\n", "")
                    message = pickle.loads(message, encoding="ASCII")

                    # Add the message to the dictionary
                    if subject == "SEND_MODEL":
                        all_models[from_] = message

        return all_models

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

        # inc values are the target node's values a.k.a me :D
        inc_hash = model[0]
        inc_ip = model[1]
        inc_port = model[2]
        model = model[3]

        with self.lock:
            ############################ UPDATE MY OWN MODEL WITH MY NEW SEARCHES ############################

            # Fetch the search history
            qi = self.search_queries

            # Generate the hashes and normalize the ratings
            ai = self.generate_hashes(qi)
            ai = self.normalize_ratings(ai)

            # Find the newly added keys
            # !! we use index 0 because the first index is the model itself, the second index is the history
            model_keys = list(self.y[0].keys())
            new_keys = {}

            for key, _ in ai.items():
                if key not in model_keys:
                    new_keys[key] = ai[key]

            # Initialize the model with the new keys
            y_new = self.initialize_model(new_keys, self.k)

            # Merge the models
            y = self.merge_models(self.y, y_new)

            ############################ UPDATE MY OWN MODEL WITH THE RECEIVED MODEL ############################

            # Merge the new model with the received model
            y = self.merge_models(y, model)

            # Update the model with the bias
            y, xi, bi = self.update_model_with_bias(
                y, self.xi, ai, self.bi, self.k, self.lr, self.rp
            )

            # Normalize the latent factors by the length of the ratings
            # This is because in update step we repeat the multiplication len(ai) times
            xi = xi / len(list(ai.keys()))
            bi = bi / len(list(ai.keys()))

            # Make predictions
            predictions = self.predict(xi, ai, y, bi)

            # If predictions are made assign them to the class variable
            if predictions:
                self.p = predictions

            # Update the class variables
            self.y = y
            self.xi = xi
            self.bi = bi

            # Add the model to the received models. This will be used in LSFM_Loop
            self.received_y.append(y)

            # Calculate the error
            error = self.error_bias(y, xi, ai, bi, self.k, self.rp)

            ########################################## SAVE THE RESULTS ##########################################

            # Form the path
            results_path = os.path.join("results", self.node_id + ".json")

            if not os.path.exists(results_path):
                # create directory
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w+", encoding="utf-8") as handler:
                    handler.write(json.dumps({}))

            with open(results_path, "r", encoding="utf-8") as handler:

                file_content = handler.read().strip()

                node_results = json.loads(file_content)

            if self.method in node_results:
                node_results[self.method].append(error)
            else:
                node_results[self.method] = [error]

            with open(results_path, "w+", encoding="utf-8") as handler:
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

        historyk = y[1]

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

        return (y_k, historyk)

    # start function
    def start(self):
        """
        Function to start the Collaborative Filtering node.
        """

        self.lrmf_run()
