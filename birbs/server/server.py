"""
This file contains the server implementation for the Birbs project.
"""

# pylint: disable=C0301, W0718, W0603

# Default imports
import os
import logging
import json
import pickle
import timeit

# Third-party imports
import flask as fl
import requests as req
from flask import Flask, jsonify
from flask_cors import CORS
from flask_httpauth import HTTPDigestAuth

# Custom imports
from birbs.communication import send_message as send_socket_message
import birbs.server.col_server_integration as cf

# Initialize the Flask app
app = Flask(__name__)
CORS(app)
auth = HTTPDigestAuth()

# Constants
MY_DIGEST_USERNAME = None
MY_DIGEST_PASSWORD = None
USERPORT = None
COL_INTEGRATION = None
YACY_SERVICE = None
server_logger = logging.getLogger("Server")

############################################################################################################
#                                           Frontend Routes                                                #
############################################################################################################


@app.route("/")
def home():
    """
    Home route, returns a simple message.
    """
    return jsonify(message="Hello, World!")


@app.route("/getPeers", methods=["GET"])
def get_peers():
    """
    This function returns the list of peers in the network.
    """

    url = f"http://{YACY_SERVICE}:{USERPORT}/yacy/seedlist.json"
    response = req.get(url, timeout=60)
    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify(error="Failed to get peers")


@app.route("/profile", methods=["GET"])
def get_profile():
    """
    This function returns the profile of the user.
    """

    url = f"http://{YACY_SERVICE}:{USERPORT}/Network.xml"
    response = req.get(url, timeout=60)
    if response.status_code == 200:
        return response.text
    else:
        return jsonify(error="Failed to get profile")


@app.route("/api/get_click", methods=["POST"])
def get_history():
    """
    This function saves the history from the request body sent by the user.
    The  format of the body is:
    {
        "query": "hello",
        "link": "link"
    }
    The format of the file to be appended is:
    {
        "history": [
            {
                "query": "hello",
                "results": [
                    {
                        "link": "link"
                        "frequency": 1
                    }
                ]
                "total_frequency": 1
            }
        ]
    }
    """

    # Get the history from the request body
    history = fl.request.json

    # Check if the history is valid
    if not history:
        return jsonify(error="Invalid history")

    # Create a json file in the resources/history folder
    if not os.path.exists("resources/history"):
        os.makedirs("resources/history")

    try:
        with open("resources/history/history.json", "r", encoding="utf-8") as f:
            history_dict = json.load(f)
    except FileNotFoundError:
        history_dict = {"history": []}
    except json.JSONDecodeError:
        return jsonify(error="Error decoding JSON")

    # if the query is already in the history, update the results
    for entry in history_dict["history"]:

        if entry["query"] == history["query"]:

            entry["total_frequency"] += 1
            for result in entry["results"]:

                if result["link"] == history["link"]:
                    result["frequency"] += 1
                    with open(
                        "resources/history/history.json", "w", encoding="utf-8"
                    ) as f:
                        json.dump(history_dict, f)
                    return jsonify(message="History updated")

                entry["results"].append({"link": history["link"], "frequency": 1})
                with open("resources/history/history.json", "w", encoding="utf-8") as f:
                    json.dump(history_dict, f)
                return jsonify(message="History updated")

            with open("resources/history/history.json", "w", encoding="utf-8") as f:
                json.dump(history_dict, f)
            return jsonify(message="History updated")

    with open("resources/history/history.json", "w", encoding="utf-8") as f:
        history_dict["history"].append(
            {
                "query": history["query"],
                "results": [{"link": history["link"], "frequency": 1}],
                "total_frequency": 1,
            }
        )
        json.dump(history_dict, f)

    return jsonify(message="History saved")


@app.route("/search", methods=["GET"])
def search():
    """
    This function searches for a query in the network.
    """

    query = fl.request.args.get("query")
    start_record = fl.request.args.get("startRecord")
    url = f"http://{YACY_SERVICE}:{USERPORT}/yacysearch.json?query={query}&resource=global&urlmaskfilter=.*&prefermaskfilter=&nav=all&startRecord={start_record}"
    response = req.get(url, timeout=60)

    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify(error="Failed to search")


@app.route("/api/get_contact_list", methods=["GET"])
def get_contact_list():
    """
    This function returns the contact list of the user.
    """

    url = f"http://{YACY_SERVICE}:{USERPORT}/Messages_p.html"
    response = req.get(
        url,
        auth=req.auth.HTTPDigestAuth(MY_DIGEST_USERNAME, MY_DIGEST_PASSWORD),
        timeout=60,
    )
    if response.status_code == 200:
        return response.text
    else:
        return jsonify(error="Failed to get contact list")


@app.route("/api/retrieve_message_ids", methods=["GET"])
def retrieve_message_ids():
    """
    This function retrieves the message ids.
    """

    url = f"http://{YACY_SERVICE}:{USERPORT}/Messages_p.xml"
    response = req.get(
        url,
        auth=req.auth.HTTPDigestAuth(MY_DIGEST_USERNAME, MY_DIGEST_PASSWORD),
        timeout=60,
    )
    if response.status_code == 200:
        return response.text
    else:
        return jsonify(error="Failed to retrieve message ids")


@app.route("/api/get_message_contents", methods=["GET"])
def retrieve_message_contents():
    """
    This function retrieves the message contents.
    """

    message_id = fl.request.args.get("messageId")
    url = f"http://{YACY_SERVICE}:{USERPORT}/Messages_p.html?action=view&object={message_id}"
    response = req.get(
        url,
        auth=req.auth.HTTPDigestAuth(MY_DIGEST_USERNAME, MY_DIGEST_PASSWORD),
        timeout=60,
    )
    if response.status_code == 200:
        return response.text
    else:
        return jsonify(error="Failed to retrieve message contents")


@app.route("/api/send_message", methods=["POST"])
def send_message():
    """
    This function sends a message to a user.
    """

    hash_ = fl.request.args.get("hash")
    subject = fl.request.args.get("subject")
    message = fl.request.args.get("message")
    url = f"http://{YACY_SERVICE}:{USERPORT}/MessageSend_p.html?hash={hash_}&subject={subject}&message={message}"
    # print('url:', url)
    response = req.post(
        url,
        auth=req.auth.HTTPDigestAuth(MY_DIGEST_USERNAME, MY_DIGEST_PASSWORD),
        timeout=60,
    )
    if response.status_code == 200:
        return jsonify(response.text)
    else:
        return jsonify(error="Failed to send message")


@app.route("/api/create_whitelist", methods=["POST"])
def create_whitelist():
    """
    This function creates a whitelist.
    """

    # get hash, ip and port from the headers
    hash_peer = fl.request.args.get("hash")
    ip = fl.request.args.get("ip")
    port = fl.request.args.get("port")
    peer_dict = {"whitelist": []}

    # Check if the hash, ip and port are valid
    if not hash_peer or not ip or not port:
        return jsonify(error="Invalid hash, IP or port")

    # Create a json file in the resources/whitelist folder
    if not os.path.exists("resources/whitelist"):
        os.makedirs("resources/whitelist")

    try:
        with open("resources/whitelist/whitelist.json", "r", encoding="utf-8") as f:
            peer_dict = json.load(f)
    except json.JSONDecodeError:
        server_logger.error("Error decoding JSON")

    for peer in peer_dict["whitelist"]:
        if peer["hash"] == hash_peer:
            return

    # Append the new entry to the existing data
    peer_dict["whitelist"].append({"hash": hash_peer, "ip": ip, "port": port})

    with open("resources/whitelist/whitelist.json", "w", encoding="utf-8") as f:
        json.dump(peer_dict, f)

    if COL_INTEGRATION is None:
        server_logger.error(
            "COL_INTEGRATION not initialized, this is happening during whitelist creation."
        )
        return jsonify(
            error="Whitelist created but couldn't send message to the added peer. COL_INTEGRATION not initialized"
        )

    if COL_INTEGRATION.col is None:
        server_logger.error(
            "col not initialized, this is happening during whitelist creation."
        )
        return jsonify(
            error="Whitelist created but couldn't send message to the added peer. COL not initialized"
        )

    try:
        # Current node info
        crr_ip = COL_INTEGRATION.col.ip_address
        crr_port = COL_INTEGRATION.col.port
        crr_hash = COL_INTEGRATION.col.node_id

        # Send a message to the added peer
        message = {
            "msg": "NODE_JOINED",
            "data": {"ip": crr_ip, "port": crr_port, "hash": crr_hash},
        }

        # TODO: +100 to avoid port conflict. This is a temporary solution
        send_socket_message(ip, int(port) + 100, message)
    except Exception as e:
        server_logger.error(
            "An error occurred while sending the message to the server: %s", e
        )
        return jsonify(
            error="Whitelist created but couldn't send message to the added peer. Error occurred while sending the message to the server "
        )

    return jsonify(message="Whitelist created")


@app.route("/api/get_whitelist", methods=["GET"])
def get_whitelist():
    """
    This function returns the whitelist.
    """

    try:
        with open("resources/whitelist/whitelist.json", "r", encoding="utf-8") as f:
            peer_dict = json.load(f)
            return jsonify(peer_dict)
    except FileNotFoundError:
        return jsonify(error="Whitelist not found")
    except json.JSONDecodeError:
        return jsonify(error="Error decoding JSON")


@app.route("/api/delete_peer_from_whitelist", methods=["DELETE"])
def delete_peer_from_whitelist():
    """
    This function deletes a peer from the whitelist.
    """

    # Get the hash from the parameters
    hash_peer = fl.request.args.get("hash")
    peer_dict = {"whitelist": []}

    # Check if the hash is valid
    if not hash_peer:
        return jsonify(error="Invalid hash")

    try:
        with open("resources/whitelist/whitelist.json", "r", encoding="utf-8") as f:
            peer_dict = json.load(f)
    except json.JSONDecodeError:
        return jsonify(error="Error decoding JSON")

    if COL_INTEGRATION is None:
        server_logger.error(
            "COL_INTEGRATION not initialized, this is happening during whitelist creation."
        )
        return jsonify(
            error="Whitelist read but couldn't send message to the added peer. COL_INTEGRATION not initialized"
        )

    if COL_INTEGRATION.col is None:
        server_logger.error(
            "col not initialized, this is happening during whitelist creation."
        )
        return jsonify(
            error="Whitelist read but couldn't send message to the added peer. COL not initialized"
        )

    peer_found = False
    # Delete the peer from the whitelist
    for peer in peer_dict["whitelist"]:
        if peer["hash"] == hash_peer:
            peer_found = True
            server_logger.info("Peer found to delete: %s", peer)
            try:
                crr_ip = COL_INTEGRATION.col.ip_address
                crr_port = COL_INTEGRATION.col.port
                crr_hash = COL_INTEGRATION.col.node_id

                # Send a message to the deleted peer
                message = {
                    "msg": "NODE_LEFT",
                    "data": {"ip": crr_ip, "port": crr_port, "hash": crr_hash},
                }

                # TODO: +100 to avoid port conflict. This is a temporary solution
                send_socket_message(peer["ip"], int(peer["port"]) + 100, message)
            except Exception as e:
                server_logger.error(
                    "An error occurred while sending the message to the server: %s", e
                )
                return jsonify(
                    error="Peer found to delete but couldn't send message to the added peer. Error occurred while sending the message to the server "
                )

            peer_dict["whitelist"].remove(peer)
            with open("resources/whitelist/whitelist.json", "w", encoding="utf-8") as f:
                json.dump(peer_dict, f)
            return jsonify(message="Peer deleted")

    if not peer_found:
        server_logger.error("Peer not found to delete: %s", hash_peer)

    return jsonify(error="Peer not found")


############################################################################################################
#                                           COL Integration                                                #
############################################################################################################


@app.route("/api/fetch_predictions", methods=["GET"])
def fetch_predictions():
    """
    This function fetches the predictions from the COL.
    """

    if COL_INTEGRATION:
        predictions = COL_INTEGRATION.fetch_predictions()
        return jsonify(predictions)
    else:
        return jsonify(error="COL not initialized")


@app.route("/api/test_backend_to_socket", methods=["POST"])
def test_backend_to_socket():
    """
    This function is to test the backend communication with the socket listener.
    """

    server_logger.info("Testing backend to socket communication...")

    # Get the message from the parameters
    message = fl.request.args.get("message")

    # Send the message to socket listener
    server_logger.info("Sending message to socket listener...")

    # Get socket info
    ip = os.getenv("SOCKET_LISTENER_HOST", "0.0.0.0")
    port = os.getenv("SOCKET_LISTENER_PORT", "3002")

    try:
        send_socket_message(ip, int(port), message)
        server_logger.info("/api/share_history: Message sent.")

        return jsonify(message="Message sent")
    except Exception as e:
        server_logger.error("/api/share_history: An error occurred: %s", e)

        return jsonify(error="Failed to send message")


@app.route("/api/receive_model", methods=["POST"])
def receive_model():
    """
    This function receives the model.
    """

    server_logger.info("Receiving model from listener...")

    # Get the model from the request body, the model is a pickle
    model = fl.request.data

    model = pickle.loads(model)

    # Check if the model is valid
    if not model:
        server_logger.error(
            "Couldn't receive the model from the request (This request is sent from listener.py). Model: %s",
            model,
        )
        return jsonify(error="Invalid model")

    server_logger.info("Model received")

    if COL_INTEGRATION:
        server_logger.info("Forwarding the received model to COL integration...")
        COL_INTEGRATION.handle_received_message(model)

    server_logger.info("Model Forwarded")
    return jsonify(message="Model received")


@app.route("/api/is_senior", methods=["GET"])
def is_senior():
    """
    This function checks if the peer is a senior peer.
    """

    return jsonify('{"is_senior": true}')


@app.route("/api/auto_whitelist", methods=["GET"])
def auto_whitelist():
    """
    This function automatically adds a peer to the whitelist. FOR DEMO PURPOSES ONLY
    """

    server_logger.info("Auto whitelist called")

    auto_wl = os.getenv("DEBUG_AUTO_WHITELIST", None)
    if auto_wl is None or int(auto_wl) == 0:
        server_logger.error("Auto whitelist not enabled")
        return jsonify(error="Auto whitelist not enabled")

    if COL_INTEGRATION is None:
        server_logger.error("COL not initialized")
        return jsonify(error="COL not initialized")

    server_logger.info("Auto whitelist enabled")

    yacy_info = COL_INTEGRATION.yacy_info

    if yacy_info is None:
        server_logger.error("YACY info not found")
        return jsonify(error="YACY info not found")

    # Current node info
    crr_ip = COL_INTEGRATION.col.ip_address
    crr_port = COL_INTEGRATION.col.port
    crr_hash = COL_INTEGRATION.col.node_id

    # Send a message to the added peer
    message = {
        "msg": "AUTO_NODE_JOINED",
        "data": {"ip": crr_ip, "port": crr_port, "hash": crr_hash},
    }

    # Get the number of nodes
    num_nodes = os.getenv("DEBUG_AUTO_WHITELIST_NUM_NODES")

    if num_nodes is None:
        server_logger.error("Number of nodes not found")
        return jsonify(error="Number of nodes not found")

    # This is the initial port number for docker testing. Whenever, we run the peer_sim, the first node is created on port 8092
    inital_port = 8092
    # This is for local testing with docker. See yacy_cf_peer_sim for more info
    # Every node created in peer_sim has a port number that is 1 more than the previous node (since the
    # socket is running on + 100 we do + 101)
    try:
        for i in range(int(num_nodes)):
            yacy_port = inital_port + i

            if yacy_port == crr_port:
                continue

            send_socket_message(crr_ip, int(yacy_port) + 100, message)

    except Exception as e:
        server_logger.error(
            "An error occurred while sending the message to the server: %s", e
        )
        return jsonify(
            error="Whitelist created but couldn't send message to the added peer. Error occurred while sending the message to the server "
        )

    server_logger.info("Auto whitelist created")
    return jsonify(message="Auto whitelist created")


def start_server(host: str, port: int, col_int: cf.COLServerIntegration):
    """
    This function starts the server.
    """

    # Check if the settings are valid
    if not host or not port:
        server_logger.error("Invalid settings.")
        return

    # Set the global variables
    global MY_DIGEST_USERNAME, MY_DIGEST_PASSWORD, USERPORT, COL_INTEGRATION, YACY_SERVICE
    MY_DIGEST_USERNAME = os.getenv("YACY_USERNAME", "admin")
    MY_DIGEST_PASSWORD = os.getenv("YACY_PASSWORD", "admin")
    USERPORT = os.getenv("YACY_PORT", "8090")
    YACY_SERVICE = os.getenv("YACY_SERVICE", "localhost")
    COL_INTEGRATION = col_int

    try:
        # Start the server
        app.run(host=host, port=port)
    except Exception as e:
        server_logger.error("An error occurred during starting the server: %s", e)
    except KeyboardInterrupt:
        server_logger.info("Server stopped.")
