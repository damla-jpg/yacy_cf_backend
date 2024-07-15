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

    url = f"http://localhost:{USERPORT}/yacy/seedlist.json"
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

    url = f"http://localhost:{USERPORT}/Network.xml"
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
    url = f"http://localhost:{USERPORT}/yacysearch.json?query={query}&resource=global&urlmaskfilter=.*&prefermaskfilter=&nav=all&startRecord={start_record}"
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

    url = f"http://localhost:{USERPORT}/Messages_p.html"
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

    url = f"http://localhost:{USERPORT}/Messages_p.xml"
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
    url = f"http://localhost:{USERPORT}/Messages_p.html?action=view&object={message_id}"
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
    url = f"http://localhost:{USERPORT}/MessageSend_p.html?hash={hash_}&subject={subject}&message={message}"
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
    if not hash or not ip or not port:
        return jsonify(error="Invalid hash, IP or port")
    else:
        # Create a json file in the resources/whitelist folder
        if not os.path.exists("resources/whitelist"):
            os.makedirs("resources/whitelist")

        try:
            with open("resources/whitelist/whitelist.json", "r", encoding="utf-8") as f:
                peer_dict = json.load(f)
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            print(f)
            print("Error decoding JSON")

        # Append the new entry to the existing data
        peer_dict["whitelist"].append({"hash": hash_peer, "ip": ip, "port": port})

        with open("resources/whitelist/whitelist.json", "w", encoding="utf-8") as f:
            json.dump(peer_dict, f)

        if COL_INTEGRATION is None:
            server_logger.error("COL_INTEGRATION not initialized, this is happening during whitelist creation.")
            return jsonify(error="Whitelist created but couldn't send message to the added peer. COL_INTEGRATION not initialized")
        
        if COL_INTEGRATION.col is None:
            server_logger.error("col not initialized, this is happening during whitelist creation.")
            return jsonify(error="Whitelist created but couldn't send message to the added peer. COL not initialized")
        
        try:
            # Current node info
            crr_ip = COL_INTEGRATION.col.ip_address
            # TODO: +100 to avoid port conflict. This is a temporary solution
            crr_port = COL_INTEGRATION.col.port + 100
            crr_hash = COL_INTEGRATION.col.node_id

            # Send a message to the added peer
            message = {
                "msg": "NODE_JOINED",
                "data": {"ip": crr_ip, "port": crr_port, "hash": crr_hash},
            }

            send_socket_message(ip, int(port), message)
        except Exception as e:
            server_logger.error("An error occurred while sending the message to the server: %s", e)
            return jsonify(error="Whitelist created but couldn't send message to the added peer. Error occurred while sending the message to the server ")

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
    else:
        try:
            with open("resources/whitelist/whitelist.json", "r", encoding="utf-8") as f:
                peer_dict = json.load(f)
        except json.JSONDecodeError:
            return jsonify(error="Error decoding JSON")

        # Delete the peer from the whitelist
        for peer in peer_dict["whitelist"]:
            if peer["hash"] == hash_peer:
                peer_dict["whitelist"].remove(peer)
                with open(
                    "resources/whitelist/whitelist.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(peer_dict, f)
                return jsonify(message="Peer deleted")

        return jsonify(error="Peer not found")


@app.route("/upload", methods=["POST"])
def upload():
    """
    This function uploads a file to the server.
    """

    if "files" not in fl.request.files:
        return jsonify(error="No files were uploaded")
    file = fl.request.files["files"]
    if file.filename == "":
        return jsonify(error="No files were uploaded")
    if file:
        if not os.path.exists("resources/history"):
            os.makedirs("history")
        file.save(os.path.join("resources/history", file.filename))
        return jsonify(message="File uploaded")
    else:
        return jsonify(error="Error uploading file")


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


@app.route("/api/share_history", methods=["POST"])
def share_history():
    """
    This function shares the history of the user.
    """
    server_logger.info("/api/share_history: sharing history...")

    # Get the ip and port from the headers
    ip = fl.request.headers.get("ip")
    port = fl.request.headers.get("port")

    # Check if the ip and port are valid
    if not ip or not port:
        server_logger.error("/api/share_history: Invalid IP or port.")
        return jsonify(error="Invalid IP or port")

    # Get the message from the parameters
    message = fl.request.args.get("message")

    server_logger.info(
        "/api/share_history: Sending message to %s:%s with message: %s",
        ip,
        port,
        message,
    )

    # Send the message to socket listener
    try:
        send_socket_message(ip, int(port), message)
        server_logger.info("/api/share_history: Message sent.")

        return jsonify(message="Message sent")
    except Exception as e:
        server_logger.error("/api/share_history: An error occurred: %s", e)

        return jsonify(error="Failed to send message")


@app.route("/test", methods=["GET"])
def test_col():
    """
    This function tests the COL.
    """

    return jsonify(message="COL started")


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


def start_server(
    yacy_settings: dict, host: str, port: int, col_int: cf.COLServerIntegration
):
    """
    This function starts the server.
    """

    # Check if the settings are valid
    if not yacy_settings or not host or not port:
        server_logger.error("Invalid settings.")
        return

    # Set the global variables
    global MY_DIGEST_USERNAME, MY_DIGEST_PASSWORD, USERPORT, COL_INTEGRATION
    MY_DIGEST_USERNAME = yacy_settings["username"]
    MY_DIGEST_PASSWORD = yacy_settings["password"]
    USERPORT = yacy_settings["port"]
    COL_INTEGRATION = col_int

    try:
        # Start the server
        app.run(host=host, port=port)
    except Exception as e:
        server_logger.error("An error occurred during starting the server: %s", e)
    except KeyboardInterrupt:
        server_logger.info("Server stopped.")
