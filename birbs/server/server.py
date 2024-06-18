# Default imports
import os
import sys
import logging

# Third party imports
import flask as fl
import requests as req

from flask import Flask, jsonify

server_logger = logging.getLogger("Server")
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify(message="Hello, World!")

def start_server(host : str, port : int):
    try:
        app.run(host=host, port=port)
    except Exception as e:
        server_logger.error(f"An error occurred during starting the server: {e}")
    except KeyboardInterrupt:
        server_logger.info("Server stopped.")