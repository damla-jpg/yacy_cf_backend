# Default imports
import os
import sys

# Third party imports
import flask as fl
import requests as req

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify(message="Hello, World!")