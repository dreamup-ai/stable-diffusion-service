from flask import Flask, request, make_response, jsonify, send_file
from PIL import Image
from io import BytesIO
import os
import logging
import time
from waitress import serve

from __version__ import VERSION

log = logging.getLogger()
log.setLevel(logging.INFO)

log.info("Version: ", VERSION)

# Load config from the environment
host = os.environ.get("HOST", "localhost")
port = int(os.environ.get("PORT", "1111"))

app = Flask(__name__)

if __name__ == "__main__":
    serve(app, host=host, port=port, ipv6=True)


@app.get("/hc")
def hc():
    log.info({"version": VERSION})
    return make_response(jsonify({"version": VERSION}), 200)
