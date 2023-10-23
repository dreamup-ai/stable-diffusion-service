from flask import Flask, request, make_response, jsonify, send_file
import traceback
from PIL import Image
from io import BytesIO
import os
import logging
import time
from waitress import serve
from stable_diffusion_models import models
from stable_diffusion import generate_image
import base64

from __version__ import VERSION

log = logging.getLogger()
log.setLevel(logging.INFO)

log.info("Version: " + VERSION)

# Load config from the environment
host = os.environ.get("HOST", "localhost")
port = int(os.environ.get("PORT", "1111"))


app = Flask(__name__)


@app.get("/hc")
def hc():
    log.info({"version": VERSION})
    return make_response(jsonify({"version": VERSION}), 200)


@app.post("/image")
def image():
    request_start = time.perf_counter()
    try:
        job = request.get_json()
    except Exception as e:
        log.error("Error parsing request body: %s", e)
        return make_response(jsonify({"error": "Error parsing request body"}), 400)

    if "model" not in job:
        log.error("No model specified")
        return make_response(jsonify({"error": "No model specified"}), 400)

    if "pipeline" not in job:
        log.error("No pipeline specified")
        return make_response(jsonify({"error": "No pipeline specified"}), 400)

    if "params" not in job:
        log.error("No params specified")
        return make_response(jsonify({"error": "No params specified"}), 400)

    if job["model"] not in models:
        log.error("Model not found")
        return make_response(jsonify({"error": "Model not found"}), 400)

    if "image" in job["params"]:
        try:
            image = Image.open(BytesIO(base64.b64decode(job["params"]["image"])))
        except Exception as e:
            log.error("Error decoding image: %s", e)
            return make_response(jsonify({"error": "Error decoding image"}), 400)
        job["params"]["image"] = image

    if "mask_image" in job["params"]:
        try:
            mask_image = Image.open(
                BytesIO(base64.b64decode(job["params"]["mask_image"]))
            )
        except Exception as e:
            log.error("Error decoding mask image: %s", e)
            return make_response(jsonify({"error": "Error decoding mask image"}), 400)
        job["params"]["mask_image"] = mask_image

    if "control_image" in job["params"]:
        try:
            control_image = Image.open(
                BytesIO(base64.b64decode(job["params"]["control_image"]))
            )
        except Exception as e:
            log.error("Error decoding control image: %s", e)
            return make_response(
                jsonify({"error": "Error decoding control image"}), 400
            )
        job["params"]["control_image"] = control_image

    if "controlnet_conditioning_scale" in job["params"]:
        job["params"]["controlnet_conditioning_scale"] = float(
            job["params"]["controlnet_conditioning_scale"]
        )
    try:
        img, seed, nsfw, gpu_duration, scheduler_name = generate_image(job)
        if img is None and nsfw is None:
            return make_response(jsonify({"error": "Error generating image"}), 400)
        elif img is None:
            img = Image.new("RGB", (1, 1))

        img_io = BytesIO()
        img.save(img_io, "PNG")
        base64_img = base64.b64encode(img_io.getvalue()).decode("utf-8")
        return make_response(
            jsonify(
                {
                    "image": base64_img,
                    "seed": seed,
                    "nsfw": nsfw,
                    "gpu_duration": gpu_duration,
                    "scheduler_name": scheduler_name,
                    "request_duration": time.perf_counter() - request_start,
                    "image_fmt": "png",
                }
            )
        )

    except Exception as e:
        log.error("Error generating image: %s", e, exc_info=True)
        return make_response(
            jsonify(
                {
                    "error": "Error generating image",
                    "msg": str(e),
                    "traceback": traceback.format_exc(),
                }
            ),
            500,
        )


if __name__ == "__main__":
    serve(app, host=host, port=port, ipv6=True)
