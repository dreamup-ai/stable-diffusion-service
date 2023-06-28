import logging
import random
import torch
import time
from stable_diffusion_models import (
    models,
    safety_checker,
    feature_extractor,
    control_net_models,
)
from PIL import Image
from prompt_tools import get_prompt_embeds


logging.getLogger().setLevel(logging.INFO)


def resize_image(image):
    w, h = image.size
    if w > 1024 and w > h:
        aspect_ratio = w / h
        w = 1024
        h = int(w / aspect_ratio)
    elif h > 1024 and h > w:
        aspect_ratio = h / w
        h = 1024
        w = int(h / aspect_ratio)

    w, h = map(lambda x: x - x % 32, (w, h))
    return image.resize((min(w, 1024), h), resample=Image.LANCZOS).convert("RGB")


def generate_image(job):
    logging.info(f"Generating image for job {job['id']}")
    start = time.perf_counter()
    model_id = job["model"]
    pipeline_id = job["pipeline"]
    params = job["params"]
    if model_id not in models:
        logging.error(f"Model {model_id} not found")
        return None, None, None, None, None
    model = models[model_id]
    if pipeline_id not in model["pipelines"]:
        logging.error(f"Pipeline {pipeline_id} not found for model {model_id}")
        return None, None, None, None, None

    pipe = model["pipelines"][pipeline_id]["pipeline"]
    clean = model["pipelines"][pipeline_id]["sanitize"]
    compel = model["compel"]

    prompt_embeds = get_prompt_embeds(compel, params["prompt"])

    if "negative_prompt" in params:
        negative_prompt_embeds = get_prompt_embeds(compel, params["negative_prompt"])
    else:
        negative_prompt_embeds = get_prompt_embeds(compel, "")

    [
        prompt_embeds,
        negative_prompt_embeds,
    ] = compel.pad_conditioning_tensors_to_same_length(
        [prompt_embeds, negative_prompt_embeds]
    )

    params["prompt_embeds"] = prompt_embeds
    params["negative_prompt_embeds"] = negative_prompt_embeds

    if "image" in params:
        params["image"] = resize_image(params["image"])
        width, height = params["image"].size
        params["width"] = width
        params["height"] = height
        logging.info("Dimensions: %s x %s", width, height)

    if "mask_image" in params:
        params["mask_image"] = resize_image(params["mask_image"])

    if pipeline_id == "controlnet":
        if "control_model" not in params:
            logging.error("Control model not specified")
            return None, None, None, None, None
        control_model = params["control_model"]
        if control_model not in control_net_models:
            logging.error(f"Control model {control_model} not found")
            return None, None, None, None, None
        pipe.controlnet = control_net_models[control_model]

    if "seed" in params:
        seed = params["seed"]
    else:
        seed = random.randint(-9007199254740991, 9007199254740991)
    params["generator"] = torch.Generator("cuda").manual_seed(seed)

    if "scheduler" in params:
        if params["scheduler"] in model["schedulers"]:
            pipe.scheduler = model["schedulers"][params["scheduler"]]
        elif params["scheduler"] not in model["schedulers"]:
            logging.error(
                f"Scheduler {params['scheduler']} not found for model {model_id}"
            )
            return None, None, None, None, None
    else:
        pipe.scheduler = model["schedulers"][model["default_scheduler"]]

    if "safety_checker" in params and params["safety_checker"] is False:
        logging.info(f"Disabling safety checker for job {job['id']}")
        pipe.safety_checker = None, None, None, None, None
        pipe.feature_extractor = None, None, None, None, None
    else:
        pipe.safety_checker = safety_checker
        pipe.feature_extractor = feature_extractor

    scheduler_name = pipe.scheduler.__class__.__name__
    logging.info(f"Using scheduler {scheduler_name} for job {job['id']}")
    try:
        pipe.to("cuda")
        output = pipe(**clean(params))
        img = output.images[0]
        if output.nsfw_content_detected is not None and not isinstance(
            output.nsfw_content_detected, bool
        ):
            nsfw = output.nsfw_content_detected[0]
        elif output.nsfw_content_detected is not None and isinstance(
            output.nsfw_content_detected, bool
        ):
            nsfw = output.nsfw_content_detected
        else:
            nsfw = False
    except TypeError as e:
        logging.error(e)
        nsfw = True
        img = None
        logging.info("Potential NSFW content detected, via TypeError")

    stop = time.perf_counter()
    logging.info(f"Generated image in {stop - start} seconds")
    return img, seed, nsfw, stop - start, scheduler_name
