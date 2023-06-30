import logging
import os
import time
from transformers import CLIPImageProcessor
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import torch

# from dynamodb_json import json_util as json
from compel import Compel

# from dynamo import dynamo
model_dir = os.environ.get("MODEL_DIR", "/models")
safety_checker_model = "CompVis/stable-diffusion-safety-checker"
safety_checker_path = os.path.join(model_dir, safety_checker_model)
feature_extractor_model = "openai/clip-vit-base-patch32"
feature_extractor_path = os.path.join(model_dir, feature_extractor_model)

logging.getLogger().setLevel(logging.INFO)

logging.info("Loading safety checker...")
start = time.perf_counter()
safety_checker = StableDiffusionSafetyChecker.from_pretrained(
    safety_checker_path, torch_dtype=torch.float16
)
safety_checker.to("cuda")
feature_extractor = CLIPImageProcessor.from_pretrained(
    feature_extractor_path, torch_dtype=torch.float16
)
stop = time.perf_counter()
logging.info("Loaded safety checker in %s seconds", stop - start)

control_net_models = {}
control_net_model_types = {
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "mlsd": "lllyasviel/control_v11p_sd15_mlsd",
    "normal": "lllyasviel/control_v11p_sd15_normalbae",
    "openpose": "lllyasviel/control_v11p_sd15_openpose",
    "scribble": "lllyasviel/control_v11p_sd15_scribble",
    "softedge": "lllyasviel/control_v11p_sd15_softedge",
    "shuffle": "lllyasviel/control_v11e_sd15_shuffle",
    "seg": "lllyasviel/control_v11p_sd15_seg",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "lineart_anime": "lllyasviel/control_v11p_sd15s2_lineart_anime",
}

configured_controlnet_models = os.getenv("CONTROLNET_MODELS", "").split(",")
if len(configured_controlnet_models) == 0:
    configured_controlnet_models = list(control_net_model_types.keys())

start = time.perf_counter()
for model_type in configured_controlnet_models:
    logging.info(f"Loading control net model {model_type}...")
    model = ControlNetModel.from_pretrained(
        os.path.join(model_dir, control_net_model_types[model_type]),
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    control_net_models[model_type] = model
    logging.info(f"Loaded control net model {model_type}")
stop = time.perf_counter()
logging.info("Loaded control net models in %s seconds", stop - start)


# function factory to create sanitizer functions
def make_sanitize_fn(allowed_keys):
    def sanitize_fn(params):
        return {k: v for (k, v) in params.items() if k in allowed_keys}

    return sanitize_fn


models = {}


configured_models = os.getenv("MODELS", "").split(",")
if len(configured_models) == 0:
    logging.error("No models configured!")
    exit(1)

for model_name in configured_models:
    logging.info(f"Loading pipelines for model {model_name}...")
    start = time.perf_counter()
    model_path = os.path.join(model_dir, model_name)

    text2img = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
    )
    text2img.to("cuda")
    text2imgSanitizer = make_sanitize_fn(
        [
            "prompt_embeds",
            "width",
            "height",
            "num_inference_steps",
            "guidance_scale",
            "negative_prompt_embeds",
            "eta",
            "generator",
        ]
    )

    img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
    img2img.to("cuda")
    img2imgSanitizer = make_sanitize_fn(
        [
            "prompt_embeds",
            "num_inference_steps",
            "guidance_scale",
            "negative_prompt_embeds",
            "eta",
            "strength",
            "image",
            "generator",
        ]
    )

    inpaint = StableDiffusionInpaintPipeline(**text2img.components)
    inpaint.to("cuda")
    inpaintSanitizer = make_sanitize_fn(
        [
            "prompt_embeds",
            "num_inference_steps",
            "guidance_scale",
            "negative_prompt_embeds",
            "eta",
            "strength",
            "image",
            "mask_image",
            "height",
            "width",
            "generator",
        ]
    )

    controlnet = StableDiffusionControlNetPipeline(
        **text2img.components, controlnet=control_net_models["depth"]
    )
    controlnet.to("cuda")
    controlnet_sanitizer = make_sanitize_fn(
        [
            "prompt_embeds",
            "num_inference_steps",
            "guidance_scale",
            "negative_prompt_embeds",
            "eta",
            "image",
            "generator",
            "controlnet_conditioning_scale",
            "height",
            "width",
        ]
    )

    models[model_name] = {}

    models[model_name]["pipelines"] = {
        "text2img": {
            "pipeline": text2img,
            "sanitize": text2imgSanitizer,
        },
        "img2img": {"pipeline": img2img, "sanitize": img2imgSanitizer},
        "inpaint": {"pipeline": inpaint, "sanitize": inpaintSanitizer},
        "controlnet": {"pipeline": controlnet, "sanitize": controlnet_sanitizer},
    }

    models[model_name]["compel"] = Compel(
        tokenizer=text2img.tokenizer,
        text_encoder=text2img.text_encoder,
        truncate_long_prompts=False,
    )
    models[model_name]["schedulers"] = {}
    for Scheduler in text2img.scheduler.compatibles:
        models[model_name]["schedulers"][Scheduler.__name__] = Scheduler.from_config(
            text2img.scheduler.config
        )

    logging.info(
        "Loaded pipelines for model %s in %s seconds",
        model_name,
        time.perf_counter() - start,
    )
