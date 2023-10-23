import logging
import os
import time
from compel import Compel

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
)

from model_loaders import (
    load_safety_checker,
    load_controlnet_model_from_hf,
    load_controlnet_model_from_civitai,
    load_checkpoint_from_civitai,
    load_checkpoint_from_hf,
)

logging.getLogger().setLevel(logging.INFO)

safety_checker, feature_extractor = load_safety_checker()

controlnet_models = {}
# controlnet_model_types = {
#     "canny": "lllyasviel/control_v11p_sd15_canny",
#     "depth": "lllyasviel/control_v11f1p_sd15_depth",
#     "mlsd": "lllyasviel/control_v11p_sd15_mlsd",
#     "normal": "lllyasviel/control_v11p_sd15_normalbae",
#     "openpose": "lllyasviel/control_v11p_sd15_openpose",
#     "scribble": "lllyasviel/control_v11p_sd15_scribble",
#     "softedge": "lllyasviel/control_v11p_sd15_softedge",
#     "shuffle": "lllyasviel/control_v11e_sd15_shuffle",
#     "seg": "lllyasviel/control_v11p_sd15_seg",
#     "lineart": "lllyasviel/control_v11p_sd15_lineart",
#     "lineart_anime": "lllyasviel/control_v11p_sd15s2_lineart_anime",
# }


def process_envvar_array(envvar):
    val = os.getenv(envvar, "")
    if len(val) == 0:
        return []
    return val.split(",")


configured_hf_controlnet_models = process_envvar_array("HF_CONTROLNET_MODELS")
configured_civitaicontrolnet_models = process_envvar_array("CIVITAI_CONTROLNET_MODELS")
if (
    len(configured_hf_controlnet_models) == 0
    and len(configured_civitaicontrolnet_models) == 0
):
    logging.info("No controlnet models configured. Skipping.")

if len(configured_hf_controlnet_models) > 0:
    logging.info(
        f"Configured control net models from Huggingface: {configured_hf_controlnet_models}"
    )

    start = time.perf_counter()
    for model_id in configured_hf_controlnet_models:
        model = load_controlnet_model_from_hf(model_id)
        controlnet_models[model_id] = model
    logging.info(
        f"Loaded {len(configured_hf_controlnet_models)} controlnet models in %s seconds",
        time.perf_counter() - start,
    )

if len(configured_civitaicontrolnet_models) > 0:
    logging.info(
        f"Configured control net models from Civitai: {configured_civitaicontrolnet_models}"
    )

    start = time.perf_counter()
    for model_id in configured_civitaicontrolnet_models:
        model = load_controlnet_model_from_civitai(model_id)
        controlnet_models[model_id] = model
    logging.info(
        f"Loaded {len(configured_civitaicontrolnet_models)} controlnet models in %s seconds",
        time.perf_counter() - start,
    )


# function factory to create sanitizer functions
def make_sanitize_fn(allowed_keys):
    def sanitize_fn(params):
        return {k: v for (k, v) in params.items() if k in allowed_keys}

    return sanitize_fn


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
        "prompt",
        "negative_prompt",
    ]
)

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
        "prompt",
        "negative_prompt",
    ]
)

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
        "prompt",
        "negative_prompt",
    ]
)

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
        "prompt",
        "negative_prompt",
    ]
)

controlnet_img2img_sanitizer = make_sanitize_fn(
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
        "prompt",
        "negative_prompt",
        "control_image",
        "strength",
    ]
)

sanitizers = {
    "text2img": text2imgSanitizer,
    "img2img": img2imgSanitizer,
    "inpaint": inpaintSanitizer,
    "controlnet": controlnet_sanitizer,
    "controlnet_img2img": controlnet_img2img_sanitizer,
}

models = {}

configured_hf_models = process_envvar_array("HF_CHECKPOINTS")

configured_civitai_models = process_envvar_array("CIVITAI_CHECKPOINTS")

if len(configured_hf_models) == 0 and len(configured_civitai_models) == 0:
    logging.error("No checkpoints configured!")
    exit(1)


for model_id in configured_hf_models:
    logging.info(f"Loading pipelines for model {model_id}...")
    start = time.perf_counter()
    text2img, img2img, inpaint = load_checkpoint_from_hf(model_id)
    models[model_id] = {
        "default_scheduler": "DPMSolverMultistepScheduler",
        "default_num_iterations": 25,
    }

    models[model_id]["pipelines"] = {
        "text2img": {
            "pipeline": text2img,
            "sanitize": text2imgSanitizer,
        },
        "img2img": {"pipeline": img2img, "sanitize": img2imgSanitizer},
        "inpaint": {"pipeline": inpaint, "sanitize": inpaintSanitizer},
    }
    models[model_id]["compel"] = Compel(
        tokenizer=text2img.tokenizer,
        text_encoder=text2img.text_encoder,
        truncate_long_prompts=False,
    )
    models[model_id]["schedulers"] = {}
    for Scheduler in text2img.scheduler.compatibles:
        models[model_id]["schedulers"][Scheduler.__name__] = Scheduler.from_config(
            text2img.scheduler.config
        )

    logging.info(
        "Loaded pipelines for model %s in %s seconds",
        model_id,
        time.perf_counter() - start,
    )


for model_id in configured_civitai_models:
    logging.info(f"Loading pipelines for model {model_id}...")
    start = time.perf_counter()
    text2img, img2img, inpaint = load_checkpoint_from_civitai(model_id)
    models[model_id] = {
        "default_scheduler": "DPMSolverMultistepScheduler",
        "default_num_iterations": 25,
    }

    models[model_id]["pipelines"] = {
        "text2img": {
            "pipeline": text2img,
            "sanitize": text2imgSanitizer,
        },
        "img2img": {"pipeline": img2img, "sanitize": img2imgSanitizer},
        "inpaint": {"pipeline": inpaint, "sanitize": inpaintSanitizer},
    }
    models[model_id]["compel"] = Compel(
        tokenizer=text2img.tokenizer,
        text_encoder=text2img.text_encoder,
        truncate_long_prompts=False,
    )
    models[model_id]["schedulers"] = {}
    for Scheduler in text2img.scheduler.compatibles:
        models[model_id]["schedulers"][Scheduler.__name__] = Scheduler.from_config(
            text2img.scheduler.config
        )

    logging.info(
        "Loaded pipelines for model %s in %s seconds",
        model_id,
        time.perf_counter() - start,
    )

allowed_pipelines = {
    "text2img": StableDiffusionPipeline,
    "img2img": StableDiffusionImg2ImgPipeline,
    "inpaint": StableDiffusionInpaintPipeline,
    "controlnet": StableDiffusionControlNetPipeline,
    "controlnet_img2img": StableDiffusionControlNetImg2ImgPipeline,
}


def get_pipeline_and_sanitizer(model_id, pipeline_type, controlnet_type=None):
    if model_id not in models:
        raise ValueError(f"Model {model_id} not found")

    if pipeline_type not in allowed_pipelines:
        raise ValueError(f"Pipeline {pipeline_type} not found")

    if pipeline_type not in models[model_id]["pipelines"]:
        kwargs = {**models[model_id]["pipelines"]["text2img"].pipeline.components}
        if pipeline_type == "controlnet" or pipeline_type == "controlnet_img2img":
            if not controlnet_type:
                raise ValueError(f"Controlnet type not specified")
            if controlnet_type not in controlnet_models:
                raise ValueError(f"Controlnet {controlnet_type} not found")
            kwargs["controlnet"] = controlnet_models[controlnet_type]

        models[model_id]["pipelines"][pipeline_type] = {
            "pipeline": allowed_pipelines[pipeline_type](**kwargs),
            "sanitize": sanitizers[pipeline_type],
        }
