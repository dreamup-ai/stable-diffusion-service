import logging
import torch
import os
import time
import xformers
import triton
from transformers import CLIPImageProcessor
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile,
    CompilationConfig,
)

import requests


model_dir = os.getenv("MODEL_DIR", "/models")
controlnet_dir = os.path.join(model_dir, "controlnet")
checkpoint_dir = os.path.join(model_dir, "checkpoints")
# Create the directories
os.makedirs(model_dir, exist_ok=True)
os.makedirs(controlnet_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

safety_checker_model = os.getenv(
    "HF_SAFETY_CHECKER", "CompVis/stable-diffusion-safety-checker"
)
feature_extractor_model = os.getenv("HF_CLIP_MODEL", "openai/clip-vit-base-patch32")
civitai_base_url = "https://civitai.com/api/v1/model-versions/"

warmup_inputs = dict(
    prompt="A painting of a cat sitting on a chair",
    height=512,
    width=512,
    num_inference_steps=10,
    num_images_per_prompt=1,
)

compile_config = CompilationConfig.Default()
compile_config.enable_xformers = True
compile_config.enable_triton = True


def load_safety_checker(load_only=False):
    print("Loading safety checker...", flush=True)
    start = time.perf_counter()
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        safety_checker_model, torch_dtype=torch.float16, cache_dir=model_dir
    )
    if not load_only:
        safety_checker.to("cuda")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        feature_extractor_model, torch_dtype=torch.float16, cache_dir=model_dir
    )
    print(
        "Loaded safety checker in %s seconds", time.perf_counter() - start, flush=True
    )
    return safety_checker, feature_extractor


def load_controlnet_model_from_hf(model_name_or_path, load_only=False):
    print(f"Loading controlnet model {model_name_or_path}...")
    start = time.perf_counter()
    model = ControlNetModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        cache_dir=controlnet_dir,
        low_cpu_mem_usage=True,
    )
    if not load_only:
        model.to("cuda", memory_format=torch.channels_last)
    print(
        f"Loaded controlnet model {model_name_or_path} in %s seconds",
        time.perf_counter() - start,
        flush=True,
    )
    return model


def get_model_and_config_from_civitai_payload(payload):
    model_file = None
    config_file = None
    for file in payload["files"]:
        if file["type"] == "Model":
            model_file = file
        elif file["type"] == "Config":
            config_file = file
    return model_file, config_file


def download_file(url, filepath):
    # Use wget to download the file, with --content-disposition to get the filename
    cmd = f'wget -q "{url}" --content-disposition -O {filepath}'
    print(f"Running command {cmd}", flush=True)
    os.system(cmd)


def download_if_not_exists(url, filepath):
    # Check if file already exists and is not empty
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        print(f"File {filepath} already exists, skipping download", flush=True)
    else:
        print(f"Downloading file {filepath}...", flush=True)
        start = time.perf_counter()
        download_file(url, filepath)
        print(
            f"Downloaded file {filepath} in {time.perf_counter() - start}s seconds",
            flush=True,
        )


def load_controlnet_model_from_civitai(model_version_id, load_only=False):
    model_info = requests.get(civitai_base_url + model_version_id).json()
    model_file, config_file = get_model_and_config_from_civitai_payload(model_info)
    model_name = model_info["model"]["name"]
    print(f"Downloading controlnet model {model_name}...", flush=True)
    if model_file:
        model_filename = model_file["name"]
        model_filepath = os.path.join(controlnet_dir, model_filename)
        download_if_not_exists(model_file["downloadUrl"], model_filepath)

    if config_file:
        config_filename = config_file["name"]
        config_filepath = os.path.join(controlnet_dir, config_filename)
        download_if_not_exists(config_file["downloadUrl"], config_filepath)

    print(f"Loading controlnet model {model_name}...", flush=True)
    start = time.perf_counter()
    model = ControlNetModel.from_single_file(model_filepath)
    if not load_only:
        model.to("cuda", memory_format=torch.channels_last)
    print(
        f"Loaded controlnet model {model_name} in %s seconds",
        time.perf_counter() - start,
        flush=True,
    )
    return model


def load_text2img_from_hf(model_id_or_path, load_only=False):
    print(f"Loading checkpoint {model_id_or_path}...", flush=True)
    start = time.perf_counter()
    text2img = StableDiffusionPipeline.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        cache_dir=checkpoint_dir,
        low_cpu_mem_usage=True,
    )
    if load_only:
        return text2img

    text2img.to("cuda", memory_format=torch.channels_last)

    print("Compiling pipeline...", flush=True)
    start = time.perf_counter()
    text2img = compile(text2img, compile_config)

    text2img(**warmup_inputs)
    print("Compiled pipeline in %s seconds", time.perf_counter() - start, flush=True)
    return text2img


def load_text2img_from_civitai(model_path, load_only=False):
    print(f"Loading checkpoint {model_path}...", flush=True)
    start = time.perf_counter()
    text2img = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        low_cpu_mem_usage=True,
    )
    if load_only:
        return text2img

    text2img.to("cuda", memory_format=torch.channels_last)

    print("Compiling pipeline...", flush=True)
    start = time.perf_counter()
    text2img = compile(text2img, compile_config)

    text2img(**warmup_inputs)
    print("Compiled pipeline in %s seconds", time.perf_counter() - start, flush=True)
    return text2img


def load_base_pipelines(text2img, load_only=False):
    img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
    inpaint = StableDiffusionInpaintPipeline(**text2img.components)
    if not load_only:
        img2img.to("cuda")
        inpaint.to("cuda")

    return text2img, img2img, inpaint


def load_checkpoint_from_hf(model_id_or_path, load_only=False):
    print(f"Loading checkpoint {model_id_or_path}...", flush=True)
    start = time.perf_counter()
    text2img = load_text2img_from_hf(model_id_or_path, load_only)
    text2img, img2img, inpaint = load_base_pipelines(text2img, load_only)
    print(
        f"Loaded checkpoint {model_id_or_path} in %s seconds",
        time.perf_counter() - start,
        flush=True,
    )

    return text2img, img2img, inpaint


def load_checkpoint_from_civitai(model_version_id, load_only=False):
    model_info = requests.get(civitai_base_url + model_version_id).json()
    model_file, config_file = get_model_and_config_from_civitai_payload(model_info)
    model_name = model_info["model"]["name"]
    print(f"Loading checkpoint {model_name}...", flush=True)
    if model_file:
        model_filename = model_file["name"]
        model_filepath = os.path.join(checkpoint_dir, model_filename)
        download_if_not_exists(model_file["downloadUrl"], model_filepath)

    if config_file:
        config_filename = config_file["name"]
        config_filepath = os.path.join(checkpoint_dir, config_filename)
        download_if_not_exists(config_file["downloadUrl"], config_filepath)

    text2img = load_text2img_from_civitai(model_filepath, load_only)
    text2img, img2img, inpaint = load_base_pipelines(text2img, load_only)

    return text2img, img2img, inpaint
