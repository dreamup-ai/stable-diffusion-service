# stable-diffusion-service
A server that runs stable diffusion tasks

## Currently Supported Pipelines
- [x] text2img
- [x] img2img
- [x] controlnet
- [x] inpainting

## Expected Directory Structure For Models

When you configure models via the `MODELS` and `CONTROLNET_MODELS` environment variables, the server expects to find them in the directory specified by the `MODEL_DIR` environment variable (`/models` by default). Models are organized like `/models/<author>/<name>`, according to their Huggingface repo. For example:

```shell
/models
├── 22h
│   ├── vintedois-diffusion-v0-1
│   └── vintedois-diffusion-v0-2
├── andite
│   └── anything-v4.0
├── CompVis
│   └── stable-diffusion-safety-checker
├── Dunkindont
│   └── Foto-Assisted-Diffusion-FAD_V0
├── gsdf
│   └── Counterfeit-V2.5
├── Intel
│   └── dpt-large
├── Linaqruf
│   └── anything-v3.0
├── lllyasviel
│   ├── Annotators
│   ├── ControlNet
│   ├── control_v11e_sd15_ip2p
│   ├── control_v11e_sd15_shuffle
│   ├── control_v11f1e_sd15_tile
│   ├── control_v11f1p_sd15_depth
│   ├── control_v11p_sd15_canny
│   ├── control_v11p_sd15_inpaint
│   ├── control_v11p_sd15_lineart
│   ├── control_v11p_sd15_mlsd
│   ├── control_v11p_sd15_normalbae
│   ├── control_v11p_sd15_openpose
│   ├── control_v11p_sd15s2_lineart_anime
│   ├── control_v11p_sd15_scribble
│   ├── control_v11p_sd15_seg
│   ├── control_v11p_sd15_softedge
│   ├── sd-controlnet-canny
│   ├── sd-controlnet-depth
│   ├── sd-controlnet-hed
│   ├── sd-controlnet-mlsd
│   ├── sd-controlnet-normal
│   ├── sd-controlnet-openpose
│   ├── sd-controlnet-scribble
│   └── sd-controlnet-seg
├── Lykon
│   ├── DreamShaper
│   └── NeverEnding-Dream
├── nitrosocke
│   ├── elden-ring-diffusion
│   └── Nitro-Diffusion
├── openai
│   └── clip-vit-base-patch32
```

## Available Docker Images

This server is available for free at this image url: 

`public.ecr.aws/i0t3i1w9/stable-diffusion-server`

There are several tags available that offer different levels of baked-in functionality:

### Base Images

- `base`, `0.1.3`: This is the base image. It contains inference code and dependencies, but no models. It is intended to be used as a base for custom images, or in a situation where you want to mount models from a volume instead of baking them into the image. This is the recommended image to use if you are running this server locally, or in a cloud that supports elastic file systems (EFS).
- `safety-checker`, `0.1.3-safety-checker`: This image contains the inference code and the safety checker models ONLY. This is really only intended to be a base image upon which you would load controlnet models and a stable diffusion model.
- `controlnet-slim`, `0.1.3-controlnet-slim`: This image contains the inference code and the controlnet models that are used in Dreamup.ai production. It is intended to be used as a base model. It notable excludes the following controlnet models:
  - segmentation
  - mlsd
  - shuffle
- `controlnet-full`, `0.1.3-controlnet-full`: This image contains the inference code and all of the controlnet models. It is intended to be used as a base model.

### Stable Diffusion Images

- `nitrosocke-nitro-diffusion-controlnet-slim`, `0.1.3-nitrosocke-nitro-diffusion-controlnet-slim`: This image contains the inference code, the nitro-diffusion model, and the controlnet-slim models. It can be run directly for inference.
- `lykon-dreamshaper-controlnet-slim`, `0.1.3-lykon-dreamshaper-controlnet-slim`: This image contains the inference code, the dreamshaper model, and the controlnet-slim models. It can be run directly for inference.

## Build Your Own Stable Diffusion Images

Building your own images is easy. Just use the `build-huggingface-controlnet-image` script.

The syntax is:
```shell
./scripts/build-huggingface-controlnet-image <huggingface repo> <controlnet base-image> <safetensors|bin>
```

 For example:

```shell
./scripts/build-huggingface-controlnet-image nitrosocke/Nitro-Diffusion slim safetensors
```

This will build an image with the nitro-diffusion model and the controlnet-slim models. It will also include the safety checker models. The image will be tagged with the name `nitrosocke-nitro-diffusion-controlnet-slim`, as well as the version of the server that was used to build it (e.g. `0.1.3-nitrosocke-nitro-diffusion-controlnet-slim`).