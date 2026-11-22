# Changelog

All notable changes to Darkroom are listed here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Z-Image Turbo support, following ComfyUI's official workflow: the model list includes `z_image` diffusion
  models, which load with the Qwen 3 4B text encoder and the `ae` VAE, for txt2img and img2img.
- Model presets: picking a model sets its recommended base resolution, steps, CFG, sampler and scheduler
  (Z-Image Turbo: 1024, 8 steps, CFG 1, `res_multistep`, `simple`).
- The Modal download step also fetches Z-Image Turbo into the model Volume.
- FLUX 3 Image Klein support, following Black Forest Labs' ComfyUI workflows: the model list includes `flux3`
  diffusion models, which load with the Qwen3-VL 4B text encoder, the FLUX.2 VAE and a timestep shift of ln 16,
  for txt2img and img2img. Presets: 1024, 16 steps, CFG 1 (distilled) or 50 steps, CFG 3.5 (base).
- FLUX 3 prompt upsampler, on by default: the Qwen3.5 9B upsampler expands the prompt into a full scene caption
  for the canvas shape, and an Expanded prompt box shows the result.
- The Modal download step fetches FLUX 3 Klein from its gated repo using the `huggingface` Modal secret.
- Qwen-Image 2.1 support, following ComfyUI's official template: the model list includes `qwen_image` diffusion
  models, which load with the Qwen3-VL 8B text encoder, the Qwen-Image 2.1 VAE and the Qwen Image 2.1 cache, with
  prompts encoded by the Text Encode Qwen Image 2.1 node. Preset: 1024, 25 steps, CFG 1, `euler`, `simple`.
- The Modal download step also fetches Qwen-Image 2.1 (base, bf16).
- Qwen-Image 2.1 Turbo: the distilled checkpoint samples on its fixed 8-step schedule (ManualSigmas with
  SamplerCustomAdvanced), and img2img runs the low-noise part of that schedule.
- Qwen-Image 2.1 prompt enhancer, off by default as in ComfyUI's template: the Qwen3.5 9B enhancer rewrites the
  prompt with reasoning on, and the Expand prompt toggle now covers both FLUX 3 and Qwen-Image.
- The Modal download step also fetches the Qwen-Image 2.1 Turbo checkpoint and the prompt enhancer.

### Fixed

- The contact sheet showed a gap above wide images and cut them off at the bottom; thumbnails now keep each image's
  aspect ratio.
- The contact sheet now shows each image's seed and prompt as a caption.

### Changed

- The Checkpoint dropdown is now called Model.

## [0.1.0] - 2026-11-22

First release.

### Added

- Gradio studio with a dark theme: prompt and negative prompt, Develop and Stop buttons, and tabs for Image,
  Sampler, Img2img and Connection.
- txt2img and img2img through ComfyUI, with batches of up to 8. img2img scales and center-crops the init image
  to the chosen size.
- Aspect ratio presets (1:1 to 16:9) with base resolutions of 512, 768 and 1024, rounded to multiples of 64,
  plus custom width and height.
- Checkpoint, sampler and scheduler lists loaded from the connected ComfyUI.
- Live progress bar and latent previews over ComfyUI's websocket.
- Stop button that interrupts the running ComfyUI job and reports "Stopped".
- Contact sheet with every image of the session, newest first.
- Clear error messages when ComfyUI is unreachable, rejects a workflow, or drops the connection mid-job.
- `COMFYUI_TOKEN` for ComfyUI servers behind an auth proxy, sent on HTTP requests and the websocket.
- `modal_comfyui.py` to run ComfyUI on a Modal L40S GPU behind a proxy token, with SD 1.5 and SDXL base
  checkpoints stored in a Modal Volume.
