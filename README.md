# Darkroom

A local image generation studio: a dark Gradio UI on top of a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) backend.
Write a prompt, pick a checkpoint, and watch the image develop step by step.

![Darkroom with a FLUX 3 Klein result](docs/screenshot.png)

## Features

- **txt2img and img2img** with any checkpoint ComfyUI can load (SD 1.5, SDXL, and other single-file checkpoints)
  and with [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), which renders photoreal images and legible
  text in 8 steps, with [FLUX 3 Image Klein](https://bfl.ai/blog/flux-3-image-klein) from Black Forest Labs, and
  with [Qwen-Image 2.1](https://github.com/QwenLM/Qwen-Image), which is strong at typography and dense text
- **FLUX 3 prompt upsampler:** Klein's own Qwen3.5 9B upsampler turns a short prompt into the full scene caption the
  model expects, and Darkroom shows the expanded prompt
- **Model presets:** picking a model sets its recommended resolution, steps, CFG, sampler and scheduler
- **Live development:** a step-by-step progress bar and latent previews while the image renders
- **Aspect ratio sizing:** pick 1:1 to 16:9 and a base resolution (512, 768, 1024); Darkroom picks the
  pixel size, rounded to multiples of 64. Custom width and height are available too.
- **Every sampler and scheduler** your ComfyUI offers, plus steps, CFG, seed (random or fixed) and batches of up to 8
- **Stop** interrupts the running job in ComfyUI
- **Contact sheet:** every image of the session, newest first
- **Any backend:** ComfyUI on this machine, another GPU box, or a [Modal](https://modal.com) cloud GPU behind a token

Images are also saved by ComfyUI in its `output/` folder with the `Darkroom` prefix.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/), plus a running ComfyUI (see below).

```sh
git clone <this repo> darkroom && cd darkroom
uv sync
uv run darkroom          # opens on http://127.0.0.1:7860
```

Darkroom connects to `http://127.0.0.1:8188` on startup. To use another ComfyUI, set `COMFYUI_URL` or change it in the
**Connection** tab and click **Connect / refresh models**.

## Backends

Darkroom only uses ComfyUI's HTTP and websocket API, so it doesn't matter where ComfyUI runs.

### ComfyUI on your own machine

```sh
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128  # NVIDIA GPU
pip install -r requirements.txt
python main.py --preview-method latent2rgb
```

Put `.safetensors` checkpoints in `ComfyUI/models/checkpoints`, then refresh models in Darkroom.

For Z-Image Turbo, download the three files from
[Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files) (about 19 GB) into:

```
ComfyUI/models/diffusion_models/z_image_turbo_bf16.safetensors
ComfyUI/models/text_encoders/qwen_3_4b.safetensors
ComfyUI/models/vae/ae.safetensors
```

Darkroom lists any diffusion model whose name contains `z_image` and expects the text encoder and VAE under these
names. Z-Image needs a GPU with about 16 GB or more of VRAM at bf16.

For FLUX 3 Image Klein, accept the license at
[black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)
(gated, FLUX Kommunity License) and download (about 38 GB):

```
ComfyUI/models/diffusion_models/flux3-image-klein-bf16.safetensors
ComfyUI/models/text_encoders/qwen3vl_4b_bf16.safetensors
ComfyUI/models/text_encoders/qwen3.5_9b_flux3_prompt_upsampler_bf16.safetensors
ComfyUI/models/vae/flux2-vae.safetensors
```

Darkroom lists any diffusion model whose name contains `flux3` and builds the graph from Black Forest Labs' ComfyUI
workflows. It needs ComfyUI 0.41 or later and, at bf16, a GPU with more than 24 GB of VRAM. The base checkpoint
(`flux3-image-klein-base-bf16.safetensors`) also works; picking it sets 50 steps and CFG 3.5.

For Qwen-Image 2.1, download from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)
(about 32 GB):

```
ComfyUI/models/diffusion_models/qwen_image_2.1_bf16.safetensors
ComfyUI/models/text_encoders/qwen3vl_8b_bf16.safetensors
ComfyUI/models/vae/qwen_image_2.1_vae_bf16.safetensors
```

Darkroom lists any diffusion model whose name contains `qwen_image` and builds the graph from ComfyUI's official
Qwen-Image 2.1 template (25 steps, CFG 1). Only the base checkpoint is supported; the Turbo checkpoint needs its own
sampling schedule. Qwen-Image 2.1 is under the Qwen Research License, which allows **non-commercial use only**.
Apple Silicon works through PyTorch's MPS backend with the default torch install.
Without a GPU, install the CPU build of torch and start ComfyUI with `--cpu`. That works but is slow: a 512×512
SD 1.5 image takes several minutes, and the model needs about 6 GB of free RAM.

### ComfyUI on a Modal cloud GPU

[`modal_comfyui.py`](modal_comfyui.py) runs ComfyUI on an NVIDIA L40S (48 GB). ComfyUI has no login of its own, so
the endpoint requires a Modal proxy token. The container shuts down after 5 idle minutes, so you pay only while
generating.

```sh
uv sync --extra modal
uv run modal setup                                           # once: sign in to Modal
uv run modal run modal_comfyui.py::download                  # SD 1.5, SDXL, Z-Image, FLUX 3, Qwen-Image into a Volume
uv run modal deploy modal_comfyui.py                         # prints https://<workspace>--darkroom-comfyui-comfyui.<region>.modal.run
uv run modal workspace proxy-tokens create --name darkroom   # prints a token ID (wk-…) and secret (ws-…), shown only once

export COMFYUI_URL=https://<workspace>--darkroom-comfyui-comfyui.<region>.modal.run
export COMFYUI_TOKEN=wk-….ws-…                               # token ID and secret joined with a dot
uv run darkroom
```

- **More checkpoints:** `uv run modal run modal_comfyui.py::download --extra "owner/repo/file.safetensors,..."`,
  using Hugging Face repo and file names.
- **Another GPU:** `DARKROOM_GPU=A10G uv run modal deploy modal_comfyui.py`.
- **Timing:** the first request after an idle period starts the GPU, which takes about 1 to 2 minutes. After that,
  on the L40S, SD 1.5 at 512×512 with 25 steps takes about 2.5 s, SDXL at 1024×1024 with 30 steps about 15 s,
  Z-Image Turbo at 1024×1024 with 8 steps about 4 s, and FLUX 3 Klein at 16 steps about 5.5 s, or about 14 s with
  the prompt upsampler.
- **FLUX 3 download:** the repo is gated. Accept its license on Hugging Face and store a token as a Modal secret
  named `huggingface` (`uv run modal secret create huggingface HF_TOKEN=hf_...`) before running the download step.

## Configuration

| Variable        | Default                  | Meaning                                                              |
|-----------------|--------------------------|----------------------------------------------------------------------|
| `COMFYUI_URL`   | `http://127.0.0.1:8188`  | ComfyUI server (also editable in the Connection tab)                 |
| `COMFYUI_TOKEN` | unset                    | Bearer token for a ComfyUI behind an auth proxy, such as Modal's     |
| `DARKROOM_HOST` | `127.0.0.1`              | Interface to listen on; use `0.0.0.0` to reach it from other devices |
| `DARKROOM_PORT` | `7860`                   | Port to listen on                                                    |
| `DARKROOM_GPU`  | `L40S`                   | Modal GPU type, read by `modal deploy modal_comfyui.py`              |

## Tips

- Use base resolution 512 for SD 1.5 and 1024 for SDXL and Z-Image; other sizes give duplicated subjects or blurry
  results. Picking a model sets this for you.
- Z-Image Turbo, the distilled FLUX 3 Klein and Qwen-Image 2.1 run at CFG 1, where the negative prompt has no effect. Describe what
  you want in the prompt instead.
- With FLUX 3, keep the upsampler on for short prompts. Turn it off only when you write a full scene caption yourself;
  the **Expanded prompt** box shows what a good one looks like.
- Press Enter in the prompt box to develop.
- For img2img, drop an image in the **Img2img** tab. It is scaled and center-cropped to the chosen size.
  Lower denoise keeps more of the original.
- Turn off **Random seed** to reuse the seed shown in the status line and vary only the prompt.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| 🔴 Cannot reach ComfyUI | ComfyUI isn't running at `COMFYUI_URL`, or the URL is wrong. |
| 0 models | No models in `ComfyUI/models/checkpoints` (or `diffusion_models` for Z-Image). Add one and click **Connect / refresh models**. |
| Qwen-Image: value not in list | The text encoder `qwen3vl_8b_bf16.safetensors` or VAE `qwen_image_2.1_vae_bf16.safetensors` is missing. |
| FLUX 3: value not in list | A FLUX 3 text encoder, upsampler or VAE is missing. See the FLUX 3 files above. |
| Z-Image: value not in list | The text encoder `qwen_3_4b.safetensors` or VAE `ae.safetensors` is missing. See the Z-Image files above. |
| 🔴 Lost connection to ComfyUI | ComfyUI stopped mid-job, usually because it ran out of memory. Check its console; try a smaller size or batch. |
| HTTP 401 from Modal | `COMFYUI_TOKEN` is missing or wrong. It must be `wk-…` and `ws-…` joined with a dot. |
| First Modal image is slow | The GPU container is starting. Later images are fast until it has been idle for 5 minutes. |

## Project layout

```
src/darkroom/app.py      Gradio UI and generation flow
src/darkroom/comfy.py    ComfyUI client (HTTP + websocket) and workflow builder
src/darkroom/flux3_upsampler_prompt.txt  BFL's system prompt for the FLUX 3 upsampler (MIT)
modal_comfyui.py         ComfyUI on a Modal GPU
```

See [CHANGELOG.md](CHANGELOG.md) for release notes.
