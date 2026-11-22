# Darkroom

A local image generation studio: a Gradio UI on top of a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) backend.
Write a prompt, pick a checkpoint, and watch the image develop live. It supports txt2img, img2img, batches,
any sampler and scheduler your ComfyUI has, and a session contact sheet.

Darkroom only talks to ComfyUI's HTTP and websocket API, so ComfyUI can run on the same machine,
on another GPU box, or on a cloud GPU such as [Modal](https://modal.com).

## Run

```sh
uv sync
uv run darkroom                      # http://127.0.0.1:7860
```

| Variable        | Default                  | Meaning                                                  |
|-----------------|--------------------------|----------------------------------------------------------|
| `COMFYUI_URL`   | `http://127.0.0.1:8188`  | ComfyUI server (also editable in the Connection tab)     |
| `COMFYUI_TOKEN` | unset                    | Bearer token for a ComfyUI behind an auth proxy (Modal)  |
| `DARKROOM_HOST` | `127.0.0.1`              | Interface to listen on                                   |
| `DARKROOM_PORT` | `7860`                   | Port to listen on                                        |

## Backend option 1: ComfyUI on your own machine

```sh
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128  # NVIDIA
pip install -r requirements.txt
python main.py --preview-method latent2rgb
```

Put `.safetensors` checkpoints in `ComfyUI/models/checkpoints`, then click **Connect / refresh models** in Darkroom.
Without a GPU, install the CPU build of torch and add `--cpu`; it works, but a 512×512 SD 1.5 image takes minutes.

## Backend option 2: ComfyUI on a Modal cloud GPU

[`modal_comfyui.py`](modal_comfyui.py) runs ComfyUI on an L40S (48 GB) behind a Modal proxy token, since ComfyUI
itself has no login. The container shuts down after 5 idle minutes, so you pay only while generating.

```sh
uv sync --extra modal
uv run modal setup                                           # once, signs in to Modal
uv run modal run modal_comfyui.py::download                  # SD 1.5 + SDXL base into the "darkroom-models" Volume
uv run modal deploy modal_comfyui.py                         # prints https://<workspace>--darkroom-comfyui-comfyui...modal.run
uv run modal workspace proxy-tokens create --name darkroom   # prints a token ID (wk-…) and secret (ws-…)

export COMFYUI_URL=https://<workspace>--darkroom-comfyui-comfyui.<region>.modal.run
export COMFYUI_TOKEN=wk-….ws-…                               # token ID and secret joined with a dot
uv run darkroom
```

- More checkpoints: `uv run modal run modal_comfyui.py::download --extra "owner/repo/file.safetensors,..."`
- Another GPU: `DARKROOM_GPU=A10G uv run modal deploy modal_comfyui.py`
- The first request after an idle period cold-starts the GPU (about 1 to 2 minutes); later images take seconds.
  On the L40S, SD 1.5 at 512×512 with 25 steps takes about 2.5 s, and SDXL at 1024×1024 with 30 steps about 15 s.
