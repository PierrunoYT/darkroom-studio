"""Run ComfyUI with MiniMax-H3-Image on a Modal GPU outside the territories its license excludes.

    uv run --extra modal modal run modal_h3.py::download   # once: fetch the H3 files into their own Volume
    uv run --extra modal modal deploy modal_h3.py          # prints the ComfyUI URL

The MiniMax-H3 license excludes the EU, the UK, South Korea and the United States, including using or displaying
outputs there. So this app runs its GPU only in Japan or Canada and routes requests through Mumbai, and Darkroom
should run on your own machine in a permitted country, pointed at this app's URL. Modal stores every Volume in the
United States; region pinning only controls where the containers run. Pinned regions cost 1.75x Modal's base price.

The endpoint requires a Modal proxy token, as modal_comfyui.py's does. Containers stop after SCALEDOWN seconds idle.
"""

import os
import subprocess

import modal

# The model (21 GB) and its Qwen3-VL 32B text encoder (27 GB) are large; ComfyUI offloads between them on smaller GPUs.
GPU = os.environ.get("DARKROOM_H3_GPU", "H100")
REGION = ["jp", "ca"]  # not the broad "ap" region, which may place containers in South Korea
ROUTING = "ap-south"  # Mumbai; Modal's other routing regions are in the US or the EU
SCALEDOWN = 300
PORT = 8000

# MiniMax-H3-Image, int8 as in ComfyUI's templates (about 55 GB): (ComfyUI folder, Hugging Face repo, file in the repo)
MODELS = [
    ("diffusion_models", "Comfy-Org/MiniMax-H3-Image", "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors"),
    ("text_encoders", "Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
    ("vae", "Comfy-Org/MiniMax-H3-Image", "vae/minimax_h3_image_vae_fp16.safetensors"),
    ("text_encoders", "Comfy-Org/Qwen3.5", "text_encoders/qwen3.5_9b_w4a8_gptq.safetensors"),  # the prompt enhancer
]
FOLDERS = sorted({folder for folder, *_ in MODELS})
EXTRA_PATHS = r"darkroom:\n  base_path: /models\n" + "".join(rf"  {f}: {f}\n" for f in FOLDERS)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("comfy-cli", "huggingface_hub")
    .run_commands("comfy --skip-prompt install --fast-deps --nvidia")
    .run_commands(f"printf '{EXTRA_PATHS}' > /root/comfy/ComfyUI/extra_model_paths.yaml")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
models = modal.Volume.from_name("darkroom-h3-models", create_if_missing=True, version=2)
app = modal.App("darkroom-h3", image=image)


@app.function(volumes={"/models": models}, timeout=3600, region=REGION, routing_region=ROUTING,
              secrets=[modal.Secret.from_name("huggingface")])
def download():
    from huggingface_hub import hf_hub_download

    for folder, repo, filename in MODELS:
        dest = f"/models/{folder}/{os.path.basename(filename)}"
        if os.path.exists(dest):
            print(f"have {dest}")
            continue
        print(f"downloading {repo}/{filename}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.replace(hf_hub_download(repo, filename, local_dir="/models/.downloads"), dest)
    models.commit()


@app.function(gpu=GPU, volumes={"/models": models}, max_containers=1, region=REGION, routing_region=ROUTING,
              memory=96 * 1024, scaledown_window=SCALEDOWN, timeout=24 * 3600)
@modal.concurrent(max_inputs=100)
@modal.web_server(PORT, startup_timeout=300, requires_proxy_auth=True)
def comfyui():
    subprocess.Popen(
        f"comfy launch -- --listen 0.0.0.0 --port {PORT} --preview-method latent2rgb", shell=True)
