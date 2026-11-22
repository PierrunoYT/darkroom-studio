"""Run ComfyUI on a Modal cloud GPU as Darkroom's backend.

    uv run --extra modal modal run modal_comfyui.py::download   # once: fetch models into a Volume
    uv run --extra modal modal deploy modal_comfyui.py          # prints the ComfyUI URL

The endpoint requires a Modal proxy token, because ComfyUI itself has no login.
Containers stop after SCALEDOWN seconds without requests, so an idle GPU costs nothing.
"""

import os
import subprocess

import modal

GPU = os.environ.get("DARKROOM_GPU", "L40S")
SCALEDOWN = 300
PORT = 8000

FLUX3 = "black-forest-labs/flux-3-image-klein-comfy"

# (ComfyUI model folder, Hugging Face repo, file in the repo)
MODELS = [
    ("checkpoints", "Comfy-Org/stable-diffusion-v1-5-archive", "v1-5-pruned-emaonly-fp16.safetensors"),
    ("checkpoints", "stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors"),
    # Z-Image Turbo, as in ComfyUI's official template (about 19 GB)
    ("diffusion_models", "Comfy-Org/z_image_turbo", "split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
    ("text_encoders", "Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors"),
    ("vae", "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),
    # Z-Image base (25 steps, CFG 4); shares the text encoder and VAE above (about 12 GB)
    ("diffusion_models", "Comfy-Org/z_image", "split_files/diffusion_models/z_image_bf16.safetensors"),
    # FLUX 3 Image Klein, distilled, bf16, with its prompt upsampler (about 38 GB). The repo is gated: accept the
    # license on Hugging Face, then store a token as the Modal secret "huggingface" (HF_TOKEN=hf_...).
    ("diffusion_models", FLUX3, "diffusion_models/flux3-image-klein-bf16.safetensors"),
    ("diffusion_models", FLUX3, "diffusion_models/flux3-image-klein-base-bf16.safetensors"),  # base: 50 steps, CFG 3.5
    ("text_encoders", FLUX3, "text_encoders/qwen3vl_4b_bf16.safetensors"),
    ("text_encoders", FLUX3, "text_encoders/qwen3.5_9b_flux3_prompt_upsampler_bf16.safetensors"),
    ("vae", FLUX3, "vae/flux2-vae.safetensors"),
    # Qwen-Image 2.1 base, bf16 (about 32 GB). Qwen Research License: non-commercial use only.
    ("diffusion_models", "Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_bf16.safetensors"),
    ("text_encoders", "Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_bf16.safetensors"),
    ("vae", "Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors"),
    # Qwen-Image 2.1 Turbo (8 steps) and the prompt enhancer (about 24 GB)
    ("diffusion_models", "Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_turbo_bf16.safetensors"),
    ("text_encoders", "Comfy-Org/Qwen-Image-2.1",
     "text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors"),
]
FOLDERS = sorted({folder for folder, _, _ in MODELS})

# Models live in a Volume; ComfyUI finds them through extra_model_paths.yaml (printf expands \n).
EXTRA_PATHS = r"darkroom:\n  base_path: /models\n" + "".join(rf"  {f}: {f}\n" for f in FOLDERS)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("comfy-cli", "huggingface_hub")
    .run_commands("comfy --skip-prompt install --fast-deps --nvidia")
    .run_commands(f"printf '{EXTRA_PATHS}' > /root/comfy/ComfyUI/extra_model_paths.yaml")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
models = modal.Volume.from_name("darkroom-models", create_if_missing=True)
app = modal.App("darkroom-comfyui", image=image)


@app.function(volumes={"/models": models}, timeout=3600, secrets=[modal.Secret.from_name("huggingface")])
def download(extra: str = ""):
    """Download the default models, plus extra checkpoints given as "owner/repo/file.safetensors,..."."""
    from huggingface_hub import hf_hub_download

    wanted = list(MODELS)
    for item in filter(None, extra.split(",")):
        repo, _, filename = item.rpartition("/")
        wanted.append(("checkpoints", repo, filename))
    for folder, repo, filename in wanted:
        dest = f"/models/{folder}/{os.path.basename(filename)}"
        if os.path.exists(dest):
            print(f"have {dest}")
            continue
        print(f"downloading {repo}/{filename}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.replace(hf_hub_download(repo, filename, local_dir="/models/.downloads"), dest)
    models.commit()


@app.function(gpu=GPU, volumes={"/models": models}, max_containers=1,
              scaledown_window=SCALEDOWN, timeout=24 * 3600)
@modal.concurrent(max_inputs=100)
@modal.web_server(PORT, startup_timeout=180, requires_proxy_auth=True)
def comfyui():
    subprocess.Popen(
        f"comfy launch -- --listen 0.0.0.0 --port {PORT} --preview-method latent2rgb", shell=True)
