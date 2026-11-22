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

# (ComfyUI model folder, Hugging Face repo, file in the repo[, local file name when it would clash])
MODELS = [
    ("checkpoints", "Comfy-Org/stable-diffusion-v1-5-archive", "v1-5-pruned-emaonly-fp16.safetensors"),
    ("checkpoints", "stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors"),
    # SD 3.5 Large and Medium with their text encoders, fp8 (27 GB). Stability AI Community License.
    ("checkpoints", "Comfy-Org/stable-diffusion-3.5", "sd3.5_large_fp8_scaled.safetensors"),
    ("checkpoints", "Comfy-Org/stable-diffusion-3.5", "sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors"),
    # SD 3.5 Large Turbo (gated: accept the license on Hugging Face) and the text encoders it loads separately
    # (about 27 GB), as in Stability's workflow.
    ("checkpoints", "stabilityai/stable-diffusion-3.5-large-turbo", "sd3.5_large_turbo.safetensors"),
    ("text_encoders", "Comfy-Org/stable-diffusion-3.5", "text_encoders/clip_g.safetensors"),
    ("text_encoders", "Comfy-Org/stable-diffusion-3.5", "text_encoders/clip_l.safetensors"),
    ("text_encoders", "Comfy-Org/stable-diffusion-3.5", "text_encoders/t5xxl_fp16.safetensors"),
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
    # The Qwen-Image 2.1 edit prompt enhancer (about 9 GB)
    ("text_encoders", "Comfy-Org/Qwen-Image-2.1",
     "text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"),
    # MOSS-Image SFT and Base, bf16, with its text encoder and VAE (about 40 GB). Apache 2.0.
    ("diffusion_models", "Comfy-Org/MOSS-Image", "diffusion_models/moss_image_sft_bf16.safetensors"),
    ("diffusion_models", "Comfy-Org/MOSS-Image", "diffusion_models/moss_image_base_bf16.safetensors"),
    ("text_encoders", "Comfy-Org/MOSS-Image", "text_encoders/qwen3.5_9b_bf16.safetensors"),
    ("vae", "Comfy-Org/MOSS-Image", "vae/flux2-vae_small.safetensors"),
    ("loras", "Comfy-Org/MOSS-Image", "loras/moss_image_edit_lora_rank_64_bf16.safetensors"),  # edit LoRA (about 0.3 GB)
    # Ideogram 4, fp8, as in ComfyUI's template: a conditional and an unconditional model plus a Qwen3-VL 8B text
    # encoder; the FLUX.2 VAE comes with FLUX 3 above (about 27 GB). Ideogram non-commercial license.
    ("diffusion_models", "Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_fp8_scaled.safetensors"),
    ("diffusion_models", "Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors"),
    ("text_encoders", "Comfy-Org/Ideogram-4", "text_encoders/qwen3vl_8b_fp8_scaled.safetensors"),
    # Ming Image 0.1 Design (int8, as in ComfyUI's template) with its Ling text encoder (about 26 GB). MIT. Its prompt
    # rewriter runs on MOSS-Image's Qwen3.5 9B text encoder.
    ("diffusion_models", "Comfy-Org/Ming-Image", "diffusion_models/ming_image_0.1_design_int8_convrot.safetensors"),
    ("text_encoders", "Comfy-Org/Ming-Image", "text_encoders/ming_image_0.1_ling_mini_2.0_int8_convrot.safetensors"),
    ("vae", "Comfy-Org/Ming-Image", "vae/ming_image_vae_bf16.safetensors"),
    # Mage-Flow (Microsoft, MIT), int8 as in ComfyUI's templates: text-to-image and edit, base and Turbo (about 26 GB).
    # Its Qwen3-VL 4B encoder differs from FLUX 3's file of the same name, so it is stored under its own name.
    ("diffusion_models", "Comfy-Org/Mage-Flow", "diffusion_models/mage_flow_int8_convrot.safetensors"),
    ("diffusion_models", "Comfy-Org/Mage-Flow", "diffusion_models/mage_flow_turbo_int8_convrot.safetensors"),
    ("diffusion_models", "Comfy-Org/Mage-Flow", "diffusion_models/mage_flow_edit_int8_convrot.safetensors"),
    ("diffusion_models", "Comfy-Org/Mage-Flow", "diffusion_models/mage_flow_edit_turbo_int8_convrot.safetensors"),
    ("text_encoders", "Comfy-Org/Mage-Flow", "text_encoders/qwen3vl_4b_bf16.safetensors", "mage_flow_qwen3vl_4b_bf16.safetensors"),
    ("vae", "Comfy-Org/Mage-Flow", "vae/mage_flow_vae_bf16.safetensors"),
    # Krea 2 Turbo, fp8 as in ComfyUI's template (about 18 GB), with its Qwen3-VL 4B encoder stored under its own name;
    # the VAE is FireRed's below. Krea 2 Community License: commercial use under $1M annual revenue.
    ("diffusion_models", "Comfy-Org/Krea-2", "diffusion_models/krea2_turbo_fp8_scaled.safetensors"),
    ("diffusion_models", "Comfy-Org/Krea-2", "diffusion_models/krea2_raw_fp8_scaled.safetensors"),  # Raw, undistilled
    ("text_encoders", "Comfy-Org/Krea-2", "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
     "krea2_qwen3vl_4b_fp8_scaled.safetensors"),
    # Fibo 2.0 Dev and Edit Turbo (Bria, CC BY-NC 4.0: non-commercial), int8 as in ComfyUI's edit templates (about 17
    # GB). Its text encoder is Mage-Flow's file, its VAE FLUX 3's, and its enhancer runs on Ideogram 4's Qwen3-VL 8B.
    ("diffusion_models", "Comfy-Org/Fibo-2.0", "diffusion_models/fibo2_dev_int8_convrot.safetensors"),
    ("diffusion_models", "Comfy-Org/Fibo-2.0", "diffusion_models/fibo2_edit_turbo_int8_convrot.safetensors"),
    # JoyAI Image Edit (JD, Apache 2.0), int8 as in ComfyUI's template, with the Wan 2.1 VAE (about 25 GB). Edit only.
    ("diffusion_models", "Comfy-Org/JoyAI-Image-Edit", "diffusion_models/joyai_image_edit_int8_convrot.safetensors"),
    ("text_encoders", "Comfy-Org/JoyAI-Image-Edit", "text_encoders/qwen3vl_8b_joyimage_edit_int8_convrot.safetensors"),
    ("vae", "Comfy-Org/JoyAI-Image-Edit", "vae/wan_2.1_vae.safetensors"),
    # Boogu Image 0.1 (Apache 2.0) as in ComfyUI's templates: Turbo text-to-image (fp8) and edit (int8), about 22 GB. Its
    # text encoder is Ideogram 4's file and its VAE Z-Image's.
    ("diffusion_models", "Comfy-Org/Boogu-Image", "diffusion_models/boogu_image_turbo_fp8_scaled.safetensors"),
    ("diffusion_models", "Comfy-Org/Boogu-Image", "diffusion_models/boogu_image_edit_int8_convrot.safetensors"),
    # Base and the Edit Turbo 1K hotfix Boogu recommends (about 21 GB)
    ("diffusion_models", "Comfy-Org/Boogu-Image", "diffusion_models/boogu_image_base_fp8_scaled.safetensors"),
    ("diffusion_models", "Comfy-Org/Boogu-Image",
     "diffusion_models/boogu_image_edit_turbo_hotfix_1k_20260708_int8_convrot.safetensors"),
    # Lens and Lens Turbo (Microsoft, MIT), bf16 as in ComfyUI's templates, with GPT-OSS 20B (NVFP4) as the text encoder
    # (about 30 GB). Its FLUX.2 VAE differs from FLUX 3's file of the same name, so it is stored under its own name.
    ("diffusion_models", "Comfy-Org/Lens", "diffusion_models/lens_bf16.safetensors"),
    ("diffusion_models", "Comfy-Org/Lens", "diffusion_models/lens_turbo_bf16.safetensors"),
    ("text_encoders", "Comfy-Org/Lens", "text_encoders/gpt_oss_20b_nvfp4.safetensors"),
    ("vae", "Comfy-Org/Lens", "vae/flux2-vae.safetensors", "lens_flux2-vae.safetensors"),
    # HiDream O1 Image (MIT), full bf16 and Dev fp8 as in ComfyUI's templates: single checkpoints with their own text
    # encoder and VAE (about 24 GB), plus Gemma 4 E4B for the templates' prompt refiner (about 9 GB).
    ("checkpoints", "Comfy-Org/HiDream-O1-Image", "checkpoints/hidream_o1_image_bf16.safetensors"),
    ("checkpoints", "Comfy-Org/HiDream-O1-Image", "checkpoints/hidream_o1_image_dev_fp8_scaled.safetensors"),
    ("text_encoders", "Comfy-Org/gemma-4", "text_encoders/gemma4_e4b_it_fp8_scaled.safetensors"),
    # ERNIE-Image and ERNIE-Image Turbo (Baidu, Apache 2.0) as in ComfyUI's templates, with the Ministral 3B text encoder
    # and the prompt enhancer (about 37 GB). Its FLUX.2 VAE is Lens's file.
    ("diffusion_models", "Comfy-Org/ERNIE-Image", "diffusion_models/ernie-image.safetensors"),
    ("diffusion_models", "Comfy-Org/ERNIE-Image", "diffusion_models/ernie-image-turbo.safetensors"),
    ("text_encoders", "Comfy-Org/ERNIE-Image", "text_encoders/ministral-3-3b.safetensors"),
    ("text_encoders", "Comfy-Org/ERNIE-Image", "text_encoders/ernie-image-prompt-enhancer.safetensors"),
    # LongCat-Image, LongCat-Image-Edit and Edit Turbo (Meituan, Apache 2.0), bf16 (about 38 GB). They use
    # FireRed's Qwen2.5-VL 7B encoder and Z-Image's VAE.
    ("diffusion_models", "Comfy-Org/LongCat-Image", "split_files/diffusion_models/longcat_image_bf16.safetensors"),
    ("diffusion_models", "Comfy-Org/LongCat-Image", "split_files/diffusion_models/longcat_image_edit_bf16.safetensors"),
    ("diffusion_models", "Comfy-Org/LongCat-Image",
     "split_files/diffusion_models/longcat_image_edit_turbo_bf16.safetensors"),
    # FireRed Image Edit 1.1 (edit only) with its 8-step Lightning LoRA, as in ComfyUI's template (about 48 GB).
    # Apache 2.0.
    ("diffusion_models", "FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI", "FireRed-Image-Edit-1.1-transformer.safetensors"),
    ("loras", "FireRedTeam/FireRed-Image-Edit-1.0-ComfyUI", "FireRed-Image-Edit-1.0-Lightning-8steps-v1.0.safetensors"),
    ("vae", "FireRedTeam/FireRed-Image-Edit-1.0-ComfyUI", "qwen_image_vae.safetensors"),
    ("text_encoders", "Comfy-Org/HunyuanVideo_1.5_repackaged",
     "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
]
FOLDERS = sorted({folder for folder, *_ in MODELS})

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
    for folder, repo, filename, *name in wanted:
        dest = f"/models/{folder}/{name[0] if name else os.path.basename(filename)}"
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
