"""Minimal ComfyUI API client and workflow builder."""

from __future__ import annotations

import io
import json
import math
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from importlib.resources import files

import requests
import websocket
from PIL import Image


@dataclass
class Params:
    checkpoint: str
    prompt: str
    negative: str
    width: int
    height: int
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    seed: int
    batch: int
    denoise: float = 1.0
    init_image: str | None = None  # name returned by ComfyClient.upload
    upsample: bool = True  # FLUX 3 and Qwen-Image: expand the prompt with the model's upsampler first
    # Instruction editing (Qwen-Image 2.1, FLUX 3 Klein): uploaded reference images, the one to edit first.
    edit_images: list[str] = field(default_factory=list)


# Z-Image ships as separate diffusion model, text encoder and VAE files (ComfyUI's official template).
ZIMAGE_TEXT_ENCODER = "qwen_3_4b.safetensors"
ZIMAGE_VAE = "ae.safetensors"


def is_zimage(model: str) -> bool:
    return "z_image" in model.lower()


# SD 3.5 Large and Medium ship as single checkpoints with their text encoders; they sample a 16-channel SD3 latent
# (ComfyUI's template). Large Turbo's checkpoint has no text encoders, so it loads them separately (Stability's
# workflow).
SD3_TURBO_TEXT_ENCODERS = ("clip_g.safetensors", "clip_l.safetensors", "t5xxl_fp16.safetensors")


def is_sd3(model: str) -> bool:
    return "sd3" in model.lower()


def is_sd3_turbo(model: str) -> bool:
    return is_sd3(model) and "turbo" in model.lower()


# Ideogram 4, as in ComfyUI's template (Ideogram non-commercial license). Guidance runs a separate unconditional
# model with text-free conditioning; the schedule's mu and std follow the template's Turbo/Default/Quality presets.
IDEOGRAM_TEXT_ENCODER = "qwen3vl_8b_fp8_scaled.safetensors"
# Ideogram 4 is trained on structured JSON captions, and its built-in safety filter blocks most plain prompts. The
# template's caption prompt turns an idea into that JSON; Darkroom runs it on MOSS-Image's Qwen3.5 9B encoder. Darkroom
# adds one rule to it: no transparent background unless the idea asks for one (it sometimes picked one unprompted).
IDEOGRAM_UPSAMPLER_PROMPT = files("darkroom").joinpath("ideogram4_pe_prompt.txt").read_text().strip()


def is_ideogram(model: str) -> bool:
    return "ideogram4" in model.lower()


def _ideogram_schedule(steps: int) -> tuple[float, float]:
    """(mu, std) of the template preset nearest to this step count: Turbo 12, Default 20, Quality 48."""
    if steps <= 12:
        return 0.5, 1.75
    return (0.0, 1.5) if steps >= 48 else (0.0, 1.75)


# FLUX 3 Image Klein, as in Black Forest Labs' ComfyUI workflows (flux-3-image-klein-comfy, MIT).
FLUX3_TEXT_ENCODER = "qwen3vl_4b_bf16.safetensors"
FLUX3_UPSAMPLER = "qwen3.5_9b_flux3_prompt_upsampler_bf16.safetensors"
FLUX3_VAE = "flux2-vae.safetensors"
FLUX3_SHIFT = math.log(16)  # the workflows' timestep shift: mu = ln(alpha), alpha = 16
FLUX3_SYSTEM_PROMPT = files("darkroom").joinpath("flux3_upsampler_prompt.txt").read_text().strip()
FLUX3_EDIT_SYSTEM_PROMPT = files("darkroom").joinpath("flux3_edit_upsampler_prompt.txt").read_text().strip()


def is_flux3(model: str) -> bool:
    return "flux3" in model.lower()


# Qwen-Image 2.1 base, as in ComfyUI's official text-to-image template (Qwen Research License: non-commercial).
QWEN_TEXT_ENCODER = "qwen3vl_8b_bf16.safetensors"
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"
QWEN_ENHANCER = "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors"
QWEN_ENHANCER_PROMPT = files("darkroom").joinpath("qwen_image_pe_prompt.txt").read_text().strip()
# Edits use a separate enhancer that also looks at the reference images (ComfyUI's Qwen-Image 2.1 edit template).
QWEN_EDIT_ENHANCER = "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"
QWEN_EDIT_ENHANCER_PROMPT = files("darkroom").joinpath("qwen_image_edit_pe_prompt.txt").read_text().strip()
# The Turbo checkpoint is distilled for this fixed 8-step schedule (ComfyUI's Turbo template).
QWEN_TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"


def is_qwen_image(model: str) -> bool:
    return "qwen_image" in model.lower()


def is_qwen_turbo(model: str) -> bool:
    return is_qwen_image(model) and "turbo" in model.lower()


# MOSS-Image, as in ComfyUI's official templates (Apache 2.0). Its text encoder also runs the prompt upsampler.
MOSS_TEXT_ENCODER = "qwen3.5_9b_bf16.safetensors"
MOSS_VAE = "flux2-vae_small.safetensors"
MOSS_UPSAMPLER_PROMPT = files("darkroom").joinpath("moss_image_pe_prompt.txt").read_text().strip()


def is_moss(model: str) -> bool:
    return "moss_image" in model.lower()


# Ming Image 0.1 Design, as in ComfyUI's templates (MIT): a 6B model for UI, posters and other text-heavy layouts.
# Text-to-image runs the authors' rewriter first; the template runs it on Qwen3.8 27B, which generates about one token
# a second on an L40S, so Darkroom runs it on MOSS-Image's Qwen3.5 9B. Edits take up to 8 images and sample with LCM.
MING_TEXT_ENCODER = "ming_image_0.1_ling_mini_2.0_int8_convrot.safetensors"
MING_VAE = "ming_image_vae_bf16.safetensors"
MING_ENHANCER_PROMPT = files("darkroom").joinpath("ming_image_pe_prompt.txt").read_text().strip()
MING_MAX_IMAGES = 8


def is_ming(model: str) -> bool:
    return "ming_image" in model.lower()


# Mage-Flow (Microsoft, MIT), as in ComfyUI's templates. Text-to-image and edit are separate checkpoints, each with a
# 4-step Turbo version; one node encodes the prompt, negative and up to 3 reference images and makes the latent.
MAGE_TEXT_ENCODER = "mage_flow_qwen3vl_4b_bf16.safetensors"  # stored under this name; FLUX 3's file of the same name differs
MAGE_VAE = "mage_flow_vae_bf16.safetensors"
MAGE_MAX_IMAGES = 3


def is_mage(model: str) -> bool:
    return "mage_flow" in model.lower()


def is_mage_edit(model: str) -> bool:
    return is_mage(model) and "edit" in model.lower()


# Krea 2 Turbo, as in ComfyUI's template (Krea 2 Community License): 8 steps at CFG 1 against a zeroed negative.
# Its own text encoder refines the prompt first (on by default), with the template's instructions in front of it.
KREA_TEXT_ENCODER = "krea2_qwen3vl_4b_fp8_scaled.safetensors"  # stored under this name, as Mage-Flow's
KREA_REFINER_PROMPT = files("darkroom").joinpath("krea2_pe_prompt.txt").read_text().strip()


def is_krea(model: str) -> bool:
    return "krea2" in model.lower()


# Fibo 2.0 (Bria, CC BY-NC 4.0), as in ComfyUI's edit templates: Dev (50 steps, CFG 5) and its distilled Edit Turbo
# (4 steps, CFG 1). The same models generate from text when no image is given. Fibo reads structured VGL captions, so an
# enhancer rewrites the prompt into one first, on Qwen3-VL 8B (the same model as Ideogram 4's encoder file): for edits
# the template's prompt, which also looks at the images; for text-to-image Bria's VGL "generate" prompt
# (github.com/Bria-AI/vgl, MIT) with the template's additions. Dev's negative is Bria's VGL negative; Turbo has none.
FIBO_TEXT_ENCODER = MAGE_TEXT_ENCODER  # the same file as Mage-Flow's
FIBO_ENHANCER = "qwen3vl_8b_fp8_scaled.safetensors"
FIBO_ENHANCER_PROMPT = files("darkroom").joinpath("fibo2_edit_pe_prompt.txt").read_text().strip()
FIBO_T2I_ENHANCER_PROMPT = files("darkroom").joinpath("fibo2_t2i_pe_prompt.txt").read_text().strip()
FIBO_DEV_NEGATIVE = files("darkroom").joinpath("fibo2_dev_negative.txt").read_text().strip()


def is_fibo(model: str) -> bool:
    return "fibo2" in model.lower()


# JoyAI Image Edit (JD, Apache 2.0), as in ComfyUI's template: edit only, 40 steps at CFG 4 with CFGNorm, on the Wan 2.1
# VAE. Its text encoding node takes up to 6 images.
JOYAI_TEXT_ENCODER = "qwen3vl_8b_joyimage_edit_int8_convrot.safetensors"
JOYAI_VAE = "wan_2.1_vae.safetensors"
JOYAI_MAX_IMAGES = 6


def is_joyai(model: str) -> bool:
    return "joyai" in model.lower()


# Boogu Image 0.1 (Apache 2.0), as in ComfyUI's templates: a Turbo text-to-image checkpoint (4 LCM steps at CFG 1
# against a zeroed negative) and an edit checkpoint (25 dpmpp_2m steps at CFG 3.5; the shift of 3.16 only shapes the
# schedule). Its Qwen3-VL 8B encoder is Ideogram 4's file and its VAE Z-Image's.
BOOGU_TEXT_ENCODER = IDEOGRAM_TEXT_ENCODER


def is_boogu(model: str) -> bool:
    return "boogu" in model.lower()


def is_boogu_edit(model: str) -> bool:
    return is_boogu(model) and "edit" in model.lower()


# Lens (Microsoft, MIT), as in ComfyUI's templates: GPT-OSS 20B as the text encoder, a Flux-style shift and CFGNorm.
# Base: 20 steps at CFG 5; Turbo: 4 steps at CFG 1. The templates' base resolution is 1440.
LENS_TEXT_ENCODER = "gpt_oss_20b_nvfp4.safetensors"
LENS_VAE = "lens_flux2-vae.safetensors"  # stored under this name; it differs from FLUX 3's file


def is_lens(model: str) -> bool:
    return model.lower().startswith("lens_")


# HiDream O1 Image (MIT), as in ComfyUI's templates: one checkpoint with its own text encoder and VAE, for text-to-image
# and edits. Full: 40 dpmpp_2m_sde_gpu steps at CFG 5 with patch seam smoothing; Dev: 28 LCM steps at CFG 1. Sizes are
# multiples of 32. It is trained at about 4 MP (2048 x 2048): edits at 1 MP come out as noise. The optional prompt
# refiner runs Gemma 4 E4B with the template's instructions and chat format. Darkroom runs it for text-to-image only: the
# template's raw chat prompt has no place for the image, so for edits it rewrote the instruction into a new scene.
HIDREAM_REFINER = "gemma4_e4b_it_fp8_scaled.safetensors"
HIDREAM_REFINER_PROMPT = files("darkroom").joinpath("hidream_o1_pe_prompt.txt").read_text()


def is_hidream(model: str) -> bool:
    return "hidream_o1" in model.lower()


def is_edit_only(model: str) -> bool:
    return is_firered(model) or is_mage_edit(model) or is_joyai(model) or is_boogu_edit(model)


# FireRed Image Edit 1.1, as in ComfyUI's template (Apache 2.0): edit only, up to 3 images. Darkroom lists it a
# second time with this suffix to run it with the template's 8-step Lightning LoRA.
FIRERED_TEXT_ENCODER = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
FIRERED_VAE = "qwen_image_vae.safetensors"
FIRERED_LORA = "FireRed-Image-Edit-1.0-Lightning-8steps-v1.0.safetensors"
LIGHTNING = " + Lightning"
FIRERED_MAX_IMAGES = 3


def is_firered(model: str) -> bool:
    return "firered" in model.lower()


def can_edit(model: str) -> bool:
    """Models that take an instruction plus reference images."""
    return (is_flux3(model) or is_qwen_image(model) or is_firered(model) or is_ming(model) or is_mage_edit(model)
            or is_joyai(model) or is_boogu_edit(model) or is_hidream(model)
            or is_fibo(model))


def can_expand(model: str) -> bool:
    """Models with a prompt upsampler or enhancer Darkroom can run."""
    return (is_flux3(model) or is_qwen_image(model) or is_moss(model) or is_ideogram(model) or is_ming(model)
            or is_krea(model) or is_fibo(model) or is_hidream(model))


def _canvas(width: int, height: int) -> str:
    g = math.gcd(width, height)
    return f"{width // g}:{height // g}"


def build_workflow(p: Params) -> dict:
    """Build a ComfyUI API-format graph: txt2img, img2img when init_image is set, or an instruction edit when
    edit_images is set."""
    if is_hidream(p.checkpoint):
        p = replace(p, width=p.width // 32 * 32, height=p.height // 32 * 32)
    if p.edit_images and not can_edit(p.checkpoint):
        raise ValueError(f"{p.checkpoint} cannot edit images; use Qwen-Image 2.1 or FLUX 3 Klein")
    # Ming's edit template has no prompt enhancer.
    expand = (can_expand(p.checkpoint) and p.upsample
              and not (p.edit_images and (is_ming(p.checkpoint) or is_hidream(p.checkpoint))))
    prompt = ["upsample", 0] if expand else p.prompt
    if is_flux3(p.checkpoint):
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.checkpoint, "weight_dtype": "default"}},
            "shift": {"class_type": "ModelSamplingFlux", "inputs": {
                "model": ["unet", 0], "max_shift": FLUX3_SHIFT, "base_shift": FLUX3_SHIFT,
                "width": p.width, "height": p.height}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX3_TEXT_ENCODER, "type": "flux3"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX3_VAE}},
        }
        model, clip, vae, empty_latent = ["shift", 0], ["clip", 0], ["vae", 0], "EmptyFlux2LatentImage"
        if expand:
            # Klein is trained on full scene captions; its upsampler LLM expands a short prompt into one.
            # For edits it also looks at the reference images, as in BFL's image-to-image workflow.
            g["upsampler"] = {"class_type": "CLIPLoader",
                              "inputs": {"clip_name": FLUX3_UPSAMPLER, "type": "flux3"}}
            if p.edit_images:
                system, request = FLUX3_EDIT_SYSTEM_PROMPT, (
                    "Reference images follow in order (the first image, the second image, ...).\n"
                    f"User request: {p.prompt.strip()}\nOutput the caption.")
            else:
                system, request = FLUX3_SYSTEM_PROMPT, f"TARGET CANVAS: {_canvas(p.width, p.height)}\n{p.prompt.strip()}"
            g["upsample"] = {"class_type": "TextGenerate", "inputs": {
                "clip": ["upsampler", 0], "system_prompt": system,
                "prompt": request,
                "max_length": 3584, "thinking": False, "use_default_template": True, "mtp": "off",
                "sampling_mode": "on", "sampling_mode.temperature": 1, "sampling_mode.top_k": 1,
                "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0, "sampling_mode.repetition_penalty": 1,
                "sampling_mode.presence_penalty": 1.5, "sampling_mode.seed": p.seed}}
            if p.edit_images:
                g["upsample"]["inputs"]["sampling_mode.presence_penalty"] = 0  # BFL's i2i setting
                g["ref_batch"] = {"class_type": "BatchImagesNode", "inputs": {
                    f"images.image{i}": [f"ref{i}", 0] for i in range(len(p.edit_images))}}
                g["upsample"]["inputs"]["image"] = ["ref_batch", 0]
            g["show_prompt"] = {"class_type": "PreviewAny", "inputs": {"source": ["upsample", 0]}}
    elif is_moss(p.checkpoint):
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.checkpoint, "weight_dtype": "default"}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": MOSS_TEXT_ENCODER, "type": "moss_image"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": MOSS_VAE}},
        }
        model, clip, vae, empty_latent = ["unet", 0], ["clip", 0], ["vae", 0], "EmptyFlux2LatentImage"
        if "base" in p.checkpoint.lower():
            # The Base template skips layer 22 on the negative branch for the first half; the SFT one does not.
            g["slg"] = {"class_type": "SkipLayerGuidanceDiTSimple", "inputs": {
                "model": ["unet", 0], "double_layers": "", "single_layers": "22",
                "start_percent": 0, "end_percent": 0.5}}
            model = ["slg", 0]
        if expand:
            g["upsample"] = {"class_type": "TextGenerate", "inputs": {
                "clip": ["clip", 0], "system_prompt": MOSS_UPSAMPLER_PROMPT, "prompt": p.prompt.strip(),
                "max_length": 4096, "thinking": False, "use_default_template": True, "mtp": "auto",
                "sampling_mode": "on", "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64,
                "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05, "sampling_mode.repetition_penalty": 1,
                "sampling_mode.presence_penalty": 0, "sampling_mode.seed": p.seed}}
            g["show_prompt"] = {"class_type": "PreviewAny", "inputs": {"source": ["upsample", 0]}}
    elif is_qwen_image(p.checkpoint):
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.checkpoint, "weight_dtype": "default"}},
            # Keeps the text prefix in memory between steps, as the template does.
            "cache": {"class_type": "QwenImage21Cache", "inputs": {"model": ["unet", 0], "device": "auto",
                                                                   "dtype": "default"}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_TEXT_ENCODER, "type": "qwen_image"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
            # Encodes positive and negative prompts in one node.
            "text": {"class_type": "TextEncodeQwenImage21", "inputs": {
                "clip": ["clip", 0], "prompt": prompt, "negative_prompt": p.negative, "resolution": 1024}},
        }
        model, clip, vae, empty_latent = ["cache", 0], None, ["vae", 0], "EmptyLatentImage"
        if p.edit_images:
            # References go into the text encoder; the canvas follows the first one, scaled to about
            # resolution x resolution pixels (ComfyUI's Qwen-Image 2.1 edit template).
            text = g["text"]["inputs"]
            text["vae"] = ["vae", 0]
            text["resolution"] = max(256, round((p.width * p.height) ** 0.5 / 32) * 32)
            for i in range(len(p.edit_images)):
                text[f"images.image_{i + 1}"] = [f"ref{i}", 0]
        if expand and p.edit_images:
            # The edit enhancer sees the reference images and rewrites the instruction; the template leaves
            # thinking off.
            g["upsampler"] = {"class_type": "CLIPLoader",
                              "inputs": {"clip_name": QWEN_EDIT_ENHANCER, "type": "qwen_image"}}
            g["ref_batch"] = {"class_type": "BatchImagesNode", "inputs": {
                f"images.image{i}": [f"ref{i}", 0] for i in range(len(p.edit_images))}}
            g["upsample"] = {"class_type": "TextGenerate", "inputs": {
                "clip": ["upsampler", 0], "image": ["ref_batch", 0], "system_prompt": QWEN_EDIT_ENHANCER_PROMPT,
                "prompt": p.prompt.strip(), "max_length": 4096, "thinking": False, "use_default_template": True,
                "mtp": "auto", "sampling_mode": "on", "sampling_mode.temperature": 1, "sampling_mode.top_k": 20,
                "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0, "sampling_mode.repetition_penalty": 1,
                "sampling_mode.presence_penalty": 0, "sampling_mode.seed": p.seed}}
            g["show_prompt"] = {"class_type": "PreviewAny", "inputs": {"source": ["upsample", 0]}}
        elif expand:
            # Qwen's prompt enhancer rewrites the request; it is trained to reason first, so thinking stays on
            # with the token budget from its model card (a smaller one can end mid-reasoning and return nothing).
            g["upsampler"] = {"class_type": "CLIPLoader",
                              "inputs": {"clip_name": QWEN_ENHANCER, "type": "stable_diffusion"}}
            g["upsample"] = {"class_type": "TextGenerate", "inputs": {
                "clip": ["upsampler", 0], "system_prompt": QWEN_ENHANCER_PROMPT, "prompt": p.prompt.strip(),
                "max_length": 16256, "thinking": True, "use_default_template": True, "mtp": "auto",
                "sampling_mode": "on", "sampling_mode.temperature": 1, "sampling_mode.top_k": 20,
                "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0, "sampling_mode.repetition_penalty": 1,
                "sampling_mode.presence_penalty": 1.5, "sampling_mode.seed": p.seed}}
            g["show_prompt"] = {"class_type": "PreviewAny", "inputs": {"source": ["upsample", 0]}}
    elif is_ideogram(p.checkpoint):
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.checkpoint, "weight_dtype": "default"}},
            "uncond": {"class_type": "UNETLoader", "inputs": {
                "unet_name": p.checkpoint.replace("ideogram4_", "ideogram4_unconditional_", 1),
                "weight_dtype": "default"}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": IDEOGRAM_TEXT_ENCODER, "type": "ideogram4"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX3_VAE}},
            # The template lowers CFG to 3 for the last 30% of sampling.
            "late_cfg": {"class_type": "CFGOverride", "inputs": {
                "model": ["unet", 0], "cfg": 3.0, "start_percent": 0.7, "end_percent": 1.0}},
            "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
            # No negative prompt: the unconditional pass drops the text.
            "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        }
        model, clip, vae, empty_latent = ["late_cfg", 0], None, ["vae", 0], "EmptyFlux2LatentImage"
        if expand:
            g["upsampler"] = {"class_type": "CLIPLoader",
                              "inputs": {"clip_name": MOSS_TEXT_ENCODER, "type": "moss_image"}}
            g["upsample"] = {"class_type": "TextGenerate", "inputs": {
                "clip": ["upsampler", 0], "system_prompt": IDEOGRAM_UPSAMPLER_PROMPT,
                "prompt": f"TARGET IMAGE ASPECT RATIO: {_canvas(p.width, p.height)} (width:height).\n"
                          f"User idea: {p.prompt.strip()}",
                "max_length": 4096, "thinking": False, "use_default_template": True, "mtp": "auto",
                "sampling_mode": "on", "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 20,
                "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0, "sampling_mode.repetition_penalty": 1,
                "sampling_mode.presence_penalty": 0, "sampling_mode.seed": p.seed}}
            g["show_prompt"] = {"class_type": "PreviewAny", "inputs": {"source": ["upsample", 0]}}
    elif is_mage(p.checkpoint):
        if is_mage_edit(p.checkpoint) != bool(p.edit_images):
            raise ValueError("Mage-Flow-Edit only edits, and the text-to-image Mage-Flow checkpoints don't")
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.checkpoint, "weight_dtype": "default"}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": MAGE_TEXT_ENCODER, "type": "mage"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": MAGE_VAE}},
            # Outputs positive, negative and the empty latent; references are resized to the output size.
            "text": {"class_type": "TextEncodeMageFlowEdit", "inputs": {
                "clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "negative_prompt": p.negative,
                "width": p.width, "height": p.height, "batch_size": 1,
                **{f"images.image_{i + 1}": [f"ref{i}", 0] for i in range(len(p.edit_images))}}},
        }
        model, clip, vae, empty_latent = ["unet", 0], None, ["vae", 0], None
    elif is_ming(p.checkpoint):
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.checkpoint, "weight_dtype": "default"}},
            # INT8 attention, as the templates set it (Nvidia and AMD GPUs).
            "attn": {"class_type": "ModelAttentionBackend", "inputs": {
                "model": ["unet", 0], "attention": "comfy kitchen attention"}},
            "shift": {"class_type": "ModelSamplingFlux", "inputs": {
                "model": ["attn", 0], "max_shift": 1.15, "base_shift": 0.5, "width": p.width, "height": p.height}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": MING_TEXT_ENCODER, "type": "qwen_image"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": MING_VAE}},
        }
        if p.edit_images:
            # The sampled latent must match the first image, so it is scaled to the output size first.
            g["resize"] = {"class_type": "ImageScale", "inputs": {
                "image": ["ref0", 0], "upscale_method": "lanczos", "width": p.width, "height": p.height,
                "crop": "center"}}
            g["pos"] = {"class_type": "TextEncodeMingImageEdit", "inputs": {
                "clip": ["clip", 0], "vae": ["vae", 0], "prompt": p.prompt,
                **{f"images.image_{i + 1}": ["resize", 0] if i == 0 else [f"ref{i}", 0]
                   for i in range(len(p.edit_images))}}}
        else:
            g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}}
        if expand:
            g["upsampler"] = {"class_type": "CLIPLoader",
                              "inputs": {"clip_name": MOSS_TEXT_ENCODER, "type": "moss_image"}}
            g["upsample"] = {"class_type": "TextGenerate", "inputs": {
                "clip": ["upsampler", 0], "system_prompt": MING_ENHANCER_PROMPT, "prompt": p.prompt.strip(),
                "max_length": 4096, "thinking": False, "use_default_template": True, "mtp": "auto",
                "sampling_mode": "on", "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64,
                "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05, "sampling_mode.repetition_penalty": 1.05,
                "sampling_mode.presence_penalty": 0, "sampling_mode.seed": p.seed}}
            g["show_prompt"] = {"class_type": "PreviewAny", "inputs": {"source": ["upsample", 0]}}
        model, clip, vae, empty_latent = ["shift", 0], None, ["vae", 0], "EmptyLatentImage"
    elif is_firered(p.checkpoint):
        if not p.edit_images:
            raise ValueError("FireRed Image Edit only edits images")
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {
                "unet_name": p.checkpoint.removesuffix(LIGHTNING), "weight_dtype": "default"}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": FIRERED_TEXT_ENCODER, "type": "qwen_image"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FIRERED_VAE}},
            # The image to edit sets the canvas, scaled to 1 MP; the other references go in as they are.
            "resize": {"class_type": "ResizeImageMaskNode", "inputs": {
                "input": ["ref0", 0], "resize_type": "scale total pixels", "resize_type.megapixels": 1.0,
                "scale_method": "lanczos"}},
        }
        model = ["unet", 0]
        if p.checkpoint.endswith(LIGHTNING):
            g["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                "model": model, "lora_name": FIRERED_LORA, "strength_model": 1.0}}
            model = ["lora", 0]
        g["shift"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": 3.1}}
        g["norm"] = {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}}
        images = {f"image{i + 1}": ["resize", 0] if i == 0 else [f"ref{i}", 0] for i in range(len(p.edit_images))}
        for key, text in (("pos", p.prompt), ("neg", p.negative)):
            g[key] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
                "clip": ["clip", 0], "vae": ["vae", 0], "prompt": text, **images}}
        model, clip, vae, empty_latent = ["norm", 0], None, ["vae", 0], "EmptyLatentImage"
    elif is_fibo(p.checkpoint):
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.checkpoint, "weight_dtype": "default"}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": FIBO_TEXT_ENCODER, "type": "fibo2"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX3_VAE}},
            # Outputs positive, negative and the latent; one image is scaled and cropped to the output size.
            # Without images it generates from text.
            "text": {"class_type": "TextEncodeFibo2", "inputs": {
                "clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt,
                "negative_prompt": "" if "turbo" in p.checkpoint.lower() else FIBO_DEV_NEGATIVE,
                "width": p.width, "height": p.height, "batch_size": 1,
                **{f"images.image_{i + 1}": [f"ref{i}", 0] for i in range(len(p.edit_images))}}},
        }
        if expand:
            g["enhancer"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": FIBO_ENHANCER, "type": "fibo2"}}
            g["upsample"] = {"class_type": "TextGenerateV2", "inputs": {
                "clip": ["enhancer", 0],
                "system_prompt": FIBO_ENHANCER_PROMPT if p.edit_images else FIBO_T2I_ENHANCER_PROMPT,
                # Bria's request wording for text-to-image; edits pass the instruction as the template does.
                "prompt": p.prompt.strip() if p.edit_images else (
                    "Generate a detailed JSON object, adhering to the expected schema, for an imagined scene based on "
                    f"the following request: {p.prompt.strip()}."), "max_length": 3000, "thinking": False,
                "use_default_template": True, "mtp": "auto",
                "sampling_mode": "on", "sampling_mode.temperature": 0.3, "sampling_mode.top_k": 20,
                "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05, "sampling_mode.repetition_penalty": 1.05,
                "sampling_mode.presence_penalty": 0, "sampling_mode.seed": p.seed,
                **{f"images.image{i + 1}": [f"ref{i}", 0] for i in range(len(p.edit_images))}}}
            g["show_prompt"] = {"class_type": "PreviewAny", "inputs": {"source": ["upsample", 0]}}
        model, clip, vae, empty_latent = ["unet", 0], None, ["vae", 0], None
    elif is_hidream(p.checkpoint):
        dev = "dev" in p.checkpoint.lower()
        g = {
            "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": p.checkpoint}},
            "noise_scale": {"class_type": "ModelNoiseScale", "inputs": {
                "model": ["ckpt", 0], "noise_scale": 7.6 if dev else 8.0}},
            "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["ckpt", 1]}},
            "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": p.negative, "clip": ["ckpt", 1]}},
        }
        model = ["noise_scale", 0]
        if not dev:
            g["seams"] = {"class_type": "HiDreamO1PatchSeamSmoothing", "inputs": {
                "model": model, "start_percent": 0.8, "end_percent": 1.0, "pattern": "single_shift",
                "passes": "ramp_2_4", "blend": "median", "strength": 1.0}}
            model = ["seams", 0]
        if p.edit_images:
            g["refs"] = {"class_type": "HiDreamO1ReferenceImages", "inputs": {
                "positive": ["pos", 0], "negative": ["neg", 0],
                **{f"images.image_{i + 1}": [f"ref{i}", 0] for i in range(len(p.edit_images))}}}
        if expand:
            g["refiner"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": HIDREAM_REFINER, "type": "stable_diffusion"}}
            g["refine"] = {"class_type": "TextGenerate", "inputs": {
                "clip": ["refiner", 0],
                "prompt": f"{HIDREAM_REFINER_PROMPT}<|turn>user\n{p.prompt.strip()}<|turn|>\n<|turn>model\n",
                "max_length": 2048, "thinking": False, "use_default_template": False,
                "sampling_mode": "on", "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64,
                "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05, "sampling_mode.repetition_penalty": 1.05,
                "sampling_mode.presence_penalty": 0, "sampling_mode.seed": p.seed}}
            # The refiner answers in JSON, sometimes fenced; the prompt is its "prompt" field.
            g["unfence_start"] = {"class_type": "RegexReplace", "inputs": {
                "string": ["refine", 0], "regex_pattern": "^```(?:json)?\\s*\\n?", "replace": ""}}
            g["unfence_end"] = {"class_type": "RegexReplace", "inputs": {
                "string": ["unfence_start", 0], "regex_pattern": "\\n?```\\s*$", "replace": ""}}
            g["upsample"] = {"class_type": "JsonExtractString", "inputs": {
                "json_string": ["unfence_end", 0], "key": "prompt"}}
            g["show_prompt"] = {"class_type": "PreviewAny", "inputs": {"source": ["upsample", 0]}}
        clip, vae, empty_latent = None, ["ckpt", 2], "EmptyHiDreamO1LatentImage"
    elif is_lens(p.checkpoint):
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.checkpoint, "weight_dtype": "default"}},
            "shift": {"class_type": "ModelSamplingFlux", "inputs": {
                "model": ["unet", 0], "max_shift": 1.15, "base_shift": 0.5, "width": p.width, "height": p.height}},
            "norm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0, "pre_cfg": False}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": LENS_TEXT_ENCODER, "type": "lens"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": LENS_VAE}},
        }
        model, clip, vae, empty_latent = ["norm", 0], ["clip", 0], ["vae", 0], "EmptyLatentImage"
    elif is_boogu(p.checkpoint):
        if is_boogu_edit(p.checkpoint) != bool(p.edit_images):
            raise ValueError("Boogu's edit checkpoint only edits, and its Turbo checkpoint doesn't")
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.checkpoint, "weight_dtype": "default"}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": BOOGU_TEXT_ENCODER, "type": "boogu"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": ZIMAGE_VAE}},
        }
        if p.edit_images:
            g["shift"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.16}}
            g["resize"] = {"class_type": "ImageScale", "inputs": {
                "image": ["ref0", 0], "upscale_method": "lanczos", "width": p.width, "height": p.height,
                "crop": "center"}}
            g["text"] = {"class_type": "TextEncodeBooguEdit", "inputs": {
                "clip": ["clip", 0], "vae": ["vae", 0], "prompt": p.prompt, "negative_prompt": p.negative,
                **{f"images.image_{i + 1}": ["resize", 0] if i == 0 else [f"ref{i}", 0]
                   for i in range(len(p.edit_images))}}}
        else:
            g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"text": p.prompt, "clip": ["clip", 0]}}
            g["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}}
        model, clip, vae, empty_latent = ["unet", 0], None, ["vae", 0], "EmptyLatentImage"
    elif is_joyai(p.checkpoint):
        if not p.edit_images:
            raise ValueError("JoyAI Image Edit only edits images")
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.checkpoint, "weight_dtype": "default"}},
            "norm": {"class_type": "CFGNorm", "inputs": {"model": ["unet", 0], "strength": 1.0, "pre_cfg": True}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": JOYAI_TEXT_ENCODER, "type": "joyimage"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": JOYAI_VAE}},
            # The image to edit is scaled to the output size; the template's latent matches it.
            "resize": {"class_type": "ImageScale", "inputs": {
                "image": ["ref0", 0], "upscale_method": "lanczos", "width": p.width, "height": p.height,
                "crop": "center"}},
        }
        images = {f"images.image{i}": ["resize", 0] if i == 0 else [f"ref{i}", 0] for i in range(len(p.edit_images))}
        for key, text in (("pos", p.prompt), ("neg", p.negative)):
            g[key] = {"class_type": "TextEncodeJoyImageEdit", "inputs": {
                "clip": ["clip", 0], "vae": ["vae", 0], "prompt": text, **images}}
        model, clip, vae, empty_latent = ["norm", 0], None, ["vae", 0], "EmptySD3LatentImage"
    elif is_krea(p.checkpoint):
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.checkpoint, "weight_dtype": "default"}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": KREA_TEXT_ENCODER, "type": "krea2"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FIRERED_VAE}},  # the same Qwen-Image VAE
            "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
            "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        }
        if expand:
            g["upsample"] = {"class_type": "TextGenerate", "inputs": {
                "clip": ["clip", 0], "prompt": f"{KREA_REFINER_PROMPT}\n{p.prompt.strip()}",
                "max_length": 512, "thinking": False, "use_default_template": True,
                "sampling_mode": "on", "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64,
                "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05, "sampling_mode.repetition_penalty": 1.05,
                "sampling_mode.presence_penalty": 0, "sampling_mode.seed": p.seed}}
            g["show_prompt"] = {"class_type": "PreviewAny", "inputs": {"source": ["upsample", 0]}}
        model, clip, vae, empty_latent = ["unet", 0], None, ["vae", 0], "EmptyLatentImage"
    elif is_zimage(p.checkpoint):
        g: dict = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": p.checkpoint, "weight_dtype": "default"}},
            "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": ZIMAGE_TEXT_ENCODER, "type": "lumina2"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": ZIMAGE_VAE}},
        }
        model, clip, vae, empty_latent = ["shift", 0], ["clip", 0], ["vae", 0], "EmptySD3LatentImage"
    else:
        g = {"ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": p.checkpoint}}}
        model, clip, vae = ["ckpt", 0], ["ckpt", 1], ["ckpt", 2]
        empty_latent = "EmptySD3LatentImage" if is_sd3(p.checkpoint) else "EmptyLatentImage"
        if is_sd3_turbo(p.checkpoint):
            g["clip"] = {"class_type": "TripleCLIPLoader", "inputs": {
                f"clip_name{i + 1}": name for i, name in enumerate(SD3_TURBO_TEXT_ENCODERS)}}
            g["shift"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": model, "shift": 3.0}}
            model, clip = ["shift", 0], ["clip", 0]
    # At CFG 1 (Z-Image Turbo's default) ComfyUI skips the negative prompt.
    if is_hidream(p.checkpoint) and p.edit_images:
        positive, negative = ["refs", 0], ["refs", 1]
    elif (is_firered(p.checkpoint) or is_ideogram(p.checkpoint) or is_krea(p.checkpoint) or is_joyai(p.checkpoint)
            or is_hidream(p.checkpoint) or (is_boogu(p.checkpoint) and not p.edit_images)):
        positive, negative = ["pos", 0], ["neg", 0]
    elif is_ming(p.checkpoint):
        positive, negative = ["pos", 0], None  # guided without a negative (BasicGuider)
    elif clip is None:
        positive, negative = ["text", 0], ["text", 1]
    else:
        g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip}}
        g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"text": p.negative, "clip": clip}}
        positive, negative = ["pos", 0], ["neg", 0]
    for i, name in enumerate(p.edit_images):
        g[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
    if p.edit_images:
        if is_flux3(p.checkpoint):
            # Each reference is resized to 1 MP and attached to the conditioning (BFL's workflow).
            refs = {f"images.image{i}": [f"ref{i}", 0] for i in range(len(p.edit_images))}
            g["refs"] = {"class_type": "ReferenceMultiLatent", "inputs": {
                "positive": positive, "negative": negative, "vae": vae,
                "resolution": "resize", "resolution.megapixels": 1.0, **refs}}
            positive, negative = ["refs", 0], ["refs", 1]
            g["latent"] = {"class_type": empty_latent,
                           "inputs": {"width": p.width, "height": p.height, "batch_size": p.batch}}
        elif is_ming(p.checkpoint) or is_joyai(p.checkpoint) or is_boogu(p.checkpoint) or is_hidream(p.checkpoint):
            g["latent"] = {"class_type": empty_latent,
                           "inputs": {"width": p.width, "height": p.height, "batch_size": p.batch}}
        elif is_firered(p.checkpoint):
            # Sampling starts from the encoded image to edit, at full denoise (the template's latent).
            g["encode"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["resize", 0], "vae": vae}}
            g["latent"] = {"class_type": "RepeatLatentBatch", "inputs": {"samples": ["encode", 0], "amount": p.batch}}
        else:
            g["latent"] = {"class_type": "RepeatLatentBatch", "inputs": {"samples": ["text", 2], "amount": p.batch}}
        denoise = 1.0
    elif p.init_image:
        g["load"] = {"class_type": "LoadImage", "inputs": {"image": p.init_image}}
        g["scale"] = {
            "class_type": "ImageScale",
            "inputs": {"image": ["load", 0], "upscale_method": "lanczos",
                       "width": p.width, "height": p.height, "crop": "center"},
        }
        g["encode"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["scale", 0], "vae": vae}}
        g["latent"] = {"class_type": "RepeatLatentBatch", "inputs": {"samples": ["encode", 0], "amount": p.batch}}
        denoise = p.denoise
    elif is_mage(p.checkpoint) or is_fibo(p.checkpoint):
        g["latent"] = {"class_type": "RepeatLatentBatch", "inputs": {"samples": ["text", 2], "amount": p.batch}}
        denoise = 1.0
    else:
        g["latent"] = {
            "class_type": empty_latent,
            "inputs": {"width": p.width, "height": p.height, "batch_size": p.batch},
        }
        denoise = 1.0
    if is_ideogram(p.checkpoint):
        # The template's schedule and guider: the conditional model at CFG 7 (3 late) against the unconditional model.
        mu, std = _ideogram_schedule(p.steps)
        g["sigmas"] = {"class_type": "Ideogram4Scheduler", "inputs": {
            "steps": p.steps, "width": p.width, "height": p.height, "mu": mu, "std": std}}
        sigmas = ["sigmas", 0]
        if denoise < 1:
            g["split"] = {"class_type": "SplitSigmasDenoise", "inputs": {"sigmas": sigmas, "denoise": denoise}}
            sigmas = ["split", 1]
        g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": p.seed}}
        g["guider"] = {"class_type": "DualModelGuider", "inputs": {
            "model": model, "model_negative": ["uncond", 0], "positive": positive, "negative": negative,
            "cfg": p.cfg}}
        g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": p.sampler}}
        g["sample"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
            "sigmas": sigmas, "latent_image": ["latent", 0]}}
    elif is_ming(p.checkpoint):
        # The templates' sampler: no CFG; edits use LCM with slightly raised noise.
        g["sigmas"] = {"class_type": "BasicScheduler", "inputs": {
            "model": model, "scheduler": p.scheduler, "steps": p.steps, "denoise": denoise}}
        g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": p.seed}}
        g["guider"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": positive}}
        g["sampler"] = ({"class_type": "SamplerLCM", "inputs": {"s_noise": 1.02, "s_noise_end": 1.0,
                                                                "noise_clip_std": 0.0}} if p.edit_images
                        else {"class_type": "KSamplerSelect", "inputs": {"sampler_name": p.sampler}})
        g["sample"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
            "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}}
    elif is_hidream(p.checkpoint):
        # The templates' sampler: Dev samples with LCM at raised noise, the full model with the chosen sampler.
        g["sigmas"] = {"class_type": "BasicScheduler", "inputs": {
            "model": ["noise_scale", 0], "scheduler": p.scheduler, "steps": p.steps, "denoise": denoise}}
        g["sampler"] = ({"class_type": "SamplerLCM", "inputs": {"s_noise": 1.0, "s_noise_end": 1.0,
                                                                "noise_clip_std": 2.5}}
                        if "dev" in p.checkpoint.lower()
                        else {"class_type": "KSamplerSelect", "inputs": {"sampler_name": p.sampler}})
        g["sample"] = {"class_type": "SamplerCustom", "inputs": {
            "model": model, "add_noise": True, "noise_seed": p.seed, "cfg": p.cfg, "positive": positive,
            "negative": negative, "sampler": ["sampler", 0], "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}}
    elif is_lens(p.checkpoint):
        # The templates' sampler: the schedule comes from the shifted model, before CFGNorm.
        g["sigmas"] = {"class_type": "BasicScheduler", "inputs": {
            "model": ["shift", 0], "scheduler": p.scheduler, "steps": p.steps, "denoise": denoise}}
        g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": p.sampler}}
        g["sample"] = {"class_type": "SamplerCustom", "inputs": {
            "model": model, "add_noise": True, "noise_seed": p.seed, "cfg": p.cfg, "positive": positive,
            "negative": negative, "sampler": ["sampler", 0], "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}}
    elif is_boogu_edit(p.checkpoint):
        # The template's sampler: the schedule comes from the shifted model, sampling from the unshifted one.
        g["sigmas"] = {"class_type": "BasicScheduler", "inputs": {
            "model": ["shift", 0], "scheduler": p.scheduler, "steps": p.steps, "denoise": denoise}}
        g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": p.sampler}}
        g["sample"] = {"class_type": "SamplerCustom", "inputs": {
            "model": model, "add_noise": True, "noise_seed": p.seed, "cfg": p.cfg, "positive": positive,
            "negative": negative, "sampler": ["sampler", 0], "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}}
    elif is_qwen_turbo(p.checkpoint):
        # Fixed schedule instead of steps/scheduler; img2img keeps the low-noise tail of it.
        g["sigmas"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": QWEN_TURBO_SIGMAS}}
        sigmas = ["sigmas", 0]
        if denoise < 1:
            g["split"] = {"class_type": "SplitSigmasDenoise", "inputs": {"sigmas": sigmas, "denoise": denoise}}
            sigmas = ["split", 1]
        g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": p.seed}}
        g["guider"] = {"class_type": "CFGGuider", "inputs": {
            "model": model, "positive": positive, "negative": negative, "cfg": p.cfg}}
        g["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": p.sampler}}
        g["sample"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
            "sigmas": sigmas, "latent_image": ["latent", 0]}}
    else:
        g["sample"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": model, "positive": positive, "negative": negative,
                "latent_image": ["latent", 0], "seed": p.seed, "steps": p.steps, "cfg": p.cfg,
                "sampler_name": p.sampler, "scheduler": p.scheduler, "denoise": denoise,
            },
        }
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": vae}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "Darkroom"}}
    return g


# A cloud GPU container (e.g. on Modal) can take a few minutes to boot on the first request.
COLD_START = 300


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, url: str, token: str | None = None):
        self.url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        # Bearer token for a ComfyUI behind an auth proxy, such as a Modal proxy token "wk-….ws-…".
        token = token or os.environ.get("COMFYUI_TOKEN")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.http = requests.Session()
        self.http.headers.update(self.headers)

    def _get(self, path: str, **params):
        r = self.http.get(f"{self.url}{path}", params=params, timeout=COLD_START)
        r.raise_for_status()
        return r

    def _choices(self, node: str, field: str) -> list[str]:
        info = self._get(f"/object_info/{node}").json()[node]["input"]["required"][field]
        # Older ComfyUI: [[choices...]]; newer COMBO spec: ["COMBO", {"options": [...]}]
        if isinstance(info[0], list):
            return info[0]
        return info[1].get("options", [])

    def checkpoints(self) -> list[str]:
        """Checkpoints plus the diffusion models Darkroom can build a workflow for (Z-Image, FLUX 3, Qwen-Image,
        MOSS-Image, Ideogram 4, Ming Image, Mage-Flow, Krea 2, Fibo 2.0, JoyAI Image Edit, Boogu, Lens, FireRed Image Edit)."""
        unets = self._choices("UNETLoader", "unet_name")
        lightning = FIRERED_LORA in self._choices("LoraLoaderModelOnly", "lora_name")
        firered = [name for u in unets if is_firered(u) for name in ([u, u + LIGHTNING] if lightning else [u])]
        return self._choices("CheckpointLoaderSimple", "ckpt_name") + [
            u for u in unets if is_zimage(u) or is_flux3(u) or is_qwen_image(u) or is_moss(u) or is_ming(u) or is_mage(u)
            or is_krea(u) or is_fibo(u) or is_joyai(u) or is_boogu(u) or is_lens(u) or (is_ideogram(u) and "unconditional" not in u)] + firered

    def samplers(self) -> list[str]:
        return self._choices("KSampler", "sampler_name")

    def schedulers(self) -> list[str]:
        return self._choices("KSampler", "scheduler")

    def upload(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        files = {"image": (f"darkroom_{uuid.uuid4().hex[:8]}.png", buf.getvalue(), "image/png")}
        r = self.http.post(f"{self.url}/upload/image", files=files, data={"overwrite": "true"},
                           timeout=COLD_START)
        r.raise_for_status()
        d = r.json()
        return f"{d['subfolder']}/{d['name']}" if d.get("subfolder") else d["name"]

    def interrupt(self) -> None:
        self.http.post(f"{self.url}/interrupt", timeout=30)

    def run(self, workflow: dict) -> Iterator[tuple]:
        """Queue a workflow and yield events while it runs.

        Yields ("progress", value, max), ("preview", Image), and finally ("done", [Image, ...], [text, ...]),
        where the texts come from text output nodes such as the FLUX 3 prompt upsampler.
        """
        ws_url = self.url.replace("http", "ws", 1) + f"/ws?clientId={self.client_id}"
        ws = websocket.create_connection(ws_url, timeout=600, header=self.headers)
        try:
            r = self.http.post(f"{self.url}/prompt",
                               json={"prompt": workflow, "client_id": self.client_id}, timeout=COLD_START)
            if r.status_code != 200:
                raise ComfyError(_format_prompt_error(r))
            prompt_id = r.json()["prompt_id"]

            while True:
                msg = ws.recv()
                if isinstance(msg, bytes):
                    # Binary frame: 4-byte event type (1 = preview), 4-byte format, image bytes.
                    if int.from_bytes(msg[:4], "big") == 1:
                        try:
                            yield ("preview", Image.open(io.BytesIO(msg[8:])).convert("RGB"))
                        except Exception:
                            pass
                    continue
                event = json.loads(msg)
                data = event.get("data", {})
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                kind = event.get("type")
                if kind == "progress":
                    yield ("progress", data["value"], data["max"])
                elif kind == "execution_error":
                    raise ComfyError(f"{data.get('node_type')}: {data.get('exception_message', '').strip()}")
                elif kind == "execution_interrupted":
                    raise ComfyError("Interrupted")
                elif kind == "executing" and data.get("node") is None and data.get("prompt_id") == prompt_id:
                    break
        finally:
            ws.close()

        history = self._get(f"/history/{prompt_id}").json()[prompt_id]
        images, texts = [], []
        for output in history["outputs"].values():
            texts += output.get("text", [])
            for im in output.get("images", []):
                raw = self._get("/view", filename=im["filename"], subfolder=im["subfolder"], type=im["type"]).content
                images.append(Image.open(io.BytesIO(raw)))
        yield ("done", images, texts)


def _format_prompt_error(r: requests.Response) -> str:
    try:
        d = r.json()
    except ValueError:
        return f"HTTP {r.status_code}: {r.text[:300]}"
    parts = [d.get("error", {}).get("message", "Invalid prompt")]
    for node in d.get("node_errors", {}).values():
        for e in node.get("errors", []):
            parts.append(f"{node.get('class_type')}: {e.get('message')} {e.get('details', '')}".strip())
    return "\n".join(parts)
