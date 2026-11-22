"""Minimal ComfyUI API client and workflow builder."""

from __future__ import annotations

import io
import json
import math
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
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
    return is_flux3(model) or is_qwen_image(model) or is_firered(model)


def can_expand(model: str) -> bool:
    """Models with a prompt upsampler or enhancer Darkroom can run."""
    return is_flux3(model) or is_qwen_image(model) or is_moss(model)


def _canvas(width: int, height: int) -> str:
    g = math.gcd(width, height)
    return f"{width // g}:{height // g}"


def build_workflow(p: Params) -> dict:
    """Build a ComfyUI API-format graph: txt2img, img2img when init_image is set, or an instruction edit when
    edit_images is set."""
    if p.edit_images and not can_edit(p.checkpoint):
        raise ValueError(f"{p.checkpoint} cannot edit images; use Qwen-Image 2.1 or FLUX 3 Klein")
    expand = can_expand(p.checkpoint) and p.upsample
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
        model, clip, vae, empty_latent = ["ckpt", 0], ["ckpt", 1], ["ckpt", 2], "EmptyLatentImage"
    # At CFG 1 (Z-Image Turbo's default) ComfyUI skips the negative prompt.
    if is_firered(p.checkpoint):
        positive, negative = ["pos", 0], ["neg", 0]
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
    else:
        g["latent"] = {
            "class_type": empty_latent,
            "inputs": {"width": p.width, "height": p.height, "batch_size": p.batch},
        }
        denoise = 1.0
    if is_qwen_turbo(p.checkpoint):
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
        MOSS-Image, FireRed Image Edit)."""
        unets = self._choices("UNETLoader", "unet_name")
        lightning = FIRERED_LORA in self._choices("LoraLoaderModelOnly", "lora_name")
        firered = [name for u in unets if is_firered(u) for name in ([u, u + LIGHTNING] if lightning else [u])]
        return self._choices("CheckpointLoaderSimple", "ckpt_name") + [
            u for u in unets if is_zimage(u) or is_flux3(u) or is_qwen_image(u) or is_moss(u)] + firered

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
