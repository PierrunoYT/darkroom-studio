"""Minimal ComfyUI API client and workflow builder."""

from __future__ import annotations

import io
import json
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

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


# Z-Image ships as separate diffusion model, text encoder and VAE files (ComfyUI's official template).
ZIMAGE_TEXT_ENCODER = "qwen_3_4b.safetensors"
ZIMAGE_VAE = "ae.safetensors"


def is_zimage(model: str) -> bool:
    return "z_image" in model.lower()


def build_workflow(p: Params) -> dict:
    """Build a ComfyUI API-format graph for txt2img, or img2img when init_image is set."""
    if is_zimage(p.checkpoint):
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
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"text": p.prompt, "clip": clip}}
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"text": p.negative, "clip": clip}}
    if p.init_image:
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
    g["sample"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model, "positive": ["pos", 0], "negative": ["neg", 0],
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
        """Checkpoints plus the diffusion models Darkroom can build a workflow for (Z-Image)."""
        unets = self._choices("UNETLoader", "unet_name")
        return self._choices("CheckpointLoaderSimple", "ckpt_name") + [u for u in unets if is_zimage(u)]

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

        Yields ("progress", value, max), ("preview", Image), and finally ("done", [Image, ...]).
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
        images = []
        for output in history["outputs"].values():
            for im in output.get("images", []):
                raw = self._get("/view", filename=im["filename"], subfolder=im["subfolder"], type=im["type"]).content
                images.append(Image.open(io.BytesIO(raw)))
        yield ("done", images)


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
