"""Darkroom: a local image generation studio on top of ComfyUI."""

from __future__ import annotations

import os
import random

import gradio as gr
from websocket import WebSocketException

from darkroom.comfy import ComfyClient, ComfyError, Params, build_workflow, is_zimage

DEFAULT_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
MAX_SEED = 2**32 - 1
ASPECTS = {
    "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2),
    "2:3": (2, 3), "16:9": (16, 9), "9:16": (9, 16),
}

CSS = """
.gradio-container { max-width: 1400px !important; }
#title h1 { font-size: 2rem; letter-spacing: 0.35em; text-transform: uppercase; margin: 0; }
#title p { opacity: 0.6; margin: 0.2rem 0 0; }
#title h1 span { color: #e5483b; }
#generate { background: #b3261e; border: none; color: white; }
#generate:hover { background: #e5483b; }
#status { min-height: 1.5rem; font-family: ui-monospace, monospace; font-size: 0.85rem; }
"""

THEME = gr.themes.Base(
    primary_hue="red", neutral_hue="zinc", font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill_dark="#0b0b0c",
    block_background_fill_dark="#141416", block_border_color_dark="#26262a",
    input_background_fill_dark="#1b1b1e",
)
# A darkroom has no lights on: always render Gradio's dark mode.
FORCE_DARK = "() => document.body.classList.add('dark')"


def connect(url: str):
    """Load models, samplers and schedulers from ComfyUI."""
    client = ComfyClient(url)
    try:
        ckpts, samplers, schedulers = client.checkpoints(), client.samplers(), client.schedulers()
    except Exception as e:
        msg = f"🔴 Cannot reach ComfyUI at {url}: {e.__class__.__name__}"
        return gr.update(), gr.update(), gr.update(), msg
    status = f"🟢 Connected to {url} · {len(ckpts)} model(s)"
    if not ckpts:
        status += " · put a model in ComfyUI/models/checkpoints and refresh"
    return (
        gr.update(choices=ckpts, value=ckpts[0] if ckpts else None),
        gr.update(choices=samplers, value="euler" if "euler" in samplers else samplers[0]),
        gr.update(choices=schedulers, value="normal" if "normal" in schedulers else schedulers[0]),
        status,
    )


def size_for(aspect: str, base: int) -> tuple[int, int]:
    """Width and height for an aspect ratio with about base*base pixels, rounded to 64."""
    a, b = ASPECTS[aspect]
    scale = (base * base / (a * b)) ** 0.5
    return max(64, round(a * scale / 64) * 64), max(64, round(b * scale / 64) * 64)


def presets(model: str | None):
    """Recommended base resolution, steps, CFG, sampler and scheduler for a model."""
    if not model:
        return (gr.update(),) * 5
    if is_zimage(model):  # ComfyUI's Z-Image Turbo template
        return 1024, 8, 1, "res_multistep", "simple"
    if "xl" in model.lower():
        return 1024, 25, 7, "euler", "normal"
    return 512, 20, 7, "euler", "normal"


def generate(url, ckpt, prompt, negative, aspect, base, width, height, custom_size,
             steps, cfg, sampler, scheduler, seed, randomize, batch, init_image, denoise, history):
    if not ckpt:
        raise gr.Error("No model selected. Connect to ComfyUI and pick one.")
    if not prompt.strip():
        raise gr.Error("Write a prompt first.")
    if not custom_size:
        width, height = size_for(aspect, int(base))
    seed = random.randint(0, MAX_SEED) if randomize else int(seed)

    client = ComfyClient(url)
    try:
        init_name = client.upload(init_image) if init_image is not None else None
        params = Params(
            checkpoint=ckpt, prompt=prompt, negative=negative, width=int(width), height=int(height),
            steps=int(steps), cfg=float(cfg), sampler=sampler, scheduler=scheduler, seed=seed,
            batch=int(batch), denoise=float(denoise), init_image=init_name,
        )
        mode = "img2img" if init_name else "txt2img"
        yield gr.update(), gr.update(), f"⏳ Queued {mode} · {width}×{height} · seed {seed}", seed, history
        for event in client.run(build_workflow(params)):
            if event[0] == "progress":
                _, value, total = event
                bar = "█" * int(20 * value / total) + "░" * (20 - int(20 * value / total))
                yield gr.update(), gr.update(), f"{bar} step {value}/{total} · seed {seed}", seed, history
            elif event[0] == "preview":
                yield gr.update(value=event[1], visible=True), gr.update(), gr.update(), seed, history
            else:
                images = event[1]
                captions = [f"seed {seed} · {prompt[:80]}" for _ in images]
                history = list(zip(images, captions)) + history
                yield (gr.update(value=None, visible=False), [img for img, _ in history],
                       f"✅ Developed {len(images)} image(s) · {width}×{height} · seed {seed}", seed, history)
    except ComfyError as e:
        msg = "⏹ Stopped" if str(e) == "Interrupted" else f"🔴 {e}"
        yield gr.update(visible=False), gr.update(), msg, seed, history
    except (OSError, WebSocketException) as e:
        msg = f"🔴 Lost connection to ComfyUI ({e}). Check the ComfyUI console; it may have run out of memory."
        yield gr.update(visible=False), gr.update(), msg, seed, history


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Darkroom") as demo:
        history = gr.State([])
        gr.Markdown("# Dark<span>room</span>\nLocal image generation, developed by ComfyUI.", elem_id="title")

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=360):
                prompt = gr.Textbox(label="Prompt", lines=4, autofocus=True,
                                    placeholder="a film photograph of a quiet harbor at dawn, 35mm, grain")
                negative = gr.Textbox(label="Negative prompt", lines=2,
                                      value="blurry, low quality, watermark, text")
                with gr.Row():
                    generate_btn = gr.Button("Develop", variant="primary", elem_id="generate", scale=3)
                    stop_btn = gr.Button("Stop", variant="secondary", scale=1)

                with gr.Tab("Image"):
                    ckpt = gr.Dropdown(label="Model", choices=[], interactive=True,
                                       info="Picking a model applies its recommended settings")
                    aspect = gr.Radio(list(ASPECTS), value="1:1", label="Aspect ratio")
                    base = gr.Radio([512, 768, 1024], value=512, label="Base resolution",
                                    info="512 for SD 1.5, 1024 for SDXL and Z-Image")
                    custom_size = gr.Checkbox(label="Custom size", value=False)
                    with gr.Row(visible=False) as custom_row:
                        width = gr.Slider(64, 2048, 512, step=64, label="Width")
                        height = gr.Slider(64, 2048, 512, step=64, label="Height")
                    batch = gr.Slider(1, 8, 1, step=1, label="Batch size")
                with gr.Tab("Sampler"):
                    with gr.Row():
                        sampler = gr.Dropdown(label="Sampler", choices=[], interactive=True)
                        scheduler = gr.Dropdown(label="Scheduler", choices=[], interactive=True)
                    steps = gr.Slider(1, 100, 20, step=1, label="Steps")
                    cfg = gr.Slider(1, 20, 7, step=0.5, label="CFG scale")
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=0, precision=0, minimum=0, maximum=MAX_SEED)
                        randomize = gr.Checkbox(label="Random seed", value=True)
                with gr.Tab("Img2img"):
                    init_image = gr.Image(label="Init image (optional)", type="pil", height=260)
                    denoise = gr.Slider(0.05, 1.0, 0.6, step=0.05, label="Denoise strength",
                                        info="Lower keeps more of the init image")
                with gr.Tab("Connection"):
                    url = gr.Textbox(label="ComfyUI URL", value=DEFAULT_URL)
                    refresh_btn = gr.Button("Connect / refresh models")

            with gr.Column(scale=6):
                status = gr.Markdown("", elem_id="status")
                preview = gr.Image(label="Developing…", visible=False, interactive=False, height=420)
                gallery = gr.Gallery(label="Contact sheet", columns=3, height=520, object_fit="contain",
                                     show_label=True, preview=False)
                clear_btn = gr.Button("Clear contact sheet", size="sm")

        connect_outputs = [ckpt, sampler, scheduler, status]
        ckpt.change(presets, ckpt, [base, steps, cfg, sampler, scheduler])
        demo.load(connect, url, connect_outputs)
        refresh_btn.click(connect, url, connect_outputs)
        custom_size.change(lambda c: gr.update(visible=c), custom_size, custom_row)

        inputs = [url, ckpt, prompt, negative, aspect, base, width, height, custom_size,
                  steps, cfg, sampler, scheduler, seed, randomize, batch, init_image, denoise, history]
        outputs = [preview, gallery, status, seed, history]
        generate_btn.click(generate, inputs, outputs, concurrency_limit=1)
        prompt.submit(generate, inputs, outputs, concurrency_limit=1)
        # Interrupting ComfyUI ends the running generate() with an "Interrupted" event.
        stop_btn.click(lambda u: ComfyClient(u).interrupt(), url, None)
        clear_btn.click(lambda: ([], []), None, [gallery, history])
    return demo


def main() -> None:
    build_ui().queue().launch(
        theme=THEME, css=CSS, js=FORCE_DARK,
        server_name=os.environ.get("DARKROOM_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("DARKROOM_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
