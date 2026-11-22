"""Darkroom: a local image generation studio on top of ComfyUI."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import gradio as gr
from PIL import Image, PngImagePlugin
from websocket import WebSocketException

from darkroom.comfy import ComfyClient, ComfyError, Params, build_workflow, can_edit, can_expand, is_flux3, is_moss, is_qwen_image, is_qwen_turbo, is_zimage

DEFAULT_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
MAX_SEED = 2**32 - 1
# Every developed image is saved here, and the gallery shows this folder.
OUTPUT_DIR = Path(os.environ.get("DARKROOM_OUTPUT_DIR", "~/Pictures/Darkroom")).expanduser().resolve()
GALLERY_LIMIT = 200
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}
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
/* Gradio's grid gives every row the height of its tallest image, which left gaps under wide images. Stack the
   thumbnails in three masonry columns instead, each keeping its whole image. */
#contact .grid-container { display: block; columns: 3; column-gap: var(--spacing-lg); }
#contact .gallery-item { break-inside: avoid; margin-bottom: var(--spacing-lg); }
#contact .thumbnail-item { aspect-ratio: auto; height: auto; width: 100%; }
#contact .thumbnail-item img { height: auto; width: 100%; }
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
    """Recommended base resolution, steps, CFG, sampler, scheduler, and upsampler visibility for a model."""
    if not model:
        return (gr.update(),) * 6
    # On by default for FLUX 3 and MOSS-Image (their workflows), off for Qwen-Image (ComfyUI's template; it adds a
    # slow rewrite).
    name = "FLUX 3 upsampler" if is_flux3(model) else "MOSS-Image upsampler" if is_moss(model) \
        else "Qwen-Image enhancer"
    upsampler = gr.update(visible=can_expand(model), value=is_flux3(model) or is_moss(model),
                          label=f"Expand prompt with the {name}")
    return (*_preset(model), upsampler)


def _preset(model: str):
    if is_flux3(model):  # Black Forest Labs' FLUX 3 Klein workflows
        return (1024, 50, 3.5, "euler", "simple") if "base" in model.lower() else (1024, 16, 1, "euler", "simple")
    if is_moss(model):  # ComfyUI's MOSS-Image templates
        return 1024, 25, 2.5, "er_sde", "simple"
    if is_qwen_turbo(model):  # fixed 8-step schedule; steps and scheduler are ignored
        return 1024, 8, 1, "euler", "simple"
    if is_qwen_image(model):  # ComfyUI's Qwen-Image 2.1 template
        return 1024, 25, 1, "euler", "simple"
    if is_zimage(model):  # ComfyUI's Z-Image Turbo and Z-Image base templates
        return (1024, 8, 1, "res_multistep", "simple") if "turbo" in model.lower() \
            else (1024, 25, 4, "res_multistep", "simple")
    if "xl" in model.lower():
        return 1024, 25, 7, "euler", "normal"
    return 512, 20, 7, "euler", "normal"


def save_images(images, meta: dict) -> None:
    """Write images to OUTPUT_DIR as PNGs with the generation settings in their text chunks."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    info = PngImagePlugin.PngInfo()
    for key, value in meta.items():
        if value not in (None, ""):
            info.add_itxt(key, str(value))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i, image in enumerate(images):
        image.save(OUTPUT_DIR / f"{stamp}-{meta['seed']}-{i}.png", pnginfo=info)


def load_gallery():
    """The newest images in OUTPUT_DIR, with captions from their metadata."""
    if not OUTPUT_DIR.is_dir():
        return []
    files = sorted((f for f in OUTPUT_DIR.iterdir() if f.suffix.lower() in IMAGE_TYPES),
                   key=lambda f: f.stat().st_mtime, reverse=True)[:GALLERY_LIMIT]
    items = []
    for f in files:
        try:
            with Image.open(f) as im:
                meta = im.info
        except OSError:
            continue
        caption = f"seed {meta['seed']} · {meta.get('prompt', '')[:80]}" if "seed" in meta else f.name
        items.append((str(f), caption))
    return items


def describe(evt: gr.SelectData):
    """Settings and location of the image picked in the gallery."""
    # Gradio hands back its cached copy; file names are unique, so point at the original in OUTPUT_DIR.
    path = OUTPUT_DIR / Path(evt.value["image"]["path"]).name
    with Image.open(path) as im:
        meta, size = dict(im.info), im.size
    lines = [f"**{path.name}** · {size[0]}×{size[1]}", f"`{path}`"]
    for key in ("model", "prompt", "expanded_prompt", "negative", "seed", "steps", "cfg", "sampler",
                "scheduler", "denoise", "mode"):
        if key in meta:
            lines.append(f"- **{key.replace('_', ' ')}:** {meta[key]}")
    return gr.update(value="\n".join(lines), visible=True)


def generate(url, ckpt, prompt, negative, aspect, base, width, height, custom_size,
             steps, cfg, sampler, scheduler, seed, randomize, batch, init_image, denoise, upsample,
             edit_image, edit_refs):
    if not ckpt:
        raise gr.Error("No model selected. Connect to ComfyUI and pick one.")
    if not prompt.strip():
        raise gr.Error("Write a prompt first.")
    edit_refs = [ref[0] if isinstance(ref, tuple) else ref for ref in (edit_refs or [])]
    if edit_image is not None:
        if not can_edit(ckpt):
            raise gr.Error("Editing needs Qwen-Image 2.1 or FLUX 3 Klein. Pick one of them, or clear the Edit tab.")
        if init_image is not None:
            raise gr.Error("Use either the Img2img tab or the Edit tab, not both.")
        if 1 + len(edit_refs) > 10:
            raise gr.Error("Editing takes up to 10 images in total.")
        # The result keeps the edited image's shape, at about base x base pixels.
        a, b = edit_image.size
        scale = (int(base) ** 2 / (a * b)) ** 0.5
        width, height = max(256, round(a * scale / 16) * 16), max(256, round(b * scale / 16) * 16)
    elif not custom_size:
        width, height = size_for(aspect, int(base))
    seed = random.randint(0, MAX_SEED) if randomize else int(seed)

    client = ComfyClient(url)
    try:
        init_name = client.upload(init_image) if init_image is not None else None
        edit_names = [client.upload(im) for im in [edit_image, *edit_refs]] if edit_image is not None else []
        params = Params(
            checkpoint=ckpt, prompt=prompt, negative=negative, width=int(width), height=int(height),
            steps=int(steps), cfg=float(cfg), sampler=sampler, scheduler=scheduler, seed=seed,
            batch=int(batch), denoise=float(denoise), init_image=init_name, upsample=bool(upsample),
            edit_images=edit_names,
        )
        expanded = gr.update(visible=False)
        if can_expand(ckpt) and upsample:
            expanded = gr.update(value="Expanding the prompt…", visible=True)
        mode = "edit" if edit_names else "img2img" if init_name else "txt2img"
        yield gr.update(), gr.update(), f"⏳ Queued {mode} · {width}×{height} · seed {seed}", seed, expanded
        for event in client.run(build_workflow(params)):
            if event[0] == "progress":
                _, value, total = event
                bar = "█" * int(20 * value / total) + "░" * (20 - int(20 * value / total))
                yield gr.update(), gr.update(), f"{bar} step {value}/{total} · seed {seed}", seed, gr.update()
            elif event[0] == "preview":
                yield gr.update(value=event[1], visible=True), gr.update(), gr.update(), seed, gr.update()
            else:
                images, texts = event[1], event[2]
                if texts:
                    expanded = gr.update(value=texts[0], visible=True)
                save_images(images, {
                    "prompt": prompt, "expanded_prompt": texts[0] if texts else None, "negative": negative,
                    "model": ckpt, "seed": seed, "steps": steps, "cfg": cfg, "sampler": sampler,
                    "scheduler": scheduler, "mode": mode, "denoise": denoise if init_name else None,
                    "software": "Darkroom"})
                yield (gr.update(value=None, visible=False), load_gallery(),
                       f"✅ Developed {len(images)} image(s) · {width}×{height} · seed {seed} · saved to "
                       f"`{OUTPUT_DIR}`", seed, expanded)
    except ComfyError as e:
        msg = "⏹ Stopped" if str(e) == "Interrupted" else f"🔴 {e}"
        yield gr.update(visible=False), gr.update(), msg, seed, gr.update(visible=False)
    except (OSError, WebSocketException) as e:
        msg = f"🔴 Lost connection to ComfyUI ({e}). Check the ComfyUI console; it may have run out of memory."
        yield gr.update(visible=False), gr.update(), msg, seed, gr.update(visible=False)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Darkroom") as demo:
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
                    upsample = gr.Checkbox(label="Expand prompt with the FLUX 3 upsampler", value=True,
                                           visible=False, info="Rewrites a short prompt into a full scene caption "
                                           "before sampling. Turn off when yours already is one.")
                    aspect = gr.Radio(list(ASPECTS), value="1:1", label="Aspect ratio")
                    base = gr.Radio([512, 768, 1024], value=512, label="Base resolution",
                                    info="512 for SD 1.5, 1024 for SDXL, Z-Image, FLUX 3, Qwen-Image and MOSS")
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
                with gr.Tab("Edit"):
                    gr.Markdown("Qwen-Image 2.1 and FLUX 3 Klein only. Write what to change as the prompt, "
                                "such as *make the boat red*. The result keeps this image's shape.")
                    edit_image = gr.Image(label="Image to edit", type="pil", height=260)
                    edit_refs = gr.Gallery(label="More reference images (optional)", type="pil", interactive=True,
                                           columns=4, height=160, file_types=["image"],
                                           elem_id="edit_refs")
                    gr.Markdown("With Qwen-Image, refer to the images as `<image1>`, `<image2>`, …; with FLUX 3, "
                                "as *the first image*, *the second image*. Up to 10 images in total.",
                                elem_classes="hint")
                with gr.Tab("Connection"):
                    url = gr.Textbox(label="ComfyUI URL", value=DEFAULT_URL)
                    refresh_btn = gr.Button("Connect / refresh models")

            with gr.Column(scale=6):
                status = gr.Markdown("", elem_id="status")
                expanded = gr.Textbox(label="Expanded prompt", lines=3, max_lines=8, interactive=False,
                                      visible=False)
                preview = gr.Image(label="Developing…", visible=False, interactive=False, height=420)
                gallery = gr.Gallery(label=f"Contact sheet · {OUTPUT_DIR}", columns=3, height=520,
                                     object_fit="contain", show_label=True, preview=False, elem_id="contact")
                details = gr.Markdown(visible=False, elem_id="details")
                refresh_gallery_btn = gr.Button("Refresh contact sheet", size="sm")

        connect_outputs = [ckpt, sampler, scheduler, status]
        ckpt.change(presets, ckpt, [base, steps, cfg, sampler, scheduler, upsample])
        demo.load(connect, url, connect_outputs)
        demo.load(load_gallery, None, gallery)
        refresh_gallery_btn.click(load_gallery, None, gallery)
        gallery.select(describe, None, details)
        refresh_btn.click(connect, url, connect_outputs)
        custom_size.change(lambda c: gr.update(visible=c), custom_size, custom_row)

        inputs = [url, ckpt, prompt, negative, aspect, base, width, height, custom_size,
                  steps, cfg, sampler, scheduler, seed, randomize, batch, init_image, denoise, upsample, edit_image, edit_refs]
        outputs = [preview, gallery, status, seed, expanded]
        generate_btn.click(generate, inputs, outputs, concurrency_limit=1)
        prompt.submit(generate, inputs, outputs, concurrency_limit=1)
        # Interrupting ComfyUI ends the running generate() with an "Interrupted" event.
        stop_btn.click(lambda u: ComfyClient(u).interrupt(), url, None)
    return demo


def main() -> None:
    build_ui().queue().launch(
        theme=THEME, css=CSS, js=FORCE_DARK, allowed_paths=[str(OUTPUT_DIR)],
        server_name=os.environ.get("DARKROOM_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("DARKROOM_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
