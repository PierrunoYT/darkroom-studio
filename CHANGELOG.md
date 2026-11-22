# Changelog

All notable changes to Darkroom are listed here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Z-Image Turbo support, following ComfyUI's official workflow: the model list includes `z_image` diffusion
  models, which load with the Qwen 3 4B text encoder and the `ae` VAE, for txt2img and img2img.
- Model presets: picking a model sets its recommended base resolution, steps, CFG, sampler and scheduler
  (Z-Image Turbo: 1024, 8 steps, CFG 1, `res_multistep`, `simple`).
- The Modal download step also fetches Z-Image Turbo into the model Volume.
- FLUX 3 Image Klein support, following Black Forest Labs' ComfyUI workflows: the model list includes `flux3`
  diffusion models, which load with the Qwen3-VL 4B text encoder, the FLUX.2 VAE and a timestep shift of ln 16,
  for txt2img and img2img. Presets: 1024, 16 steps, CFG 1 (distilled) or 50 steps, CFG 3.5 (base).
- FLUX 3 prompt upsampler, on by default: the Qwen3.5 9B upsampler expands the prompt into a full scene caption
  for the canvas shape, and an Expanded prompt box shows the result.
- The Modal download step fetches FLUX 3 Klein from its gated repo using the `huggingface` Modal secret.
- Qwen-Image 2.1 support, following ComfyUI's official template: the model list includes `qwen_image` diffusion
  models, which load with the Qwen3-VL 8B text encoder, the Qwen-Image 2.1 VAE and the Qwen Image 2.1 cache, with
  prompts encoded by the Text Encode Qwen Image 2.1 node. Preset: 1024, 25 steps, CFG 1, `euler`, `simple`.
- The Modal download step also fetches Qwen-Image 2.1 (base, bf16).
- Qwen-Image 2.1 Turbo: the distilled checkpoint samples on its fixed 8-step schedule (ManualSigmas with
  SamplerCustomAdvanced), and img2img runs the low-noise part of that schedule.
- Qwen-Image 2.1 prompt enhancer, off by default as in ComfyUI's template: the Qwen3.5 9B enhancer rewrites the
  prompt with reasoning on, and the Expand prompt toggle now covers both FLUX 3 and Qwen-Image.
- The Modal download step also fetches the Qwen-Image 2.1 Turbo checkpoint and the prompt enhancer.
- Z-Image base: picking `z_image_bf16` sets ComfyUI's base template settings (25 steps, CFG 4, `res_multistep`,
  `simple`) instead of the Turbo ones.
- The Modal download step also fetches Z-Image base and the FLUX 3 Klein base checkpoint.
- MOSS-Image support (SFT and Base, Apache 2.0), following ComfyUI's official templates: the Qwen3.5 9B text
  encoder, the small FLUX.2 VAE, 25 steps, CFG 2.5, `er_sde`, and skip-layer guidance for Base. Its text encoder
  runs the prompt upsampler, which is on by default.
- The Modal download step also fetches MOSS-Image SFT and Base.

- Images are saved to `~/Pictures/Darkroom` (or `DARKROOM_OUTPUT_DIR`) as PNGs with the generation settings
  embedded, and the contact sheet shows that folder, newest first, across restarts. Clicking an image shows its
  path and settings; Refresh contact sheet picks up files added by hand.
- Edit mode for Qwen-Image 2.1 (base and Turbo) and FLUX 3 Klein (distilled and base): an Edit tab takes the image
  to edit plus up to 9 more reference images, and the prompt says what to change. The output keeps the edited
  image's aspect ratio at the base resolution. Qwen-Image encodes the images with its text encoder; FLUX 3 uses
  reference latents, and its upsampler uses BFL's edit system prompt and sees the images. Both use model files that
  are already downloaded.
- Qwen-Image 2.1 edit prompt enhancer, following ComfyUI's edit template: in edit mode, Expand prompt runs the
  i2i enhancer, which sees the images and rewrites the instruction. The Modal download step fetches it.
- FireRed Image Edit 1.1 (Apache 2.0), an edit-only model, following ComfyUI's official template: the Qwen2.5-VL 7B
  text encoder, the Qwen-Image VAE, AuraFlow shift 3.1 and CFGNorm, up to 3 images, 40 steps at CFG 4. When its
  Lightning LoRA is installed, the model list also has a `+ Lightning` entry that samples in 8 steps at CFG 1. The
  Modal download step fetches it, and the status line now shows the size of the developed image.
- SD 3.5 Large and Medium (fp8, text encoders included): `sd3` checkpoints sample an SD3 latent. Picking Large sets
  ComfyUI's template settings (1024, 20 steps, CFG 4, `euler`, `sgm_uniform`), and Medium sets the model card's 40
  steps and CFG 4.5. The Modal download step fetches both.
- SD 3.5 Large Turbo, following Stability's ComfyUI workflow: it loads CLIP-G, CLIP-L and T5-XXL separately, uses
  shift 3, and picking it sets 4 steps, CFG 1, `dpmpp_2m`, `sgm_uniform`. The Modal download step fetches it from the
  gated repo with the `huggingface` secret.
- Ideogram 4 (open weights, fp8, non-commercial), following ComfyUI's template: the Qwen3-VL 8B text encoder, the
  FLUX.2 VAE, the Ideogram 4 scheduler and a dual-model guider against the unconditional checkpoint (CFG 7, 3 for the
  last 30%). Picking it sets 20 steps; 12 and 48 steps use the template's Turbo and Quality schedules. A JSON
  captioner, on by default, rewrites the prompt into the structured caption the model expects, using the template's
  caption prompt on the Qwen3.5 9B encoder; without it the built-in safety filter blocks most plain prompts. The Modal
  download step fetches it.
- The Ideogram 4 captioner no longer picks a transparent background on its own: Darkroom adds a rule to the template's
  caption prompt so it uses one only when the prompt asks for it.
- Ming Image 0.1 Design (MIT), following ComfyUI's templates: the Ling text encoder, INT8 attention, Flux-style shift
  and 12 steps without CFG. Text-to-image runs the authors' layout rewriter first (on by default), on the Qwen3.5 9B encoder because the template's Qwen3.8 27B is too slow;
  edits take up to 8 images and sample with LCM. The Modal download step fetches it.
- Mage-Flow (Microsoft, MIT), following ComfyUI's templates: text-to-image and edit checkpoints, each with a 4-step
  Turbo version (30 steps at CFG 5, or 4 at CFG 1). The edit checkpoints take up to 3 images. The Modal download step
  fetches them and can now store a file under a different name when two repos ship different files with the same name.
- Krea 2 Turbo (Krea 2 Community License), following ComfyUI's template: fp8, 8 steps at CFG 1 against a zeroed
  negative, for txt2img and img2img. The template's prompt refiner runs on Krea's own Qwen3-VL 4B encoder and is on by
  default. The Modal download step fetches it.
- Fibo 2.0 Dev and Edit Turbo (Bria, CC BY-NC 4.0) for text-to-image and editing, following ComfyUI's edit templates:
  50 steps at CFG 5 with Bria's structured negative, or 4 steps at CFG 1. A VGL prompt enhancer, on by default, runs on
  Ideogram 4's Qwen3-VL 8B encoder: the template's prompt for edits, which also looks at the images, and Bria's VGL
  `generate` prompt for text-to-image. The Modal download step fetches the two models; the text encoder,
  VAE and enhancer are files Darkroom already has.
- JoyAI Image Edit (JD, Apache 2.0), edit only, following ComfyUI's template: 40 steps at CFG 4 with CFGNorm on the
  Wan 2.1 VAE, with up to 6 images. The Modal download step fetches it.
- Boogu Image 0.1 (Apache 2.0), following ComfyUI's templates: the Turbo checkpoint for text-to-image (4 LCM steps at
  CFG 1) and the edit checkpoint for edits (25 dpmpp_2m steps at CFG 3.5). Its text encoder and VAE are files Darkroom
  already has; the Modal download step fetches the two models.
- Lens and Lens Turbo (Microsoft, MIT), following ComfyUI's templates: GPT-OSS 20B as the text encoder, a Flux-style
  shift and CFGNorm; 20 steps at CFG 5, or 4 at CFG 1. The Modal download step fetches them.
- Send to Edit: after picking an image in the contact sheet, the button loads it as the image to edit and opens the
  Edit tab, so edits can be chained.

### Changed

- The contact sheet is backed by the output folder; the Clear contact sheet button is replaced by Refresh.

### Fixed

- The contact sheet showed a gap above wide images and cut them off at the bottom; thumbnails now keep each image's
  aspect ratio.
- The contact sheet now shows each image's seed and prompt as a caption.
- Mixed wide and tall images left large gaps in the contact sheet; it now stacks thumbnails in three masonry
  columns.

### Changed

- The Checkpoint dropdown is now called Model.

## [0.1.0] - 2026-11-22

First release.

### Added

- Gradio studio with a dark theme: prompt and negative prompt, Develop and Stop buttons, and tabs for Image,
  Sampler, Img2img and Connection.
- txt2img and img2img through ComfyUI, with batches of up to 8. img2img scales and center-crops the init image
  to the chosen size.
- Aspect ratio presets (1:1 to 16:9) with base resolutions of 512, 768 and 1024, rounded to multiples of 64,
  plus custom width and height.
- Checkpoint, sampler and scheduler lists loaded from the connected ComfyUI.
- Live progress bar and latent previews over ComfyUI's websocket.
- Stop button that interrupts the running ComfyUI job and reports "Stopped".
- Contact sheet with every image of the session, newest first.
- Clear error messages when ComfyUI is unreachable, rejects a workflow, or drops the connection mid-job.
- `COMFYUI_TOKEN` for ComfyUI servers behind an auth proxy, sent on HTTP requests and the websocket.
- `modal_comfyui.py` to run ComfyUI on a Modal L40S GPU behind a proxy token, with SD 1.5 and SDXL base
  checkpoints stored in a Modal Volume.
