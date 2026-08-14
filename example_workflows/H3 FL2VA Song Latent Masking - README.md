# H3 FL2VA Song-Latent Masking — Reproducible Music-Video Example

This example demonstrates the working music-video setup used to validate **exact master-song audio latent masking** with **reference images** and MiniMax H3 FL2VA.

## What is special about this workflow

- Model checkpoint: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- Two persistent character reference images are supplied through `MiniMaxH3ReferenceToVideo` as `<Picture 1>` and `<Picture 2>`.
- No song audio is connected to any `ref_audio_*` input.
- The original song is loaded once and passed into `H3 Song Audio + Masked Video Context` (`MiniMaxH3SongMaskedAVContext`).
- The node VAE-encodes the exact master-song interval into the H3 audio latent and sets the audio denoise mask to **0 for the full raw clip**.
- Continuation clips also preserve the prior decoded video tail with a video denoise mask of **0 only over the visual context prefix**.
- H3 therefore denoises the new visual region while the actual song already occupies the joint AV latent.
- The final soundtrack remains the untouched original master song.
- The workflow keeps the tested prompts, sampler settings, reference-image wiring, timing math, and KJ linear visual blending unchanged.

## Included assets

Copy these three files from `example_workflows/assets/` into your ComfyUI `input/` directory before loading the workflow:

- `be6f4e89-4c3e-43e0-93f5-cc723ccd9b14.png` — face / identity reference (`Picture 1`)
- `c90ee577-98eb-4f6c-9b0c-562a6b448d69.png` — full-body / wardrobe reference (`Picture 2`)
- `I'll Know You by the Scar.wav` — original master song

`lyrics.txt` is included for reference when editing or extending the prompts.

## Required custom node

This workflow requires the node added by this PR:

**H3 Song Audio + Masked Video Context**  
Class: `MiniMaxH3SongMaskedAVContext`

It also relies on the repository's Update 3 per-token H3 video/audio masking compatibility.

## Important implementation detail

The graph intentionally uses `MiniMaxH3ReferenceToVideo` to retain the two image references, but the loaded H3 checkpoint is the **FL2VA** checkpoint. The song itself is **not** used as Ref2VA audio conditioning: every `ref_audio_*` socket is disconnected. The master song is instead inserted directly into the target H3 audio latent and protected from denoising.

## Master audio timing

At 24 fps with the demonstrated 362-frame raw generation and 39-frame visual context:

- raw clip duration: `362 / 24 = 15.083333 s`
- protected visual context: `39 / 24 = 1.625 s`
- new timeline progression: `(362 - 39) / 24 = 13.458333 s`

The workflow computes later clip starts from its existing timing/math nodes; do not manually offset the master audio independently.

## Final output

Use the full loaded master song as the final soundtrack. Per-clip H3 audio decode nodes are debug/inspection only; they are not the authoritative final soundtrack.
