# H3 Masked AV Bridge

`H3 Masked AV Bridge` prepares a MiniMax H3 AV latent for true two-ended masked generation.

For a 192-frame target with `preserve_frames = 39`:

- source 1 last 39 frames/audio -> target start, preserved
- target middle 114 frames -> denoised/generated
- source 2 first 39 frames/audio -> target end, preserved

The output is a normal H3 `LATENT` with a nested `noise_mask`:

- `0.0` = preserve the supplied latent
- `1.0` = denoise/generate

The node uses the same H3 AV denoise-mask mechanism as ComfyUI PR #15375. If the running ComfyUI build does not contain native H3 AV mask support, the repo's lazy #15375 compatibility layer is enabled only when this node executes.

## Inputs

- `latent`: target H3 AV latent, e.g. 192 frames
- `vae`: H3 video VAE
- `audio_vae`: H3 audio VAE
- `start_frames`: full first source clip or at least its final preserved window
- `start_audio`: matching first-source audio
- `end_frames`: full second source clip or at least its initial preserved window
- `end_audio`: matching second-source audio
- `start_fps`, `end_fps`: represented CFR frame rates; sources are mapped to H3 24 fps
- `preserve_frames`: exact H3 run length `5, 22, 39, 56, ...`; use 39 for an exact AV boundary
- `crop`: resize behavior for source frames

## Outputs

- `latent`: H3 AV latent with clean preserved ends and nested AV denoise mask
- `middle_frames`: number of frames that will actually be generated
- `preserve_frames`: resolved preserved frame count

## Wiring

Use the output `latent` directly as the sampler's latent input. Do **not** add stock `MiniMaxH3AddGuide` for the same start/end windows; the source AV content is already embedded into the target latent and protected by the denoise mask.

For the intended bridge workflow, use a plain H3 text conditioning node for the transition prompt, then sample the masked latent normally.
