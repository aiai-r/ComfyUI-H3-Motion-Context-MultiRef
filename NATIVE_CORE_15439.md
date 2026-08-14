# Native ComfyUI H3 guide architecture

This revision targets ComfyUI core with the MiniMax H3 guide architecture from PR #15439.

## Normal continuation path: no core patch

`H3 Motion Context` is now a thin convenience wrapper around the same data model
as stock `Add Guide for MiniMax H3`:

- previous clip tail video -> one native multi-frame `minimax_keyframes` guide at frame 0;
- matching previous audio -> `audio_latent` on that native guide (`cond_audio`);
- persistent identity/voice references -> untouched ordinary `minimax_refs`;
- ComfyUI core owns layout coordinates and payload merging.

The old `patch_layout.py` and keyframe/ref merge patch are intentionally absent.

## Existing-video masked extension

`H3 Existing Video Masked Context` is different: it writes source AV into the
*target latent* and protects the prefix with a denoise mask. That mechanism is
based on PR #15375, not #15439. On a ComfyUI build without native H3 AV-mask
support, the repo installs that compatibility lazily only when this node runs.
Normal Motion Context never activates it.


## Update 3 masked bridge

`H3 Masked AV Bridge` uses the same target-latent masking mechanism as the existing-video masked prefix, but protects both a source-A prefix and a source-B suffix. The endpoint latents are part of the target itself rather than separate AddGuide conditioning.

The compatibility layer is capability-aware: native H3 AV-mask behavior wins when present; otherwise the repo installs its rebased #15375-equivalent behavior lazily for the masked feature only.
