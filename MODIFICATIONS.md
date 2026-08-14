# Fork history and modifications

This repository is a modified version of **NikoDemon80/ComfyUI-H3-Motion-Context**.

- Upstream: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context
- Original author: NikoDemon80
- H3 AV masking upstream credit: [Barish Ozbay (`drozbay`)](https://github.com/drozbay), author of ComfyUI PR [#15375 — Support per-token video and audio latent noise masks on MiniMax-H3](https://github.com/Comfy-Org/ComfyUI/pull/15375)
- License: GPL-3.0

This file records the main changes made in this fork. Dates refer to the fork/update milestones, not to upstream ComfyUI releases.

## Initial fork — 2026-08-09

The initial fork added **Ref2VA/MultiRef + Motion Context timeline-audio coexistence**.

Main changes:

- Existing `minimax_refs` are preserved instead of being replaced by the Motion Context audio ref.
- The marked Motion Context timeline-audio ref is appended after ordinary Ref2VA refs.
- `patch_layout.py` identifies that marked audio block and moves only its timeline coordinates onto the continuation timeline.
- The Motion Context audio ref is intentionally required to remain the final ref block.
- Runtime compatibility remains local to the custom node; no ComfyUI source files are modified on disk.

The fork also established its own example-workflow family around Simple, Advanced/Ref2VA and Music Video continuation use cases.

## Update 1 — 2026-08-10

Update 1 introduced **H3 Custom Keyframes**.

Main changes:

- Added the `H3 Custom Keyframes` node for placing images as pinned frames at any position in the generated video, as well as a Custom Keyframes Example workflow.
- Added a dynamic keyframe UI with configurable positions and support for multiple anchors.

## Update 2 — 2026-08-11

Update 2's H3 AV masking work builds on **ComfyUI PR [#15375 — Support per-token video and audio latent noise masks on MiniMax-H3](https://github.com/Comfy-Org/ComfyUI/pull/15375)**, authored by **[Barish Ozbay (`drozbay`)](https://github.com/drozbay)**. His PR introduced the H3-aware per-token video/audio denoise-mask design and upstream implementation used as the technical basis for preserving an existing AV prefix while generating the continuation. This fork adapts and rebases that capability for its existing-video extension and later masked bridge workflows.

Main changes:

- Added **seamless extension of existing MP4 videos**, including the new `H3 Existing Video Masked Context`, `H3 Assemble Existing Video Extension`, and `H3 Crop Source To /32` nodes plus a new `Advanced Extension of Input Videos.json` example workflow.
- Added **video overlap blending by default** to the main continuation workflows using KJNodes' `Image Batch Extend With Overlap`; the example workflows now use `linear_blend` to make small reconstruction differences at the seam less visible.
- Changed the main continuation-workflow default to **39 context frames**, which gives an exact H3 video/audio boundary; longer exact-aligned options include 90, 141, 192 and 243 frames.
- Improved the Director Prompt notes used by the main example workflows.
- Added capability-aware ComfyUI compatibility so the required runtime patches only activate when ComfyUI does not already provide the needed functionality.
- Updated all workflows


## Native ComfyUI guide migration — 2026-08-13

ComfyUI PR #15439 is now available in the supported core and supersedes the
fork's classic Motion Context / MultiRef runtime monkey-patches.

Changes in this revision:

- Classic `H3 Motion Context` now emits native `minimax_keyframes` guide data.
  A valid 5/22/39/... frame context is one multi-frame guide block, matching
  stock `MiniMaxH3AddGuide`; off-grid context falls back to native arbitrary
  still-guide positions.
- Timeline Motion Context audio is attached as native keyframe `audio_latent`
  (`cond_audio`) instead of masquerading as a specially marked Ref2VA audio ref.
- Ordinary Ref2VA image/video/audio references remain in `minimax_refs`.
  Current ComfyUI merges guide and reference video/audio latents natively.
- `patch_layout.py` and the old keyframe/ref payload-merge patch were removed.
  Normal Motion Context, MultiRef, and Custom Keyframes no longer monkey-patch
  ComfyUI core.
- `H3 Custom Keyframes` writes real `resolved_frame_index` values directly;
  native PackedLayout supports arbitrary target positions.
- The separate existing-video masked extension still depends on PR #15375-style
  AV denoise-mask support.  Because the current supported ComfyUI build does not
  yet expose those mask hooks natively, that compatibility remains lazy and is
  activated only when the existing-video masked node is executed. Its snapshot
  was rebased onto the #15439 `PackedLayout` signature and `cond_audio` segment.


## Update 3 — Per-Token Noise Masking on Video and Audio Latents


- Corrected the one-video masked-extension example audio path: removed the fixed `TrimAudioDuration` timestamp cut and now drive `H3 Motion Context Trim` from the masked-context node's actual `trim_frames` output, followed by `H3 Assemble Existing Video Extension` for frame/sample-exact source + continuation audio.
- The one-video example now uses VHS for source loading and final exports, with separate `VHS_VideoCombine` outputs for the raw unstitched 192-frame H3 clip and the final 39-frame-linear-blended extension.

Update 3 adds true target-latent AV masking workflows on top of the post-#15439 native H3 guide architecture.

Main changes:

- Added `H3 Masked AV Bridge` for two-ended generation: source-A tail and source-B head are VAE-encoded directly into the target H3 AV latent and protected by nested video/audio denoise masks.
- Added one-video and two-video 192-frame example workflows using 39-frame exact AV boundaries.
- Added static/mock regression tests for bridge construction and example workflow wiring.
- Kept linear visual overlap in the delivered examples to smooth source-pixel vs VAE-reconstruction differences.
- Normal Motion Context remains native #15439 guide conditioning. The #15375-equivalent compatibility path remains lazy and capability-aware, so it is activated only for masked target-latent features when the running ComfyUI core lacks equivalent H3 AV-mask behavior.
- Strengthened #15375 self-retirement detection for AV-mask payload extraction: native payload keys or the native MiniMaxH3 AV-mask hook set suppress the fallback wrapper, without depending on `inspect.getsource()`; added explicit native-payload no-op and legacy-wrapper regression tests.
