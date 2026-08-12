# Fork history and modifications

This repository is a modified version of **NikoDemon80/ComfyUI-H3-Motion-Context**.

- Upstream: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context
- Original author: NikoDemon80
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

Update 2 was **heavily inspired by ComfyUI pull request #15375 by drozbay**, especially its H3-aware video/audio denoise-mask approach for preserving an existing AV prefix while generating the continuation.

Main changes:

- Added **seamless extension of existing MP4 videos**, including the new `H3 Existing Video Masked Context`, `H3 Assemble Existing Video Extension`, and `H3 Crop Source To /32` nodes plus a new `Advanced Extension of Input Videos.json` example workflow.
- Added **video overlap blending by default** to the main continuation workflows using KJNodes' `Image Batch Extend With Overlap`; the example workflows now use `linear_blend` to make small reconstruction differences at the seam less visible.
- Changed the main continuation-workflow default to **39 context frames**, which gives an exact H3 video/audio boundary; longer exact-aligned options include 90, 141, 192 and 243 frames.
- Improved the Director Prompt notes used by the main example workflows.
- Added capability-aware ComfyUI compatibility so the required runtime patches only activate when ComfyUI does not already provide the needed functionality.
- Updated all workflows
