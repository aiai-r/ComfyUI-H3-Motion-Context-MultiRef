# H3 Motion Context — MultiRef + Existing Video Extension

> **Modified fork.** Original project by [NikoDemon80](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context), licensed GPL-3.0.

This fork originally focused on making H3 Motion Context work cleanly alongside Ref2VA/MultiRef, including timeline-audio context without replacing ordinary references.

## Update 2 — 2026-08-11

- **Existing video extension** — seamlessly extend an already existing input video by preserving its final video/audio context and generating the continuation from that point.
- **Smoother seams by default** — the continuation workflows use KJNodes **Image Batch Extend With Overlap** with `linear_blend` to make small reconstruction differences at clip joins less visible.
- Added a more **capability-aware runtime patching architecture** that only activates the H3 compatibility behavior ComfyUI is actually missing, while avoiding unnecessary or conflicting patches when equivalent native support is already present.

## Update 1 — 2026-08-10

Update 1 introduced **H3 Custom Keyframes**, allowing still-image conditioning anchors at arbitrary positions in the H3 timeline.

See [MODIFICATIONS.md](MODIFICATIONS.md) for more information about the fork history and each update.

## Install

Clone into your ComfyUI `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef.git
```

Restart ComfyUI and hard-refresh the browser.

## Context length

The updated continuation examples expose one **GLOBAL CONTEXT FRAMES** control and a separate **GLOBAL VIDEO CROSSFADE** control. Both default to `39`.

H3-native temporal video runs through 243 frames are:

```text
5, 22, 39*, 56, 73, 90*, 107, 124, 141*, 158, 175, 192*, 209, 226, 243*
* exact video+audio boundary
```

`39` is recommended because at 24 fps it is exactly `1.625 s`, which is exactly `65` H3 audio-latent steps at 40 Hz. `90`, `141`, `192`, `243`, ... have the same exact AV-boundary property with progressively longer context.

Classic Motion Context accepts any positive frame count. Native run lengths are more efficient in `video` encode mode; off-grid lengths automatically use exact per-frame conditioning.

The existing-MP4 masked-prefix node is stricter because its preserved prefix is written directly into the target H3 temporal latent. Off-grid requests therefore snap down to the nearest native full video run.

## Existing MP4 / video extension

The existing-video path is intended for the **first continuation from an arbitrary decoded video**, where no original H3 sampler latent is available.

Recommended defaults:

```text
source FPS:       24 fps
source size:      crop down to width/height divisible by 32
context_length:   39 frames
video crossfade:  39 frames
final FPS:        24
```

### H3 Crop Source To /32

Center-crops a source IMAGE batch down to the nearest width and height divisible by 32 and outputs the cropped frames plus target width/height.

### H3 Existing Video Masked Context

Takes decoded source frames/audio plus a target H3 AV latent. It:

1. canonicalizes the source to H3's 24 fps timeline,
2. selects an exact native H3 prefix length (`5`, `22`, `39`, `56`, ...),
3. extracts matching video and audio tails,
4. VAE-encodes them into the beginning of the target H3 AV latent,
5. creates nested video/audio denoise masks where the known prefix is preserved and only the future is generated.

The node uses native H3 AV-mask support when available. On older ComfyUI builds it lazily installs only the missing #15375-equivalent compatibility pieces.

### H3 Assemble Existing Video Extension

Assembles source + continuation after the generated overlap has been hard-trimmed. Audio is fit to exact integer-sample durations before concatenation to prevent cumulative AV drift.

The included MP4 workflow uses the assembler's exact hard-cut audio while KJNodes handles the final video overlap.

## Final video stitching

Update 2 extends **H3 Motion Context Trim** without changing its original first two outputs:

- `images` — fully hard-trimmed continuation image stream,
- `audio` — fully hard-trimmed/matched continuation audio,
- `crossfade_images` — video stream retaining only the requested final matching overlap,
- `crossfade_frames` — effective overlap clamped to the actual context length.

This allows context and video crossfade lengths to differ safely. Example: `context=90`, `crossfade=39` drops the older 51 duplicated context frames, keeps only the final 39 matching frames for the blend, and still trims all 90 frames from generated audio.

Recommended video stitch uses KJNodes **Image Batch Extend With Overlap** with:

```text
overlap_side: source
overlap_mode: linear_blend
```

Audio is deliberately **not crossfaded**. The repeated generated audio prefix is removed and continuation audio is joined at the exact sample boundary.

KJNodes is a workflow dependency only; the Python package does not import it.

## Classic H3 Motion Context

Classic Motion Context remains separate from the existing-video masked-prefix mechanism.

Recommended continuation default:

```text
context_length:       39
encode_mode:          video
anchor_mode:          head
audio_context_length: 0   # follows context_length
audio_mode:          timeline
```

For H3-to-H3 chaining, `context_latent` can provide the previous sampler's joint AV latent for direct audio-tail reuse, avoiding repeated audio decode/re-encode.

For native H3 video-run lengths such as `39` or `90`, `video` mode VAE-encodes the temporal run once. For other exact frame counts, Update 2 automatically falls back to the existing per-frame representation rather than moving the context endpoint.

## Ref2VA + timeline audio

A graph may contain ordinary Ref2VA refs before the Motion Context timeline-audio ref. Motion Context appends its marked timeline-audio ref last.

The payload compatibility layer preserves keyframe video latents together with ref video/audio latents instead of allowing the refs branch to overwrite the keyframe list.

## H3 Custom Keyframes

The Custom Keyframes workflow/node is unchanged in Update 2.

It supports arbitrary-position still-image H3 conditioning anchors and keeps the existing dynamic **+ Add keyframe / - Remove keyframe** UI.

## Capability-aware compatibility

Update 2 does not use ComfyUI version numbers to decide whether to patch.

At feature execution time it checks the live implementation for the specific capabilities needed:

- keyframe + ref payload composition,
- H3 video/audio mask conditions in `MiniMaxH3.extra_conds`,
- H3-specific mask preprocessing,
- H3-specific inpaint scaling,
- the H3 diffusion-model per-row mask engine.

Native functionality is left untouched. If ComfyUI later merges PR #15375 or equivalent behavior, the corresponding compatibility code becomes a no-op automatically.

The Motion Context layout compatibility remains repo-specific because it also implements timeline-audio placement and the experimental `before` anchor mode.

## Example workflows

- `Simple Motion Context - No Reference Images.json` — 39-frame visual/timeline-audio default, KJ linear video stitching, hard-trimmed audio.
- `Advanced Motion Context - Reference Images.json` — Ref2VA/MultiRef + Motion Context, same global context/crossfade controls.
- `Music Video Motion Context - Song Driven Lipsync + Reference Images.json` — 39-frame visual-only Motion Context, KJ crossfade, original-song slice architecture. Song-slice start times and durations are calculated automatically from the current H3-valid frame count and visual context length.
- `Advanced Extension of Input Videos.json` — advanced existing-MP4 extension with two character reference images, 39-frame masked AV prefix, KJ linear video overlap and exact hard-joined audio.
- `Custom Keyframes Example.json` — unchanged.

See [example_workflows/README.md](example_workflows/README.md).

## Workflow dependencies

The updated continuation examples use:

- **ComfyUI-KJNodes** for `Image Batch Extend With Overlap`,
- **ComfyUI-VideoHelperSuite** for video loading/preview/output where present.

These are example-workflow dependencies only.

## Important limitations

- Existing-video extension currently targets MiniMax H3 joint video+audio latents and batch size 1.
- Source video should be CFR or decoder-normalized; the included MP4 example forces 24 fps.
- Masked MP4 prefix lengths must be native H3 full video runs (`5`, `22`, `39`, `56`, ...); off-grid requests snap down.
- The masked prefix primarily solves the **join/seam** problem. It does not guarantee that H3 keeps the same composition indefinitely after leaving the preserved prefix.
- Turbo/speedup settings can affect continuation behavior; do not assume one continuation-safe LoRA strength for every source/seed.
- Long recursive generated-audio chains remain lossy. For fixed-song lip-sync, the Music Video workflow continues to use original-song timeline slices instead.

## Development / regression checks

Run the CPU/static Update 2 suite without importing ComfyUI itself:

```bash
python tests/run_update2_tests.py
```

The suite covers:

- the original MultiRef timeline-audio structure checks,
- exact 39-frame / 65-audio-step existing-video prefix construction,
- exact final frame/sample accounting,
- keyframe + ref payload composition,
- AV mask condition extraction,
- idempotent/self-healing wrapper installation,
- native-capability no-op behavior,
- H3 timing helpers and preferred AV boundaries,
- updated workflow context/crossfade wiring,
- Music Video 39-frame song-slice timeline defaults.

## License / upstream

Original project and copyright: **NikoDemon80**. This modified version remains under **GPL-3.0**. See [LICENSE](LICENSE).

Upstream repository: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context

Relevant ComfyUI compatibility reference:

- PR #15375 — MiniMax H3 AV latent denoise-mask / inpainting support: https://github.com/Comfy-Org/ComfyUI/pull/15375
