# H3 Motion Context — MultiRef with per-token noise masking on audio and video latents and custom keyframes

## 🎬 FL2VA Latent Audio Masking Song Demo
https://github.com/user-attachments/assets/33e22c59-d23f-4470-b52a-6fabb0e4a66b
**Full 72-second generated example.** FL2VA uses two image references while the
original master song is encoded directly into H3's audio latent and protected
with an audio denoise mask of `0`. No `ref_audio_*` input is connected.

The complete reproducible workflow, reference images, song, and lyrics are
included in [`example_workflows/`](example_workflows/).

> **Modified fork.** Original project by [NikoDemon80](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context), licensed GPL-3.0.

This fork originally focused on making H3 Motion Context work cleanly alongside Ref2VA/MultiRef, including timeline-audio context without replacing ordinary references.


## Acknowledgements

The MiniMax H3 video/audio latent masking work in this fork builds on **ComfyUI PR [#15375 — Support per-token video and audio latent noise masks on MiniMax-H3](https://github.com/Comfy-Org/ComfyUI/pull/15375)**, authored by **[Barish Ozbay (`drozbay`)](https://github.com/drozbay)**. That PR introduced the H3-aware per-token video/audio denoise-mask approach that this repo adapts and rebases for its masked existing-video extension and masked AV bridge workflows. Credit for the original H3 AV mask design and upstream implementation belongs to Barish Ozbay.

## Update 4 — FL2VA Exact Song-Latent Masking

- Added **H3 Song Audio + Masked Video Context** (`MiniMaxH3SongMaskedAVContext`) for song-driven FL2VA generation.
- The node slices the original master audio on the project timeline, VAE-encodes that exact interval into H3's target audio latent, and sets the full audio denoise mask to `0` so the sampler changes only the video stream.
- Continuation clips can independently preserve a previous decoded video tail at the head of the target latent while leaving the rest of the video stream denoisable.
- Reference images remain available through `MiniMaxH3ReferenceToVideo`; the song does **not** need to occupy a `ref_audio_*` socket.
- Added a reproducible FL2VA music-video example with two image references, the master-song asset, linear visual overlap, and the untouched master song attached to the final render.
- Added static workflow coverage that verifies the FL2VA checkpoint, disconnected audio-reference sockets, song-latent nodes, reference-image wiring, final master-audio output, and included assets.

See [UPDATE_4_2026-08-14.md](UPDATE_4_2026-08-14.md) and [the reproducible example notes](example_workflows/H3%20FL2VA%20Song%20Latent%20Masking%20-%20README.md).

## Update 3 — Per-Token Noise Masking on Video and Audio Latents

- Added **H3 Masked AV Bridge**, a two-ended MiniMax H3 target-latent bridge built on the repo's PR #15375-style AV denoise-mask compatibility.
- Added a tested **one-video masked extension** example: 39 preserved AV frames + 153 generated frames in a 192-frame target, with frame/sample-exact audio trimming/assembly and separate VHS raw-H3 + stitched outputs.
- Added a tested **two-video masked bridge** example: 39 preserved AV frames + 114 generated middle frames + 39 preserved AV frames.
- Update 3 targets the post-#15439 native H3 guide architecture: normal Motion Context uses native `minimax_keyframes` / `cond_audio`; the #15375 compatibility layer is only used for masked target-latent operations when native equivalent support is absent.
- Endpoint source frames are written into the actual H3 target latent and protected with `0 = preserve`, `1 = generate`; the examples do not use AddGuide for those endpoint windows.
- Added bridge and example-workflow regression checks.

See [UPDATE_3_2026-08-14.md](UPDATE_3_2026-08-14.md) and [H3_MASKED_AV_BRIDGE.md](H3_MASKED_AV_BRIDGE.md).

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

Native functionality is left untouched. If ComfyUI later merges PR #15375 or equivalent behavior, the corresponding compatibility code becomes a no-op automatically. Payload retirement is detected without relying on source-file text: native AV mask output keys or the native MiniMaxH3 AV-mask hook set are sufficient, so stripped builds and ordinary upstream refactors do not keep the fallback wrapper active unnecessarily.

The Motion Context layout compatibility remains repo-specific because it also implements timeline-audio placement and the experimental `before` anchor mode.

## Example workflows

- `Simple Motion Context - No Reference Images.json` — 39-frame visual/timeline-audio default, KJ linear video stitching, hard-trimmed audio.
- `Advanced Motion Context - Reference Images.json` — Ref2VA/MultiRef + Motion Context, same global context/crossfade controls.
- `Music Video Motion Context - Song Driven Lipsync + Reference Images.json` — 39-frame visual-only Motion Context, KJ crossfade, original-song slice architecture. Song-slice start times and durations are calculated automatically from the current H3-valid frame count and visual context length.
- `Advanced Extension of Input Videos.json` — advanced existing-MP4 extension with two character reference images, 39-frame masked AV prefix, KJ linear video overlap and exact hard-joined audio.
- `Custom Keyframes Example.json` — unchanged.
- `H3 Masked AV Extension - One Video Example - 192f.json` — VHS input, 39-frame masked AV prefix, 153 newly generated frames, frame/sample-exact audio assembly, a 39-frame linear visual overlap, plus separate VHS outputs for the raw 192-frame H3 clip and final stitched result.
- `H3 Masked AV Bridge - Two Video Example - 192f.json` — two inputs, 39-frame preserved AV windows at both ends, 114 generated middle frames, and linear visual overlaps at both delivered joins.
- `H3 FL2VA Song Latent Masking - Reference Images - Music Video.json` — FL2VA + two image references + exact master-song audio inserted into the target H3 audio latent with audio denoise mask `0`; later clips independently protect a 39-frame visual prefix, and the final export uses the untouched master song.

See [example_workflows/README.md](example_workflows/README.md).

## Workflow dependencies

The updated continuation examples use:

- **ComfyUI-KJNodes** for `Image Batch Extend With Overlap`,
- **ComfyUI-VideoHelperSuite** for video loading/preview/output where present,
- **rgthree-comfy** for the optional-clip group bypass control in the FL2VA song-latent music-video example.

These are example-workflow dependencies only.

## Important limitations

- Existing-video extension currently targets MiniMax H3 joint video+audio latents and batch size 1.
- Source video should be CFR or decoder-normalized; the included MP4 example forces 24 fps.
- Masked MP4 prefix lengths must be native H3 full video runs (`5`, `22`, `39`, `56`, ...); off-grid requests snap down.
- The masked prefix primarily solves the **join/seam** problem. It does not guarantee that H3 keeps the same composition indefinitely after leaving the preserved prefix.
- Turbo/speedup settings can affect continuation behavior; do not assume one continuation-safe LoRA strength for every source/seed.
- Long recursive generated-audio chains remain lossy. For fixed-song lip-sync, prefer the Update 4 FL2VA song-latent path so every clip is conditioned by the original master timeline rather than recursively generated H3 audio.
- `H3 Song Audio + Masked Video Context` pads with silence if a requested clip extends past the end of the supplied master audio. The included example intentionally lets the final raw clip overrun the song and uses `trim_to_audio` on the final master-song render.

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
- Music Video 39-frame song-slice timeline defaults,
- FL2VA exact-song latent masking and reproducible example-workflow wiring.

## License / upstream

Original project and copyright: **NikoDemon80**. This modified version remains under **GPL-3.0**. See [LICENSE](LICENSE).

Upstream repository: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context

Relevant ComfyUI compatibility reference:

- PR [#15375 — Support per-token video and audio latent noise masks on MiniMax-H3](https://github.com/Comfy-Org/ComfyUI/pull/15375), authored by [Barish Ozbay (`drozbay`)](https://github.com/drozbay).
