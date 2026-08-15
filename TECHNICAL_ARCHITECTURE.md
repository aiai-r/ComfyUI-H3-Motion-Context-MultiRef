# Technical Architecture

This document explains the implementation behind the workflows in this repository. It is intentionally more technical than the main README.

If you only want to choose and run a workflow, start with [README.md](README.md) and [example_workflows/README.md](example_workflows/README.md).

---

## 1. Architecture overview

The repository contains two different continuation families.

### Legacy Motion Context

The previous clip is represented as **native H3 guide conditioning**. The target H3 latent remains a fresh generation target; previous motion/audio is supplied as conditioning that tells H3 what is already happening.

Main node:

- `H3 Motion Context`

Typical workflow label:

- `OLD - Motion Context - ...`

### Current latent masking

Known video/audio content is written directly into the H3 target latent. A per-token noise mask marks the known region as protected and the unknown region as generative.

Main nodes:

- `H3 Existing Video Masked Context`
- `H3 Generated AV Masked Context`
- `H3 Masked AV Bridge`
- `H3 Song Audio + Masked Video Context`

Typical workflow label:

- `NEW - Latent Masking - ...`

The two families can coexist in the same repository because they solve continuation in different ways.

---

## 2. MiniMax H3 joint AV latent

MiniMax H3 works with a joint audiovisual latent. In this repository it is treated as two streams inside the ComfyUI `LATENT` object:

- video latent: approximately `[B, C, T, H, W]`;
- audio latent: approximately `[B, C, 2, T_audio]`.

Current ComfyUI H3 represents these together using a nested tensor-like structure.

The custom nodes therefore avoid assuming that `latent["samples"]` is one ordinary tensor. Helper functions explicitly unpack the video and audio streams.

This matters for:

- target-latent masking;
- checkpoint serialization;
- direct generated-latent continuation;
- master-song audio replacement.

---

## 3. Video and audio clocks

H3 video output is treated as **24 fps**.

H3 audio latents run at **40 latent steps per second**.

Therefore:

```text
40 / 24 = 5 / 3
```

One video frame is not one audio latent step.

### Video-VAE temporal pattern

The H3 video latent uses the repeating pixel-frame coverage pattern:

```text
1, 4, 4, 4, 4
```

The repository exposes this internally as:

```python
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
```

A complete native temporal video run therefore follows the pixel-frame sequence:

```text
5, 22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192, ...
```

which is:

```text
17k + 5
```

for integer `k >= 0`.

### Exact AV boundaries

Some native video runs also land exactly on the 40 Hz audio grid.

Examples:

```text
39 frames  = 1.625 s = 65 audio steps
90 frames  = 3.750 s = 150 audio steps
141 frames = 5.875 s = 235 audio steps
192 frames = 8.000 s = 320 audio steps
```

This is why **39 frames** is the default continuation context throughout the current workflows: it is a valid H3 video-VAE run and an exact video/audio timing boundary.

---

## 4. Why some target clips have a rounded audio tail

A full H3 target does not always end on an exact 40 Hz audio boundary.

Example:

```text
124 video frames / 24 fps = 5.166666... s
5.166666... s * 40 Hz = 206.666666... audio steps
```

The target AV latent may therefore contain:

```text
207 audio latent steps
```

The target latent length is authoritative.

This distinction is important for the master-song workflow because an audio VAE given only the exact picture-duration waveform can, on some encode paths, return 206 steps while the H3 target contains 207.

`H3 Song Audio + Masked Video Context` handles this by:

1. keeping its public `clip_audio` output at the exact picture duration;
2. calculating how much waveform is required to cover the complete target audio grid;
3. encoding that slightly longer real master-song interval when necessary;
4. retrying with a small amount of additional real-audio lookahead if an encoder boundary still floors the output;
5. cropping the encoded latent to the exact target-audio length.

The implementation does not solve the mismatch by inventing/repeating a latent token.

---

## 5. Noise-mask semantics

The latent-masking workflows use per-stream H3 noise masks.

Conceptually:

```text
0 = preserve / do not denoise this known token
1 = denoise / generate this token
```

For a normal continuation:

```text
VIDEO: [ protected previous context ][ generate future ... ]
MASK:  [ 0 0 0 0 ...               ][ 1 1 1 1 ...      ]
```

For an AV continuation:

```text
VIDEO: [ protected AV prefix ][ future ]
AUDIO: [ protected AV prefix ][ future ]
```

both streams can have a protected prefix.

For the master-song music-video workflow:

```text
VIDEO MASK: protected previous visual prefix = 0, future = 1
AUDIO MASK: entire master-song audio latent = 0
```

The song is therefore treated as authoritative target content rather than something H3 must reconstruct through denoising.

---

## 6. `H3 Existing Video Masked Context`

Class:

```text
MiniMaxH3ExistingVideoMaskedContext
```

Purpose: start a latent-masked H3 continuation from a normal decoded video/audio source for which no original H3 sampler latent exists.

The node:

1. normalizes source timing to the H3 24 fps timeline;
2. selects the final requested context window;
3. snaps a masked prefix to a valid H3 video run where required;
4. resizes/crops source frames to the target H3 geometry;
5. VAE-encodes the source video tail;
6. takes the matching physical audio interval and audio-VAE encodes it;
7. writes both encoded streams into the beginning of the fresh target AV latent;
8. creates video/audio masks protecting that prefix;
9. leaves the rest of the target denoisable.

This node is normally used only for the **first** generated extension after an arbitrary uploaded video.

---

## 7. `H3 Generated AV Masked Context`

Class:

```text
MiniMaxH3GeneratedAVMaskedContext
```

Purpose: continue from a previous **generated H3 clip** without decoding and re-encoding its continuation context.

The previous sampler/checkpoint already contains the H3 video/audio latent representation. The node therefore copies the previous clip's final valid AV latent run directly into the next target's prefix.

For a 39-frame continuation window, it derives the corresponding H3 video-latent and audio-latent lengths, copies the tail, and protects the copied prefix with mask `0`.

Advantages:

- no previous-clip video decode → VAE encode round trip for continuation conditioning;
- no previous-clip audio decode → audio VAE encode round trip;
- smaller continuation path;
- exact use of the generated H3 representation.

Source and target latent geometry must match, so chained clips should keep the same H3 resolution/model configuration.

---

## 8. `H3 Masked AV Bridge`

Class:

```text
MiniMaxH3MaskedAVBridge
```

Purpose: generate a missing middle section between two known audiovisual endpoints.

The node places:

- the end of source A into the beginning of the target latent;
- the beginning of source B into the end of the target latent;
- mask `0` on the known endpoint regions;
- mask `1` over the unknown middle.

H3 then generates only the middle region.

The delivered workflow still uses visual overlap treatment at the final source/generated joins because decoded source pixels and VAE-reconstructed generated endpoint pixels can differ slightly even when they represent the same content.

---

## 9. Master-song latent masking

Class:

```text
MiniMaxH3SongMaskedAVContext
```

Display name:

```text
H3 Song Audio + Masked Video Context
```

Purpose: make one original song the authoritative audio timeline for a multi-clip H3 music video.

For each clip the node:

1. inspects the target H3 AV latent;
2. determines the target video duration and target audio-grid length;
3. selects the appropriate interval of the complete master song from `clip_start_seconds`;
4. resamples to the H3 audio-VAE input rate when required;
5. audio-VAE encodes enough waveform to fill the target H3 audio grid;
6. writes that audio latent into the target audio stream;
7. sets the complete audio denoise mask to `0`;
8. optionally inserts/protects previous visual context at the beginning of the target video stream;
9. leaves the new visual region denoisable.

The final delivered music video uses the original master song, not a chain of decoded H3-generated audio clips.

### Clip start timing

For equal-sized raw H3 clips with a protected visual overlap:

```text
raw clip duration = raw_frame_count / 24
context duration  = context_frames / 24
new timeline advance = (raw_frame_count - context_frames) / 24
```

Therefore:

```text
Clip N start = (N - 1) * new timeline advance
```

The song slices overlap by the same physical time interval as the visual continuation context.

---

## 10. Reference images and Ref2VA

Reference images and latent continuation have different jobs.

A reference image primarily defines stable identity/appearance.

The protected continuation context defines the current physical state at the clip boundary:

- pose;
- current expression;
- camera framing;
- camera trajectory;
- motion;
- lighting state;
- environment/object state.

The current workflows therefore allow native `MiniMaxH3ReferenceToVideo` conditioning to coexist with latent-masked continuation.

`H3 Optional Reference Image` is a lazy helper node used by the multi-clip extension workflow. Disabled slots return no image; enabled slots request the connected image and pass it into the normal H3 reference input.

---

## 11. Legacy `H3 Motion Context`

Class:

```text
MiniMaxH3MotionContext
```

The legacy workflows do **not** insert previous content into the target latent. Instead they create native H3 guide/keyframe conditioning.

Current classic Motion Context uses ComfyUI's native H3 guide architecture.

Important concepts:

- previous visual frames become native H3 video guide data;
- timeline audio can become native H3 audio-guide data;
- ordinary Ref2VA references remain in their normal reference-conditioning path;
- ComfyUI combines native guide and reference payloads.

### Native video run vs per-frame fallback

For an exact H3 video run such as 39 frames, Motion Context can VAE-encode the whole temporal guide in one call.

For an arbitrary off-grid context length, the implementation can fall back to per-frame/still-guide representation so the requested endpoint is preserved rather than silently changing the context length.

This is different from target-latent masking, where the protected prefix itself must map cleanly into H3's target temporal latent.

---

## 12. Motion Context trimming and legacy assembly

A classic guide-based continuation repeats the guided visual prefix at the beginning of the generated output.

`H3 Motion Context Trim` removes that repeated region for delivery and can keep a smaller visual overlap specifically for a final blend.

The legacy workflows historically used KJNodes `ImageBatchExtendWithOverlap` to accumulate and blend clips.

That visual blend is valid; the long-form memory problem came from the **cumulative tensor topology**:

```text
clip 1 + clip 2 -> large IMAGE batch
large batch + clip 3 -> larger IMAGE batch
larger batch + clip 4 -> ...
```

The current checkpoint assemblers preserve the blend operation while avoiding the ever-growing ComfyUI image batch.

---

## 13. Checkpoint format

Current long-form workflows can save each completed H3 joint AV sampler output using:

```text
H3 Checkpoint Save
```

The checkpoint is a safetensors file containing at least:

```text
video
audio
```

for the H3 video/audio latent streams.

Fixed clip slots use names like:

```text
clip_00001.safetensors
clip_00002.safetensors
...
```

The save is designed to be atomic so a failed/interrupted write does not silently replace a valid completed checkpoint with a partial file.

### Why save latents rather than intermediate MP4s?

Latents:

- avoid lossy intermediate video encoding;
- are much smaller than decoded full-resolution frame batches;
- preserve the expensive sampler result;
- can be decoded again for final assembly;
- can provide continuation context after restarting ComfyUI.

---

## 14. Lazy resume

Resume is supported at **completed clip boundaries**.

The repository does not attempt to resume a sampler halfway through its diffusion steps.

### Music-video resume

`H3 Resume / Live Tail Frames` has a lazy live input.

Normal run:

```text
use live previous clip
```

Resume run:

```text
load previous completed checkpoint
-> decode only the visual information required for the next continuation
-> do not request the earlier live generation branch
```

Because the live input is lazy, ComfyUI can skip the earlier upstream clip tree when the checkpoint path is selected.

### AV-extension resume

`H3 Resume / Live AV Latent` performs the equivalent job for workflows that want the previous **joint H3 AV latent** rather than decoded tail frames.

This allows the multi-clip masked AV extension to continue directly from a saved previous latent.

---

## 15. Final checkpoint trigger

`H3 Checkpoint Final Trigger` is a lazy dependency helper.

Its job is to make final assembly depend on only the configured last active checkpoint rather than eagerly requesting every optional clip group.

This is especially important in workflows with many available slots but only some active clips.

---

## 16. RAM-safe music-video assembly

`H3 Assemble Checkpoints` assembles the master-song music video from saved H3 checkpoints.

It deliberately does not create one final ComfyUI `IMAGE` output containing the complete movie.

The process is approximately:

```text
load checkpoint 1
-> decode clip 1
-> write completed frames
-> retain only overlap tail
-> release clip 1

load checkpoint 2
-> decode clip 2
-> blend previous tail with current overlap
-> write completed frames
-> retain only next overlap tail
-> release clip 2

...
```

### Linear blend

The implementation matches the source-side KJNodes `linear_blend` convention used by the previous workflow.

For an overlap of `N` frames, alpha values are the interior values of:

```python
linspace(0, 1, N + 2)[1:-1]
```

and each overlap frame is:

```text
(1 - alpha) * previous_tail + alpha * current_overlap
```

The endpoints therefore do not use exact alpha 0 or 1 inside the blended overlap.

Every adjacent clip pair is still blended. For 20 clips there can be 19 seams.

### Output streaming

Decoded images remain floating-point while the seam is calculated. Completed frames are then quantized to RGB24 at the final ffmpeg streaming boundary.

The final H.264 encode happens once.

For the master-song workflow, ffmpeg muxes the original master audio.

---

## 17. RAM-safe existing-video extension assembly

`H3 Assemble Extension Checkpoints` performs the corresponding job for normal audiovisual extension.

Update 5 additionally introduces `H3 Start Masked Context`, `H3 Start Canvas Selector`, and `H3 Assemble Starter + Extension Checkpoints` so the same multi-clip masked-AV workflow can either begin from an uploaded source video or from a generated starter clip (pure T2V or I2V). The starter path is also checkpointed and assembled sequentially, so it keeps the same low-memory / low-OOM behavior as the source-video path.

Unlike the music-video workflow, generated H3 audio is part of the delivered continuation.

The assembler therefore:

- starts with the original source video/audio;
- loads generated checkpoints sequentially;
- decodes video one clip at a time;
- blends every visual seam;
- decodes generated audio;
- removes the duplicated protected AV prefix from the delivered continuation audio;
- fits audio segments to the exact frame-derived sample timeline so small latent-grid rounding differences do not accumulate;
- streams the final result to ffmpeg.

---

## 18. Why checkpoint assembly fixes the long-video RAM problem

A decoded ComfyUI `IMAGE` batch is normally a floating-point tensor.

A long sequence at high resolution can therefore consume many gigabytes even before additional intermediate tensors are considered.

With cumulative overlap nodes, multiple increasingly large intermediate batches can coexist because they are graph outputs/cacheable values.

Sequential checkpoint assembly changes the memory shape from approximately:

```text
all clips + all cumulative intermediate movies
```

to:

```text
one decoded clip + one small overlap tail + encoder/runtime overhead
```

Disk becomes the durable intermediate store, while RAM remains bounded by the current clip rather than total movie length.

---

## 19. Runtime compatibility layers

The repository has historically supported ComfyUI versions at different stages of H3 feature development.

The guiding rule now is:

> Use native ComfyUI behavior when the required capability exists. Install compatibility behavior only for the specific missing capability needed by the executed node.

### Native guide compatibility

`h3_compat.py` covers the classic Motion Context/native-guide requirements.

The current capability check verifies live behavior rather than depending on one exact source-code spelling.

It also checks only capabilities relevant to the current conditioning:

- simple Motion Context without refs does not require Ref2VA merge behavior;
- video/image refs require the video merge behavior;
- audio refs require the audio merge behavior.

This avoids false negatives such as the environment reported in repository Issue #7.

### AV-mask compatibility

`h3_mask_compat.py` and `h3_mask_payload_compat.py` cover the target-latent video/audio denoise-mask path.

The mask layer is kept separate from normal Motion Context so using a legacy guide workflow does not unnecessarily install masked-target compatibility behavior.

Detection is capability-oriented so newer native ComfyUI support can make the fallback path retire itself.

---

## 20. Main node reference

### Guide/legacy nodes

| Display name | Purpose |
|---|---|
| H3 Motion Context | Native H3 previous-motion/audio guide conditioning |
| H3 Motion Context Trim | Remove repeated guide prefix and produce delivery overlap |
| H3 Motion Context Save Latent | Older Motion Context latent persistence helper |
| H3 Motion Context Load Latent | Older Motion Context latent load helper |
| H3 Custom Keyframes | Native H3 still-image anchors at chosen timeline positions |

### Latent-masking nodes

| Display name | Purpose |
|---|---|
| H3 Existing Video Masked Context | Start masked continuation from ordinary decoded source AV |
| H3 Generated AV Masked Context | Continue from a previous generated H3 AV latent directly |
| H3 Masked AV Bridge | Protect source A/B endpoints and generate the middle |
| H3 Song Audio + Masked Video Context | Put the master-song interval into target audio latent and optionally protect previous visual context |
| H3 Optional Reference Image | Lazy optional global reference-image slot |
| H3 Crop Source To /32 | Prepare source image geometry for H3 workflows |

### Checkpoint/resume/assembly nodes

| Display name | Purpose |
|---|---|
| H3 Checkpoint Save | Atomic fixed-slot H3 AV latent checkpoint |
| H3 Checkpoint Load | Load a saved checkpoint as a decodable H3 latent |
| H3 Resume / Live Tail Frames | Lazy live-vs-checkpoint visual continuation selector |
| H3 Resume / Live AV Latent | Lazy live-vs-checkpoint joint AV latent selector |
| H3 Checkpoint Final Trigger | Lazy final active checkpoint dependency |
| H3 Assemble Checkpoints | Sequential master-song video assembly from checkpoints |
| H3 Assemble Extension Checkpoints | Sequential existing-video + generated AV assembly |
| H3 Start Masked Context | First-extension selector: source-video prefix or generated-starter latent tail |
| H3 Start Canvas Selector | Shared width/height selector for source-video vs generated-starter starts |
| H3 Assemble Starter + Extension Checkpoints | Sequential assembly for either source-video or generated-starter chains |
| H3 Assemble Existing Video Extension | Single-extension source + continuation assembly helper |

---

## 21. Workflow categories

The filenames communicate architecture rather than release status.

### `NEW - Latent Masking - ...`

Current recommended target-latent workflows.

### `OLD - Motion Context - ...`

Legacy native-guide workflows. They are intentionally kept available and regression-tested.

### `OLD - Hybrid - ...`

Mixed architecture, generally retained for comparison/backward compatibility.

### `UTILITY - ...`

Small feature examples.

### Master-song music video

```text
NEW - Latent Masking - Music Video - Lip-Sync + Reference images.json
```

Use for long-form lip-sync/song-driven generation with reference images, checkpoints, resume, and original-master final audio.

### Existing-video multi-clip extension

```text
NEW - Latent Masking - AV Extension - Multiple Clips + Reference Images.json
```

Use for extending one uploaded source through multiple generated H3 clips, optionally with global reference images.

### Minimal extension

```text
NEW - Latent Masking - AV Extension - Minimal Single Clip.json
```

Use to understand/debug the first source-video masked continuation.

### Two-video bridge

```text
NEW - Latent Masking - AV Bridge - Two Videos.json
```

Use when both endpoints already exist and the missing content should be generated between them.

---

## 23. Testing

The repository contains CPU/static/mock regression tests because full H3 inference requires the user's ComfyUI model/runtime installation.

The tests cover, among other things:

- native Motion Context structure;
- Simple and Advanced legacy workflow wiring;
- native guide/reference compatibility detection;
- per-token AV mask capabilities;
- existing-video extension behavior;
- direct generated-AV latent continuation;
- master-song audio masking;
- the one-token audio-grid boundary regression;
- checkpoint save/load and lazy resume;
- workflow JSON consistency;
- checkpoint assembly logic.

Run the repository test runner with:

```bash
python tests/run_update2_tests.py
```

The historical filename of the runner is retained even though it now covers later updates too.

---

## 24. Additional focused references

- [MODIFICATIONS.md](MODIFICATIONS.md) — release history.
- [example_workflows/README.md](example_workflows/README.md) — the single workflow chooser, setup guide, reproduction guide, and per-workflow reference.
