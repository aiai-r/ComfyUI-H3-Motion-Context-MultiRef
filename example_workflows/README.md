# Example workflows

If you are new to the repository, start with one of the **Current — Latent Masking** workflows below.

The filename prefixes describe the architecture:

- **`NEW - Latent Masking - ...`** — current/recommended continuation method;
- **`OLD - Motion Context - ...`** — legacy guide-based continuation;
- **`OLD - Hybrid - ...`** — mixes both approaches;
- **`UTILITY - ...`** — small feature examples.

`OLD` means **legacy architecture**, not “broken.”

---

## Current — Latent Masking

### `NEW - Latent Masking - Music Video - Lip-Sync + Reference images.json`

**Recommended for long-form music videos.**

Use it when you want:

- one original song as the exact master timeline;
- character/reference images;
- up to 20 generated clips;
- fixed seeds;
- checkpoints and clip-boundary resume;
- temporary per-clip VHS previews that can be bypassed globally;
- linear blending across every clip seam;
- RAM-safe final assembly with direct in-node preview of the already-written MP4;
- the original song as the final soundtrack.

The full Director Prompt and exact audible-word alignment instructions are embedded in the large note at the top of the workflow itself.

### `NEW - Latent Masking - AV Extension - Multiple Clips + Reference Images.json`

**Recommended for either extending an existing video or generating a T2V/I2V starter clip and extending that through several H3 clips.**

- Uses one global start-mode switch: `load_video` or `generate_starter`.
- `generate_starter` can run as pure T2V or as I2V by enabling the optional starter image.
- Can generate multiple sequential extensions after the initial source/starter clip.
- Reference images are optional.
- Later clips continue from the previous generated H3 AV latent directly.
- The generated starter is checkpointed before Extension 1, and every extension is checkpointed too. Both start modes use the same RAM-safe sequential final assembly.
- Checkpoints allow clip-boundary resume.
- Final assembly is streamed sequentially instead of building one huge image batch.

See the detailed workflow guide later in this README for controls, checkpoint/resume behavior, previews, and final assembly.

### `NEW - Latent Masking - AV Extension - Minimal Single Clip.json`

Use this when learning or debugging the masked-extension architecture.

It demonstrates one source video, one protected continuation prefix, and one generated continuation.

### `NEW - Latent Masking - AV Bridge - Two Videos.json`

Use this when you already have both endpoints.

The workflow protects the end of the first video and the beginning of the second, then lets H3 generate the missing middle.

---

## Legacy — Motion Context

These workflows use native H3 guide conditioning rather than target-latent masking.

They are retained for existing projects, comparison, and experimentation.

### `OLD - Motion Context - Simple - No Reference Images.json`

The simplest classic Motion Context chain. No Ref2VA character images.

### `OLD - Motion Context - Advanced - Reference Images.json`

Classic Motion Context combined with Ref2VA/MultiRef reference images.


---

## Legacy hybrid

### `OLD - Hybrid - Input Video Extension + Motion Context - Reference Images.json`

Starts the source-video extension with latent masking, then uses classic Motion Context for later generated clips.

It is retained mainly for backward compatibility and comparison. For new work, prefer the multi-clip latent-masked AV extension.

---

## Utility

### `UTILITY - Custom Keyframes Example.json`

Demonstrates H3 Custom Keyframes / timeline image anchors.

---

## Common dependencies

Different examples use different third-party nodes.

- **ComfyUI-VideoHelperSuite** — video file loading, previewing, and some exports.
- **ComfyUI-KJNodes** — present in some legacy workflows and optional attention/utility paths.
- **rgthree-comfy** — used by workflows with group bypass/switch controls.

If a workflow opens with missing nodes, install the node pack named by ComfyUI and restart the application.

The new streamed checkpoint assemblers perform their own final linear overlap blending; they do not require a cumulative KJ image-stitching chain.

---

## Context timing

The current examples default to **39 continuation frames** because this is a convenient H3-native video run and an exact video/audio timing boundary.

You do not need to understand the H3 temporal grid to use the workflows. If you want the details, see:

[../TECHNICAL_ARCHITECTURE.md](../TECHNICAL_ARCHITECTURE.md)

---

# Detailed workflow guides

The sections below replace the old per-workflow README/guide files. Keep this file open as the single guide for choosing, configuring, reproducing, and extending the included workflows.


## Detailed guide — AV Extension — Multi Clip + Video/T2V/I2V Start + Optional References

### NEW — Latent Masking — AV Extension — Multi Clip + Video/T2V/I2V Start + Optional References

This is the current multi-clip masked-AV extension example.

A single **GLOBAL START MODE** switch chooses one of two ways to start the chain:

- **load_video** — extend an uploaded source video.
- **generate_starter** — generate the first clip with **T2V** or **I2V**, checkpoint it, then extend that starter clip.

The start-mode control also gates the unused branch **before ComfyUI prompt validation**: generated-starter mode does not require a placeholder VHS video, and load-video mode does not validate the unused starter-generation branch.

It is deliberately different from the classic **OLD — Motion Context** workflows: continuation after the first generated extension stays in H3 joint AV latent space instead of converting the previous clip into Motion Context guide conditioning.

#### Architecture

1. **Chain start → Extension 1**
   - `load_video` mode uses `VHS_LoadVideo` + `H3 Crop Source To /32`, then `H3 Start Masked Context` pulls the final source-video AV window into the beginning of Extension 1's target latent.
   - `generate_starter` mode uses the new `STARTER — T2V / I2V SOURCE CLIP` branch. The starter uses the FL2VA path. Leave **STARTER FIRST FRAME** disabled for pure T2V, or enable it and connect a Load Image node for I2V. The starter clip is saved with `H3 Checkpoint Save`, and `H3 Start Masked Context` then continues from that starter clip's final H3 AV latent tail.
   - `H3 Start Canvas Selector` lets the whole chain use either the cropped source-video size or a manual starter width/height.
   - In both modes, the protected prefix receives video mask `0` and audio mask `0`; only the future region denoises.

2. **Extension 2+ → latent-to-latent continuation**
   - `H3 Resume / Live AV Latent` supplies the previous generated H3 checkpoint/live latent.
   - `H3 Generated AV Masked Context` copies the previous H3 clip's final valid video/audio latent run directly into the next target prefix.
   - No previous-clip VAE decode/re-encode is used for continuation conditioning.
   - The copied video/audio prefix is protected with mask `0`; the future region remains mask `1`.

3. **Optional Ref2VA image references**
   - The workflow uses native `MiniMaxH3ReferenceToVideo` targets for every extension.
   - The two purple `H3 Optional Reference Image` nodes are disabled by default.
   - Add a normal `Load Image` node to either purple node and enable that slot to make the reference global across all generated extensions.
   - When references are disabled, those inputs return `None` and the chain operates without image references.

4. **Checkpoints and resume**
   - In `load_video` mode, every completed extension is atomically saved as `h3_extension_checkpoints/clip_0000N.safetensors`.
   - In `generate_starter` mode, the starter clip is additionally saved as `h3_extension_checkpoints/starter_00001.safetensors`.
   - The workflow defaults to **Extension 1 only**. Extensions 2–6 are bypassed by default.
   - Use **OPTIONAL EXTENSIONS 2–6 — ENABLE SEQUENTIALLY** to activate later extension groups.
   - Set `GLOBAL ACTIVE EXTENSION COUNT` to the highest enabled extension number. For example, if Extensions 1–3 are enabled, set the count to `3`. The switch controls group bypass state; the count tells the lazy checkpoint trigger and streamed assembler where the active chain ends.
   - To resume after a crash while generating Extension N, leave the completed checkpoints in place and set `GLOBAL RESUME FROM EXTENSION` to `N`. The Extension N lazy selector loads checkpoint `N-1` and does not request the earlier live generation tree.
   - To resume Extension 1 in `generate_starter` mode without regenerating the starter clip, set `GLOBAL RESUME FROM EXTENSION = 1`; `H3 Start Masked Context` loads the saved starter checkpoint lazily.

5. **Final assembly**
   - `H3 Assemble Starter + Extension Checkpoints` reads either the original source clip or the saved starter checkpoint plus the selected saved extension checkpoints.
   - It decodes only one generated clip at a time, including the starter clip when `generate_starter` is active.
   - Every visual seam is linearly blended using the same interior-alpha convention used by KJNodes' source-side `linear_blend` mode.
   - The protected AV prefix is removed from each generated clip's delivered audio, then generated audio is sample-fitted to the exact delivered frame timeline.
   - The finished RGB frames are streamed to ffmpeg; the workflow never creates one cumulative full-movie IMAGE tensor.
   - In `load_video` mode, VideoHelperSuite still decodes the uploaded source video itself, so a very long/high-resolution source can use RAM proportional to that input. Generated extension memory does not accumulate with clip count.

#### Recommended defaults

- H3 output fps: `24`
- protected AV context: `39` frames
- visual overlap: `39` frames
- fixed seeds: enabled
- active extensions: `1` by default
- Extensions 2–6: bypassed by default; enable sequentially with the rgthree group switch
- same resolution for all generated extensions

`39` frames is recommended because it is both an H3-valid video run and an exact boundary on the 40 Hz H3 audio-latent grid.

#### OLD vs NEW naming

`OLD - Motion Context - ...` means the workflow uses the classic guide-conditioning Motion Context architecture. It does **not** mean the workflow is intentionally broken.

`NEW - Latent Masking - ...` means continuation is implemented by writing protected data into the H3 target latent and using per-stream/per-token denoise masks.


##### One start-mode control

`GLOBAL START MODE` is the only start-source choice. Its user-facing options are `start with T2V/I2V` and `Start from existing video`. Selecting T2V/I2V automatically bypasses the source-video group before queueing so `VHS_LoadVideo` is not submitted or validated; selecting existing video enables that group. The node still sends the internal `generate_starter` / `load_video` value to the H3 context, canvas, and final assembler.

#### Preview behavior

- Each generated extension clip now has its own preview output node (`PREVIEW — Extension N Clip`).
- Those preview nodes decode only one saved checkpoint at a time and automatically skip extension slots above `GLOBAL ACTIVE EXTENSION COUNT`, so inactive extension placeholders do not try to load stale checkpoints.
- The final streamed assembler node now also shows a preview of the fully assembled MP4 directly from the encoded output file. This avoids decoding the full movie back into a giant IMAGE batch just for previewing.


## Detailed guide — AV Extension — Minimal Single Clip

### H3 Masked AV Extension — One Video, 192 Frames

One-sided PR #15375-style H3 AV masked continuation.

- Source input uses `VHS_LoadVideo`, forced to 864×480 at 24 fps.
- The final 39 source frames and matching audio are VAE-encoded directly into the beginning of the target H3 AV latent.
- Those target latent regions use `noise_mask = 0` and remain preserved.
- The remaining 153 frames use `noise_mask = 1` and are generated.
- `H3 Motion Context Trim` receives the actual `trim_frames` value from `H3 Existing Video Masked Context`; it removes the preserved head from both decoded picture and sound and clamps H3's rounded audio tail to the exact remaining frame duration.
- `H3 Assemble Existing Video Extension` builds the delivered audio from exact source audio + exact 153-frame continuation audio, preventing H3 audio-grid rounding from accumulating at the join.
- The delivered picture keeps the required 39-frame KJNodes `linear_blend` overlap.
- No AddGuide endpoint conditioning is used.

Timeline:

```text
raw decoded H3 clip:     39 preserved + 153 generated = 192 frames = 8.000 s
final stitched addition:                  153 frames = 6.375 s new footage
```

#### VHS outputs

The example intentionally has two `VHS_VideoCombine` outputs:

1. `video/H3_masked_AV_extension_raw_H3_192f`
   - the full decoded 192-frame H3 result **before source stitching**
   - includes the protected 39-frame source-derived head
   - uses `H3 Motion Context Trim` with `trim_frames = 0` only to clamp decoded audio to exactly 192 / 24 = 8.000 seconds

2. `video/H3_masked_AV_extension_stitched`
   - full normalized source video + the 153 new continuation frames
   - picture uses the 39-frame KJ linear overlap
   - audio uses frame/sample-exact hard assembly from `H3 Assemble Existing Video Extension`

The fixed `Trim Audio Duration 1.625 → 6.375 s` path used by the earlier draft example has been removed. Audio timing now follows the actual preserved-frame count and the decoded waveform sample rate.


## Detailed guide — AV Bridge — Two Videos

### H3 Masked AV Bridge — Two Videos, 192 Frames

Two-ended PR #15375-style H3 AV masked bridge.

- Both inputs are normalized to 864×480 at 24 fps.
- The final 39 AV frames of source A are written into the beginning of the target latent.
- The first 39 AV frames of source B are written into the end of the target latent.
- Both endpoint regions use `noise_mask = 0`; only the middle uses `noise_mask = 1`.
- The delivered video keeps 39-frame KJNodes `linear_blend` overlaps at both joins.
- Endpoint AddGuide conditioning is not used.

Timeline:

```text
39 preserved + 114 generated + 39 preserved = 192 H3 frames
```

The generated middle is 4.75 seconds at 24 fps.


## Detailed guide — Music Video — Lip-Sync + Reference images

### Reproducible bundled music-video example

The current checkpoint/resume workflow ships preconfigured with the bundled music-video demo used to validate **exact master-song audio latent masking** with **reference images** and MiniMax H3.

#### What is special about this workflow

- Model checkpoint: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- Two persistent character reference images are supplied through `MiniMaxH3ReferenceToVideo` as `<Picture 1>` and `<Picture 2>`.
- No song audio is connected to any `ref_audio_*` input.
- The original song is loaded once and passed into `H3 Song Audio + Masked Video Context` (`MiniMaxH3SongMaskedAVContext`).
- The node VAE-encodes the exact master-song interval into the H3 audio latent and sets the audio denoise mask to **0 for the full raw clip**.
- Continuation clips also preserve the prior decoded video tail with a video denoise mask of **0 only over the visual context prefix**.
- H3 therefore denoises the new visual region while the actual song already occupies the joint AV latent.
- The final soundtrack remains the untouched original master song.
- The workflow keeps the tested six-clip demo prompts, sampler settings, reference-image wiring, timing math, persistent checkpoints, and linear visual blending across every seam.

#### Included assets

Copy these three files from `example_workflows/assets/` into your ComfyUI `input/` directory before loading the workflow:

- `be6f4e89-4c3e-43e0-93f5-cc723ccd9b14.png` — face / identity reference (`Picture 1`)
- `c90ee577-98eb-4f6c-9b0c-562a6b448d69.png` — full-body / wardrobe reference (`Picture 2`)
- `I'll Know You by the Scar.wav` — original master song

`lyrics.txt` is included for reference when editing or extending the prompts.


#### Demo defaults

`NEW - Latent Masking - Music Video - Lip-Sync + Reference images.json` is preconfigured with the included assets and the original authored six-clip demo prompts. Its default active chain is Clips 1–6, with Clips 7–20 bypassed as generic template slots.

For reproducibility, the original numeric demo seeds are retained but set to **fixed**. This allows the persistent signed-checkpoint gate to reuse unchanged clips on later queues.

#### Required custom node

This workflow requires the node added by this PR:

**H3 Song Audio + Masked Video Context**
Class: `MiniMaxH3SongMaskedAVContext`

It also relies on the repository's Update 3 per-token H3 video/audio masking compatibility.

#### Important implementation detail

The graph intentionally uses `MiniMaxH3ReferenceToVideo` to retain the two image references, but the loaded H3 checkpoint is the **FL2VA** checkpoint. The song itself is **not** used as Ref2VA audio conditioning: every `ref_audio_*` socket is disconnected. The master song is instead inserted directly into the target H3 audio latent and protected from denoising.

#### Master audio timing

At 24 fps with the demonstrated 362-frame raw generation and 39-frame visual context:

- raw clip duration: `362 / 24 = 15.083333 s`
- protected visual context: `39 / 24 = 1.625 s`
- new timeline progression: `(362 - 39) / 24 = 13.458333 s`

The workflow computes later clip starts from its existing timing/math nodes; do not manually offset the master audio independently.

#### Final output

Use the full loaded master song as the final soundtrack. Per-clip H3 audio decode nodes are debug/inspection only; they are not the authoritative final soundtrack.

The full music-video Director Prompt is embedded in the large note at the top of `NEW - Latent Masking - Music Video - Lip-Sync + Reference images.json`.
