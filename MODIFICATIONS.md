# Fork history and modifications

This repository is a modified version of **NikoDemon80/ComfyUI-H3-Motion-Context**.

- Original project: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context
- Original author: **NikoDemon80**
- License: **GPL-3.0**
- H3 AV masking design credit: **Barish Ozbay (`drozbay`)**, author of ComfyUI PR #15375

Low-level implementation details live in [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md).

---

## Initial fork — MultiRef + Motion Context coexistence

The fork began by adapting H3 Motion Context for workflows that also used MiniMax H3 reference conditioning.

Main goals included preserving normal Ref2VA/MultiRef references while carrying previous visual/audio context into continuation clips, and keeping compatibility behavior inside the custom-node pack rather than permanently modifying ComfyUI files on disk.

---

## Update 1 — Custom keyframes — 2026-08-10

Merged as [PR #1](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/pull/1).

### Main changes

- Added **H3 Custom Keyframes**.
- Added arbitrary visual keyframe positions on the H3 target timeline.
- Added the Custom Keyframes example workflow.
- Added lazy runtime compatibility installation for the functionality required at the time.

---

## Update 2 — Existing-video extension and compatibility improvements — 2026-08-12

Merged as [PR #2](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/pull/2).

### Main changes

- Added **H3 Existing Video Masked Context**.
- Added **H3 Assemble Existing Video Extension**.
- Added **H3 Crop Source To /32**.
- Added the advanced input-video extension workflow with optional Ref2VA/MultiRef images.
- Added configurable continuation context and visual overlap blending.
- Improved music-video song-slice timing.
- Reworked compatibility handling to be capability-aware: use native ComfyUI behavior when available and activate local compatibility only when required.
- Added CPU/static regressions for timing, masking, assembly, and workflow structure.

Update 2 introduced the first practical source-video latent-masking path. Later updates expanded that idea into general per-token AV masking and long multi-clip chains.

---

## Compatibility milestone — native H3 guide architecture — 2026-08-13

This was an important backend migration, but **not a separate numbered update PR**.

As ComfyUI gained native H3 arbitrary guide/reference coexistence, the classic Motion Context path was moved away from the older layout/payload monkey-patching design and toward native H3 keyframe/audio-guide conditioning.

The legacy Motion Context workflows therefore remain guide-based workflows, but they use the newer native H3 mechanisms where available. See [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) for details.

---

## Update 3 — Per-token video/audio latent masking — 2026-08-14

Merged as [PR #4](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/pull/4).

### Main changes

- Added per-token noise masking for MiniMax H3 video and audio latents.
- Added **H3 Masked AV Bridge**.
- Updated the existing-video masked extension path for the post-native-guide H3 architecture.
- Added one-video masked extension and two-video masked bridge examples.
- Added exact audio trimming/assembly for masked extension workflows.
- Removed obsolete pre-native-guide layout/payload monkey-patching where it was no longer needed.
- Added additional masking and workflow regressions.

Typical latent-mask semantics are:

- `0` = preserve known content;
- `1` = denoise/generate new content.

This became the foundation for the current `NEW - Latent Masking` workflow family.

---

## Update 4 — Exact song-latent masking — 2026-08-14

Merged as [PR #6](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef/pull/6).

### Main changes

- Added **H3 Song Audio + Masked Video Context** (`MiniMaxH3SongMaskedAVContext`).
- Added the original master song directly to the target H3 audio latent.
- Protected the song audio latent from denoising.
- Allowed previous visual context to be protected independently while future video remained generative.
- Kept normal Ref2VA image conditioning available without using the song as a `ref_audio_*` reference.
- Added the FL2VA/master-song music-video example and reproducible media assets/tests.
- Used the untouched master song as the authoritative final soundtrack.

Update 4 established the exact master-song latent-masking path that the current music-video workflow builds on.

---

## Update 5 — Persistent checkpoints, multi-clip workflows, and RAM-safe long-form assembly — 2026-08-15

### Main changes

- Added persistent per-clip H3 AV checkpoints with generation signatures, so unchanged clips can be reused from disk instead of being sampled again on every queue.
- Added crash/restart resume at clip boundaries for both music-video and AV-extension chains.
- Reworked long-form continuation so later clips and preview branches load saved checkpoint paths instead of depending on large cached latent tensors from earlier samplers.
- Added RAM-safe sequential final assembly that decodes one saved clip at a time, preserves **linear visual overlap blending at every seam**, and streams the result to ffmpeg.
- Added per-clip video previews and direct in-node preview of the already-written final MP4 without decoding the complete movie back into one giant `IMAGE` batch.
- Added the current multi-clip AV Extension workflow, which can either extend an existing video or generate a new T2V/I2V starter clip and then continue extending it.
- Expanded the current master-song music-video workflow to a 20-slot checkpoint/resume architecture and shipped it preconfigured with the bundled six-clip reproducible demo.
- Fixed the master-song 40 Hz audio-grid boundary case where an exact picture-duration encode can be one latent token short.
- Reworked native H3 compatibility detection for Issue #7 so compatibility is checked behaviorally instead of by brittle source-string inspection.
- Consolidated and renamed example workflows, removed obsolete validation/demo duplicates, and moved workflow guidance into one `example_workflows/README.md` plus notes embedded directly in the workflows.

### 1. Persistent checkpoint gates and selective regeneration

The long-form workflows now use **persistent checkpoint gates** at clip boundaries.

Each completed H3 video/audio latent is saved to a fixed safetensors checkpoint together with a SHA-256 generation signature derived from the submitted graph that produced that clip. Before requesting the sampler again, the gate compares the current generation signature with the saved checkpoint:

- unchanged seed/prompt/upstream inputs → reuse the saved checkpoint without sampling;
- changed Clip N inputs → regenerate Clip N;
- later clips that depend on Clip N regenerate naturally;
- earlier unchanged clips remain reusable.

This behavior does not depend on ComfyUI retaining large latent tensors in RAM and continues to work after cache eviction or process restart as long as the matching checkpoint remains on disk.

The disk-backed path nodes also separate the small cached checkpoint-path string from the large H3 AV latent. Continuation and preview branches reload the saved latent from that path instead of forcing earlier samplers back into execution.

### 2. Crash-safe resume and checkpointed continuation

Update 5 adds checkpoint/resume helpers for long H3 chains, including:

- **H3 Persistent Checkpoint Gate**;
- **H3 Checkpoint Load Path**;
- **H3 Checkpoint Tail Frames**;
- **H3 Resume / Saved AV Latent**;
- **H3 Resume / Live Tail Frames**;
- **H3 Resume / Live AV Latent**;
- **H3 Checkpoint Final Trigger**.

The music-video workflow can resume from completed clip checkpoints, and the AV Extension workflow can resume from either an existing generated extension or a saved generated starter clip.

### 3. RAM-safe assembly and previews

Added streamed assemblers for the long-form workflows:

- **H3 Assemble Checkpoints**;
- **H3 Assemble Extension Checkpoints**;
- **H3 Assemble Starter + Extension Checkpoints**.

Instead of repeatedly concatenating decoded `IMAGE` tensors into an ever-growing movie, the assemblers:

1. load one saved H3 checkpoint at a time;
2. decode only that clip;
3. retain only the small seam tail required for the next join;
4. apply the same linear visual overlap blend at every seam;
5. stream completed video/audio to ffmpeg.

This removes the cumulative decoded-image pattern that caused extremely large host-RAM usage in long chains.

The current long-form workflows also include clip-local VHS previews. The complete assembled MP4 is previewed directly from the encoded file through ComfyUI's native video-preview UI, so a full-movie `IMAGE` tensor is not recreated just for viewing.

### 4. Current music-video workflow

The single current music-video example is:

**`NEW - Latent Masking - Music Video - Lip-Sync + Reference images.json`**

It provides:

- up to 20 clip/sampler slots;
- persistent checkpoint reuse and clip-boundary resume;
- exact master-song audio latent masking;
- optional reference images through the normal H3 reference-conditioning path;
- clip-local previews;
- streamed final assembly with linear visual blending at every seam;
- the untouched master song as the authoritative final soundtrack.

The shipped default reproduces the bundled six-clip demo using the included reference images, `I'll Know You by the Scar.wav`, the original authored Clip 1–6 prompts, and fixed versions of the original numerical seeds. Clips 7–20 remain generic template slots and are bypassed by default.

The full Director Prompt and exact audible-word alignment instructions are embedded in the large note at the top of the workflow itself. The director instructions require alignment to the **actual audible words in each calculated song slice**, rather than estimating lyric timing from transcript order or line count.

### 5. Current multi-clip AV Extension workflow

The recommended multi-clip extension example is:

**`NEW - Latent Masking - AV Extension - Multiple Clips + Reference Images.json`**

A single visible **H3 Extension Start Mode** choice controls the start source:

- **Start from existing video** — load a source video/audio clip and extend it;
- **start with T2V/I2V** — generate a new H3 starter first, then extend that generated clip.

For a generated starter:

- starter first frame off = T2V;
- starter first frame on + Load Image = I2V;
- the starter uses the FL2VA generation path;
- the starter is checkpointed before Extension 1;
- Extension 1 can resume directly from that saved starter checkpoint.

The same start-mode control also synchronizes the source-video branch so the unused VHS source loader is not submitted/validated when starting from T2V/I2V.

For both start modes:

- optional global reference images can guide the generated clips;
- up to six extension slots are available;
- every generated clip is checkpointed persistently;
- later extensions continue from the previous saved H3 AV latent tail;
- unchanged earlier clips can be reused from disk when regenerating a later clip;
- clip-local previews are available;
- final assembly is streamed checkpoint-by-checkpoint with linear visual blending at every seam.

### 6. Master-song 40 Hz audio-grid boundary fix

Fixed the boundary case where the H3 target audio grid can require one more 40 Hz latent step than an audio-VAE encode of the exact picture-duration waveform initially produces—for example, a target of 207 latent steps versus an initial 206-step encode.

The target H3 audio grid is now authoritative. The node uses enough **real master-song waveform** to cover the required grid and then crops the encoded latent to the exact target length. It does not solve the shortfall with fake silence, repeated latent tokens, or guessed padding.

### 7. Motion Context compatibility fix for Issue #7

The native H3 guide/Ref2VA compatibility check no longer depends on finding an exact Python source-code pattern inside `MiniMaxH3.extra_conds()`.

Update 5 probes the behavior of the active conditioning implementation instead:

- Simple Motion Context without references does not require Ref2VA merge behavior;
- image references require the required video-reference merge behavior;
- audio references additionally require the audio-reference merge behavior.

This avoids false failures after compatible ComfyUI refactors and improves coexistence with other H3 nodepacks that wrap or patch core H3 conditioning behavior, while still rejecting wrappers that actually break the required merge semantics.

### 8. Workflow and documentation cleanup

The example folder was simplified around clear architecture labels:

- **`NEW - Latent Masking - ...`** — current/recommended latent-masking workflows;
- **`OLD - Motion Context - ...`** — legacy guide-conditioning examples;
- **`OLD - Hybrid - ...`** — mixed architectures retained for existing projects/experimentation;
- **`UTILITY - ...`** — helper/example graphs.

`OLD` means legacy architecture, not broken or unsupported.

Obsolete validation workflows, duplicate music-video examples, and per-workflow sidecar READMEs were removed. All workflow guidance now lives in **`example_workflows/README.md`** and in notes embedded directly inside the relevant workflow JSON files.

For low-level H3 timing, masking, checkpoint/signature behavior, compatibility probing, and streamed assembly details, see [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md).
