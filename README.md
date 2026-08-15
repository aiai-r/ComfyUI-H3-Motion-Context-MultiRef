# ComfyUI H3 Motion Context — MultiRef & Latent Masking

A ComfyUI custom-node pack and workflow collection for making **longer MiniMax H3 videos from multiple generations** while keeping motion, identity, audio continuity, or an exact master-song timeline consistent across clip boundaries.

For new projects, start with the workflows whose names begin with **`NEW - Latent Masking`**. The older **`OLD - Motion Context`** workflows are retained for existing projects and experimentation.

> Modified fork of [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context), GPL-3.0.

## Which workflow should I use?

| I want to… | Use this workflow |
|---|---|
| Make a long music video synced to one exact song | **NEW - Latent Masking - Music Video - Lip-Sync + Reference images** |
| Extend an uploaded video through several H3 generations | **NEW - Latent Masking - AV Extension - Multiple Clips + Reference Images** |
| Generate a new T2V or I2V starter clip and then keep extending it | **NEW - Latent Masking - AV Extension - Multiple Clips + Reference Images** |
| Learn the simplest masked existing-video extension | **NEW - Latent Masking - AV Extension - Minimal Single Clip** |
| Generate a transition between two existing videos | **NEW - Latent Masking - AV Bridge - Two Videos** |
| Use the older guide-based continuation method without references | **OLD - Motion Context - Simple - No Reference Images** |
| Use the older guide-based continuation method with Ref2VA images | **OLD - Motion Context - Advanced - Reference Images** |
| Place custom still-image anchors on the H3 timeline | **UTILITY - Custom Keyframes Example** |

See [example_workflows/README.md](example_workflows/README.md) for the single workflow chooser and detailed guide for every included workflow.

## Recommended starting points

### Music video

Use **`NEW - Latent Masking - Music Video - Lip-Sync + Reference images.json`**.

It supports:

- up to 20 sequential H3 clips;
- reference images for performer identity/appearance;
- the original master song as the exact H3 audio timeline;
- per-clip checkpoints and clip-boundary resume;
- temporary per-clip VHS previews with a global preview bypass;
- RAM-safe sequential final assembly with direct full-MP4 preview;
- linear visual blending at every clip seam;
- the untouched original song as the final soundtrack.

### General video extension

Use **`NEW - Latent Masking - AV Extension - Multiple Clips + Reference Images.json`**.

One switch chooses how the chain begins:

- **Load video** — start from an uploaded video and continue it.
- **Generate starter** — first generate a new H3 clip, then continue it. Leave the optional starter image off for **T2V**, or enable it for **I2V**.

Optional Ref2VA images can be used across the extension clips. The workflow opens with **only Extension 1 active**. Use the rgthree **OPTIONAL EXTENSIONS 2–6 — ENABLE SEQUENTIALLY** switch to enable later groups, and set **GLOBAL ACTIVE EXTENSION COUNT** to the highest enabled extension number.

The generated starter and every extension clip are saved as H3 AV latent checkpoints. Later clips continue from the previous latent tail directly, and the final movie is assembled one checkpoint at a time. This avoids the cumulative decoded-image workflow pattern that can consume very large amounts of RAM.

In **Load video** mode, the uploaded source is still decoded by the video loader, so an unusually long/high-resolution source can use RAM proportional to that source file. That source-memory cost does not multiply with the number of generated extensions.

## Basic terminology

**Reference image** — tells H3 who or what something should look like.

**Protected context** — already-existing video/audio at a continuation boundary that should not be regenerated.

**Latent masking** — protects known H3 video/audio latent regions while H3 denoises only the new region.

**Motion Context** — the older continuation method that supplies previous motion/audio as H3 guide conditioning.

**Master song** — the authoritative song timeline used by the music-video workflow.

**Checkpoint** — a saved H3 video+audio latent for a completed clip, used for resume and low-memory assembly.

## Current vs legacy workflows

### Current — Latent Masking

Recommended for new projects. These workflows can preserve known video/audio directly inside H3's target latent and are the main focus of current development.

The long-form workflows also use disk checkpoints and sequential assembly so completed clips do not need to accumulate as one giant decoded `IMAGE` tensor.

### Legacy — Motion Context

Files beginning with **`OLD - Motion Context`** use the earlier guide-based continuation architecture.

`OLD` means **legacy architecture**, not “known broken.” They remain useful for existing projects, comparison, and experimentation.

Technical details are intentionally kept out of this README. See [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) if you want to understand the differences internally.

## Installation

Clone the repository into ComfyUI's `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef.git
```

Then restart ComfyUI and hard-refresh the browser.

A recent ComfyUI installation is recommended. MiniMax H3 model files are not included in this repository.

### Common workflow dependencies

Different example workflows use different third-party nodes. Common ones include:

- **ComfyUI-VideoHelperSuite** — video loading and some exports/previews;
- **ComfyUI-KJNodes** — used by some legacy workflows and optional attention/utility nodes;
- **rgthree-comfy** — used by workflows with group bypass/switch controls.

If a workflow opens with missing nodes, install the node pack named by ComfyUI and restart.

## Updates

The repository currently has four completed public update PRs. This work is **Update 5**.

- **Update 1 — 2026-08-10:** Custom H3 keyframes.
- **Update 2 — 2026-08-12:** Existing-video extension and compatibility/workflow improvements.
- **Update 3 — 2026-08-14:** Per-token H3 video/audio latent masking and AV bridge workflows.
- **Update 4 — 2026-08-14:** Exact FL2VA/master-song latent masking for song-driven generation.
- **Update 5 — 2026-08-15:** Checkpoints/resume, RAM-safe long-form assembly, 20-clip music video, compatibility fixes, refreshed latent-masked extension workflows, T2V/I2V starter mode, and documentation cleanup.

For the detailed history, see **[MODIFICATIONS.md](MODIFICATIONS.md)**.

For implementation details, see **[TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md)**.

## Troubleshooting / reporting an issue

Before reporting a problem:

1. Update ComfyUI.
2. Update this node pack.
3. Restart ComfyUI completely.
4. Confirm the workflow's required models and third-party nodes are installed.
5. Try the smallest relevant example workflow if possible.

When opening an issue, please include:

- the full traceback;
- your ComfyUI commit (`git rev-parse HEAD` from the ComfyUI folder);
- `git status --short` from the ComfyUI folder;
- how you updated ComfyUI;
- your ComfyUI startup log or custom-node list;
- the workflow that triggered the error.

## Technical documentation

For H3 timing, video/audio latent structure, guide conditioning, latent masking, master-song timing, checkpoint/resume behavior, compatibility layers, and streamed seam assembly, read:

**[TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md)**

Focused historical/reference documents are also retained:


## Credits

The original Motion Context project is by [NikoDemon80](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context).

The H3 per-token video/audio latent masking work in this fork builds on the design introduced by **Barish Ozbay (`drozbay`)** in ComfyUI PR **#15375**.

See [MODIFICATIONS.md](MODIFICATIONS.md) for the detailed fork/update history and attribution.

## License

GPL-3.0. See [LICENSE](LICENSE).
