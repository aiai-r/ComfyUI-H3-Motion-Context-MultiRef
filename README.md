# ComfyUI H3 Motion Context — MultiRef & Latent Masking

A ComfyUI custom-node pack and workflow collection for extending MiniMax H3 video generations with motion continuity, references, audio, and latent masking.

Update 6 reduces RAM and cache pressure during long-form final output, making out-of-memory errors less likely.

Modified fork of [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context).

## Main workflows

### NEW - Music Video

Creates a sequence of H3 video clips around a single song.

Example Music Video you can recreate:
https://github.com/user-attachments/assets/33e22c59-d23f-4470-b52a-6fabb0e4a66b

A complete example with reference images and a song is included in:

`example_workflows/NEW - Music Video.json`

The example assets are in:

`example_workflows/assets/`

### NEW - AV Extension

Extends an existing video, or starts from a newly generated T2V/I2V clip and continues it through multiple H3 generations.

Open:

`example_workflows/NEW - AV Extension.json`

The repository also includes utility and legacy workflows for clip bridging, custom keyframes, and earlier Motion Context / hybrid continuation methods.

See [example_workflows/README.md](example_workflows/README.md) for the workflow guide.

## Installation

Clone the repository into ComfyUI's `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef.git
```

Restart ComfyUI and refresh the browser.

MiniMax H3 model files are not included.

## Dependencies

Some included workflows use:

- ComfyUI-VideoHelperSuite
- ComfyUI-KJNodes
- rgthree-comfy

If a workflow opens with missing nodes, install the required node pack and restart ComfyUI.

## Documentation

- [Workflow Guide](example_workflows/README.md) — how to use the included workflows
- [Technical Architecture](TECHNICAL_ARCHITECTURE.md) — implementation and technical details
- [Modifications](MODIFICATIONS.md) — detailed history of changes made in this fork

## Credits

Original project by [NikoDemon80](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context).

See [MODIFICATIONS.md](MODIFICATIONS.md) for additional attribution and implementation history.

## License

GPL-3.0. See [LICENSE](LICENSE).
