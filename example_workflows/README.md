# Example Workflows

## NEW - Music Video

`NEW - Music Video.json`

Creates a multi-clip music video around one song.

The included workflow comes with a complete example using the files in:

`example_workflows/assets/`

Copy the example images and song into your ComfyUI `input/` folder before running it.

### Main controls

**Active Clips**
Sets how many clip sections are used.

**Previews**
Controls which clip previews are generated.

**Reference Images**
Use the included references or replace them with your own.

**Master Song**
The song used by the workflow. Replace it with your own audio when starting a new project.

The workflow contains up to 20 clip sections. Only the number selected by **Active Clips** is used.

---

## NEW - AV Extension

`NEW - AV Extension.json`

Continues a video across multiple H3 generations.

It can start from either:

- an existing video;
- a new T2V generation;
- a new I2V generation.

### Start mode

Choose whether the workflow begins with an existing video or generates the first clip itself.

### Extensions

Enable as many extension sections as you need and set **Active Extensions** to the same number.

Extensions should be enabled in order.

### References

Optional reference images can be used when needed.

### Previews

Use the preview control to enable or disable extension previews.

---

The folder also contains utility workflows for AV bridging and custom keyframes, plus legacy Motion Context and hybrid workflows retained from earlier versions.

## More information

For implementation details, timing, masking, audio handling, and other internals, see:

[../TECHNICAL_ARCHITECTURE.md](../TECHNICAL_ARCHITECTURE.md)

For the detailed update history, see:

[../MODIFICATIONS.md](../MODIFICATIONS.md)
