# Example workflows

## Recommended context

The updated continuation workflows expose one **GLOBAL CONTEXT FRAMES** control and a separate **GLOBAL VIDEO CROSSFADE** control. Both default to 39.

```text
H3-native video runs: 5, 22, 39*, 56, 73, 90*, 107, 124, 141*, 158, 175, 192*, 209, 226, 243*
* exact video+audio boundary. 39 recommended.
```

Classic Motion Context accepts any frame count. The native values above are especially efficient; starred values also end exactly on H3's 40 Hz audio-latent clock.

The Existing MP4 masked-prefix node is stricter: an off-grid request snaps down to the nearest native full video run because the preserved prefix is written directly into the target H3 latent.

The crossfade control is independent. Update 2's Trim node keeps only the final matching overlap needed by KJNodes, so a longer context (for example 90) can safely use a shorter crossfade (for example 39). Audio always trims the full context.

## Workflows

### Simple Motion Context - No Reference Images

- Global visual + timeline-audio context (39 default).
- KJNodes linear cumulative video stitching; crossfade may be shorter than context safely.
- Trimmed/hard-appended audio; no audio crossfade.

### Advanced Motion Context - Reference Images

- Same global context/crossfade controls.
- Ref2VA/MultiRef character references preserved.
- KJNodes linear cumulative video stitching.
- Trimmed/hard-appended audio.

### Music Video Motion Context - Song Driven Lipsync + Reference Images

- Global **visual-only** Motion Context (39 default).
- KJNodes linear cumulative video stitching.
- Original-song slice/final-song architecture remains; slice start times and durations are calculated automatically from the current H3-valid frame count and visual context length.

### Advanced Extension of Input Videos

- Existing video forced to 24 fps and cropped down to /32.
- Two picture inputs for Ref2VA identity/appearance reference.
- Source MP4 supplies the temporal video/audio history.
- Preserved target AV prefix using native #15375-equivalent behavior or the capability-aware compatibility layer.
- KJNodes linear video overlap.
- Exact hard-joined audio; no audio crossfade.

### Custom Keyframes Example

Unchanged in Update 2.

## Workflow dependencies

The continuation examples expect:

- `ComfyUI-KJNodes`
- `ComfyUI-VideoHelperSuite`

These are workflow dependencies only; the Motion Context Python package does not import them.


### H3 Masked AV Extension - One Video Example - 192f

- `VHS_LoadVideo` source normalized to 864×480 at 24 fps.
- Last 39 AV frames become a protected target-latent prefix; 153 future frames are generated in a 192-frame target.
- `H3 Motion Context Trim` uses the node's actual preserved-frame count and trims picture/audio together, including H3 audio-tail rounding.
- `H3 Assemble Existing Video Extension` produces frame/sample-exact final audio.
- 39-frame KJNodes linear visual overlap remains on the delivered picture.
- Two `VHS_VideoCombine` outputs are included: the raw unstitched 192-frame H3 clip and the final stitched extension.

### H3 Masked AV Bridge - Two Video Example - 192f

- Two input videos normalized to 24 fps.
- 39 AV frames are protected at each target endpoint.
- 114 middle frames are generated.
- 39-frame KJNodes linear visual overlaps are retained at both delivered joins.
