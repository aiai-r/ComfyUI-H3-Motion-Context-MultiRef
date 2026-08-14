# H3 Masked AV Extension — One Video, 192 Frames

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

## VHS outputs

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
