# H3 Masked AV Bridge — Two Videos, 192 Frames

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
