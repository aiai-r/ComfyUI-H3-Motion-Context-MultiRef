"""MiniMax H3 existing-video continuation via preserved AV target-prefix masks.

The node uses native ComfyUI H3 AV-mask support when available. On older
ComfyUI builds, Update 2 installs only the missing runtime compatibility pieces
needed from ComfyUI PR #15375. No ComfyUI source files are modified on disk.
"""

import logging

import torch

import comfy.nested_tensor
import comfy.utils

from .h3_compat import ensure_existing_video_compat
from .h3_timing import largest_h3_video_run

try:
    import torchaudio
except ImportError:
    torchaudio = None


_LOG = logging.getLogger("h3_motion_context.masked_extension")


def _require_h3_mask_support():
    # Lazy, capability-aware compatibility. If current ComfyUI already has
    # equivalent native support, this is a no-op.
    ensure_existing_video_compat()

    # PR #15375 adds H3-specific overrides for both operations. Checking the
    # subclass dictionary avoids mistaking a future/base generic mask hook for
    # the AV-aware implementation this node requires.
    import comfy.model_base

    cls = getattr(comfy.model_base, "MiniMaxH3", None)
    if (
        cls is None
        or "process_denoise_mask" not in cls.__dict__
        or "scale_latent_inpaint" not in cls.__dict__
    ):
        raise RuntimeError(
            "h3_masked_extension: H3 AV mask compatibility could not be enabled. "
            "Check the ComfyUI console capability report."
        )


FPS = 24.0
AUDIO_HZ = 40.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _pixel_frames(latent_t):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


def _streams_from_latent(latent):
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3_masked_extension: expected a MiniMax H3 AV latent, got %r"
            % type(samples)
        )
    if len(parts) < 2:
        raise ValueError(
            "h3_masked_extension: expected joint H3 video+audio latent streams"
        )
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "h3_masked_extension: video latent must be [B,C,T,H,W], got %s"
            % (tuple(video.shape),)
        )
    if audio.ndim != 4:
        raise ValueError(
            "h3_masked_extension: audio latent must be [B,C,2,T], got %s"
            % (tuple(audio.shape),)
        )
    return video, audio


def _resize_images(images, width, height, crop, chunk=32):
    """Resize a video batch without asking common_upscale to hold it all at once."""
    if int(images.shape[0]) <= chunk:
        x = images[..., :3].movedim(-1, 1)
        x = comfy.utils.common_upscale(x, width, height, "lanczos", crop)
        return x.movedim(1, -1)
    out = []
    for start in range(0, int(images.shape[0]), chunk):
        part = images[start : start + chunk, ..., :3].movedim(-1, 1)
        part = comfy.utils.common_upscale(part, width, height, "lanczos", crop)
        out.append(part.movedim(1, -1))
    return torch.cat(out, dim=0)


def _cfr_index_map(frame_count, source_fps, device, target_fps=FPS):
    """Map a CFR source onto target-fps frame centers without touching pixels."""
    source_fps = float(source_fps)
    if source_fps <= 0.0:
        raise ValueError("h3_masked_extension: source_fps must be > 0")
    n = int(frame_count)
    if n < 1:
        raise ValueError("h3_masked_extension: source video has no frames")
    out_n = max(1, int(round(n * float(target_fps) / source_fps)))
    if out_n == n and abs(source_fps - target_fps) < 1e-6:
        return torch.arange(n, device=device, dtype=torch.long)
    i = torch.arange(out_n, device=device, dtype=torch.float64)
    t = (i + 0.5) / float(target_fps)
    src = torch.round(t * source_fps - 0.5).to(torch.long)
    return src.clamp_(0, n - 1)


def _resample_frames_cfr(images, source_fps, target_fps=FPS):
    """Deterministic nearest-frame CFR conversion using frame-center timestamps."""
    if getattr(images, "ndim", 0) != 4 or int(images.shape[0]) < 1:
        raise ValueError(
            "h3_masked_extension: source_frames must be IMAGE [N,H,W,C]"
        )
    idx = _cfr_index_map(int(images.shape[0]), source_fps, images.device, target_fps)
    return images.index_select(0, idx)


def _stereo_first_batch(waveform, label):
    if getattr(waveform, "ndim", 0) != 3:
        raise ValueError(
            "h3_masked_extension: %s waveform must be [B,C,L], got %s"
            % (label, tuple(getattr(waveform, "shape", ())))
        )
    waveform = waveform[:1]
    channels = int(waveform.shape[1])
    if channels == 1:
        return waveform.repeat(1, 2, 1)
    if channels == 2:
        return waveform
    raise ValueError(
        "h3_masked_extension: %s has %d channels. Downmix multichannel audio "
        "to stereo before this node; silently taking two channels can destroy "
        "the source mix." % (label, channels)
    )


def _resample_waveform(waveform, source_sr, target_sr, label):
    source_sr = int(source_sr)
    target_sr = int(target_sr)
    if source_sr == target_sr:
        return waveform
    if torchaudio is None:
        raise RuntimeError(
            "h3_masked_extension: %s is %d Hz but %d Hz is required and "
            "torchaudio is unavailable" % (label, source_sr, target_sr)
        )
    return torchaudio.functional.resample(waveform, source_sr, target_sr)


def _fit_waveform(waveform, want, label, pad=True):
    want = int(want)
    have = int(waveform.shape[-1])
    if have == want:
        return waveform
    if have > want:
        _LOG.info(
            "h3_masked_extension: trimming %s by %d samples to exact timeline",
            label,
            have - want,
        )
        return waveform[..., :want]
    if not pad:
        raise ValueError(
            "h3_masked_extension: %s is %d samples short of the video timeline"
            % (label, want - have)
        )
    _LOG.warning(
        "h3_masked_extension: %s is %d samples short; padding silence at tail",
        label,
        want - have,
    )
    return torch.nn.functional.pad(waveform, (0, want - have))


def _canonical_audio(audio, target_sr, frame_count):
    if audio is None:
        raise ValueError(
            "h3_masked_extension: existing-video AV extension requires source_audio"
        )
    waveform = _stereo_first_batch(audio["waveform"], "source_audio")
    waveform = _resample_waveform(
        waveform, int(audio["sample_rate"]), int(target_sr), "source_audio"
    )
    want = int(round(int(frame_count) / FPS * int(target_sr)))
    waveform = _fit_waveform(waveform, want, "source audio")
    return {"waveform": waveform, "sample_rate": int(target_sr)}


def _snap_context_length(requested, available, target_frames):
    # A masked target prefix must end on a video-VAE run boundary because it is
    # written directly into the target H3 video latent. Unlike classic Motion
    # Context conditioning, there is no safe still-anchor remainder channel here.
    cap = min(int(requested), int(available), int(target_frames) - 1)
    run = largest_h3_video_run(cap)
    if run < 5:
        raise ValueError(
            "h3_masked_extension: need at least 5 source frames and a target "
            "longer than the preserved prefix"
        )
    if run != int(requested):
        _LOG.warning(
            "h3_masked_extension: context_length %d -> exact H3 prefix %d "
            "(valid runs are 5, 22, 39, 56, ...; 39/90/141/... align AV clocks)",
            int(requested), run)
    return run


class MiniMaxH3ExistingVideoMaskedContext:
    """Prepare an arbitrary decoded video as a clean H3 AV target prefix."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Target AV latent from MiniMax H3 Image to Video."
                }),
                "vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE."
                }),
                "audio_vae": ("VAE", {
                    "tooltip": "MiniMax H3 audio VAE."
                }),
                "source_frames": ("IMAGE", {
                    "tooltip": "Decoded source video frames. The node converts CFR timing to H3's 24 fps."
                }),
                "source_audio": ("AUDIO", {
                    "tooltip": "Audio from the same source video. Mono is duplicated to stereo; multichannel must be downmixed first."
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Source-frame FPS. Use CFR or a decoder-normalized frame stream; the example forces 24 fps."
                }),
                "context_length": ("INT", {
                    "default": 39, "min": 5, "max": 9999,
                    "tooltip": "Masked prefixes use H3 runs 5/22/39/56/...; other values snap down. 39/90/141/... align AV exactly."
                }),
                "crop": (["disabled", "center"], {"default": "disabled"}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "trim_frames")
    FUNCTION = "prepare"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Prepare an existing decoded video as an exact preserved H3 AV prefix. "
        "The source tail is normalized to H3 timing, VAE-encoded into the target "
        "latent, and protected with a per-stream denoise mask so H3 generates "
        "only the future portion."
    )

    def prepare(
        self,
        latent,
        vae,
        audio_vae,
        source_frames,
        source_audio,
        source_fps,
        context_length=39,
        crop="disabled",
    ):
        _require_h3_mask_support()
        target_video, target_audio = _streams_from_latent(latent)
        if int(target_video.shape[0]) != 1 or int(target_audio.shape[0]) != 1:
            raise ValueError(
                "h3_masked_extension: existing-video extension currently supports H3 batch size 1"
            )

        target_frames = _pixel_frames(int(target_video.shape[2]))
        expected_target_audio_steps = int(round(target_frames / FPS * AUDIO_HZ))
        if int(target_audio.shape[-1]) != expected_target_audio_steps:
            raise RuntimeError(
                "h3_masked_extension: target latent has %d audio steps for %d "
                "video frames; expected %d on H3's 40 Hz audio grid"
                % (int(target_audio.shape[-1]), target_frames, expected_target_audio_steps)
            )
        width = int(target_video.shape[4]) * 16
        height = int(target_video.shape[3]) * 16

        # Build the canonical 24-fps index map, but only materialize the tail
        # here.  The full source is normalized later by the assembly node, which
        # uses this exact same mapping; this avoids keeping a second full-size
        # resized source video alive during H3 sampling.
        if getattr(source_frames, "ndim", 0) != 4 or int(source_frames.shape[0]) < 1:
            raise ValueError(
                "h3_masked_extension: source_frames must be IMAGE [N,H,W,C]"
            )
        idx = _cfr_index_map(
            int(source_frames.shape[0]), float(source_fps), source_frames.device, FPS
        )
        available = int(idx.numel())
        n = _snap_context_length(context_length, available, target_frames)
        tail_idx = idx[-n:]
        video_tail = source_frames.index_select(0, tail_idx)
        video_tail = _resize_images(video_tail, width, height, crop)

        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        canonical_audio = _canonical_audio(source_audio, vae_sr, available)

        # Video prefix: encode the exact last n canonical frames.
        video_prefix = vae.encode(video_tail)
        if getattr(video_prefix, "ndim", 0) != 5:
            raise ValueError(
                "h3_masked_extension: video VAE returned %s, expected [B,C,T,H,W]"
                % (tuple(getattr(video_prefix, "shape", ())),)
            )
        video_steps = int(video_prefix.shape[2])
        covered = _pixel_frames(video_steps)
        if covered != n:
            raise RuntimeError(
                "h3_masked_extension: %d context frames encoded to %d video "
                "latent steps covering %d frames; refusing a phase-shifted seam"
                % (n, video_steps, covered)
            )
        if video_steps >= int(target_video.shape[2]):
            raise ValueError(
                "h3_masked_extension: video context consumes the whole target latent"
            )

        # Audio prefix: exact same physical interval, end-aligned to source end.
        context_samples = int(round(n / FPS * vae_sr))
        waveform = canonical_audio["waveform"]
        if int(waveform.shape[-1]) < context_samples:
            raise ValueError(
                "h3_masked_extension: source audio is shorter than the selected context"
            )
        audio_tail = waveform[..., -context_samples:]
        audio_prefix = audio_vae.encode(audio_tail.movedim(1, -1))
        if getattr(audio_prefix, "ndim", 0) != 4:
            raise ValueError(
                "h3_masked_extension: audio VAE returned %s, expected [B,C,2,T]"
                % (tuple(getattr(audio_prefix, "shape", ())),)
            )
        exact_audio_steps = n / FPS * AUDIO_HZ
        expected_audio_steps = int(round(exact_audio_steps))
        if abs(exact_audio_steps - expected_audio_steps) > 1e-9:
            _LOG.warning(
                "h3_masked_extension: %d-frame context ends between H3 audio "
                "latent ticks (%.6f steps -> %d). 39 frames is recommended "
                "for an exact AV boundary.",
                n,
                exact_audio_steps,
                expected_audio_steps,
            )
        got_audio_steps = int(audio_prefix.shape[-1])
        if got_audio_steps < expected_audio_steps:
            raise RuntimeError(
                "h3_masked_extension: %d-frame context needs %d audio latent "
                "steps but the audio VAE produced only %d"
                % (n, expected_audio_steps, got_audio_steps)
            )
        if got_audio_steps > expected_audio_steps:
            _LOG.warning(
                "h3_masked_extension: audio VAE produced %d steps for a %d-step "
                "context; end-aligning and keeping the final %d",
                got_audio_steps,
                expected_audio_steps,
                expected_audio_steps,
            )
            audio_prefix = audio_prefix[..., -expected_audio_steps:]
        audio_steps = expected_audio_steps
        if audio_steps >= int(target_audio.shape[-1]):
            raise ValueError(
                "h3_masked_extension: audio context consumes the whole target latent"
            )

        # Fill the clean prefix of the actual target streams.
        out_video = target_video.clone()
        out_audio = target_audio.clone()
        vp = video_prefix[:1].to(device=out_video.device, dtype=out_video.dtype)
        ap = audio_prefix[:1].to(device=out_audio.device, dtype=out_audio.dtype)
        if tuple(vp.shape[1:2] + vp.shape[3:]) != tuple(
            out_video.shape[1:2] + out_video.shape[3:]
        ):
            raise ValueError(
                "h3_masked_extension: encoded video prefix shape %s does not match target %s"
                % (tuple(vp.shape), tuple(out_video.shape))
            )
        if tuple(ap.shape[1:3]) != tuple(out_audio.shape[1:3]):
            raise ValueError(
                "h3_masked_extension: encoded audio prefix shape %s does not match target %s"
                % (tuple(ap.shape), tuple(out_audio.shape))
            )
        out_video[:, :, :video_steps] = vp
        out_audio[..., :audio_steps] = ap

        # ComfyUI PR #15375 unbinds this nested mask, expands each stream to the
        # corresponding latent shape, then H3 turns the two masks into per-row
        # video/audio timesteps.  0 = preserve, 1 = generate.
        video_mask = torch.ones(
            (1, 1, int(out_video.shape[2]), int(out_video.shape[3]), int(out_video.shape[4])),
            device=out_video.device,
            dtype=torch.float32,
        )
        audio_mask = torch.ones(
            (1, 1, int(out_audio.shape[2]), int(out_audio.shape[3])),
            device=out_audio.device,
            dtype=torch.float32,
        )
        video_mask[:, :, :video_steps] = 0.0
        audio_mask[..., :audio_steps] = 0.0

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

        _LOG.info(
            "h3_masked_extension: source %.6g fps -> %d canonical 24fps frames at "
            "%dx%d; preserved prefix %d frames = %d video steps / %d audio "
            "steps (%.3fs); target %d frames",
            float(source_fps),
            available,
            width,
            height,
            n,
            video_steps,
            audio_steps,
            n / FPS,
            target_frames,
        )
        return (out, n)


class MiniMaxH3GeneratedAVMaskedContext:
    """Continue from a previous generated H3 clip directly in latent space.

    Unlike ExistingVideoMaskedContext, this node does not decode and re-encode
    the previous clip. It copies the previous clip's final valid H3 video/audio
    latent run into the new target's prefix and protects both streams with
    per-token mask=0. This is the preferred way to chain generated H3 clips.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Fresh target AV latent from MiniMax H3 Image/Reference to Video."
                }),
                "source_latent": ("LATENT", {
                    "tooltip": "Previous generated H3 sampler/checkpoint latent. Its final AV run is copied directly into the new target prefix."
                }),
                "context_length": ("INT", {
                    "default": 39, "min": 5, "max": 9999,
                    "tooltip": "Protected AV prefix. Uses H3-valid runs 5/22/39/56/...; 39 recommended because it aligns the 24 fps video and 40 Hz audio clocks exactly."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "trim_frames")
    FUNCTION = "prepare"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Chain generated H3 clips without a decode/re-encode round trip. The "
        "previous clip's final joint video/audio latent run is copied into the "
        "new target prefix and protected with AV noise-mask 0; only the future "
        "region denoises."
    )

    def prepare(self, latent, source_latent, context_length=39):
        _require_h3_mask_support()
        target_video, target_audio = _streams_from_latent(latent)
        source_video, source_audio = _streams_from_latent(source_latent)

        if int(target_video.shape[0]) != 1 or int(target_audio.shape[0]) != 1:
            raise ValueError(
                "h3_masked_extension: generated-latent continuation currently supports target batch size 1"
            )
        if int(source_video.shape[0]) != 1 or int(source_audio.shape[0]) != 1:
            raise ValueError(
                "h3_masked_extension: generated-latent continuation currently supports source batch size 1"
            )

        target_frames = _pixel_frames(int(target_video.shape[2]))
        source_frames = _pixel_frames(int(source_video.shape[2]))
        n = _snap_context_length(context_length, source_frames, target_frames)

        # H3-valid pixel runs 5/22/39/... map to temporal latent runs
        # 2/7/12/... . Both full H3 clips and valid context runs are 2 mod 5
        # latent steps, so the source tail starts at the same temporal phase as
        # the target prefix. Direct slicing is therefore phase-aligned.
        video_steps = 2 + 5 * ((n - 5) // 17)
        if _pixel_frames(video_steps) != n:
            raise RuntimeError(
                "h3_masked_extension: internal H3 context mapping failed for %d frames" % n
            )
        audio_steps = int(round(n / FPS * AUDIO_HZ))

        if video_steps >= int(target_video.shape[2]):
            raise ValueError(
                "h3_masked_extension: video context consumes the whole target latent"
            )
        if audio_steps >= int(target_audio.shape[-1]):
            raise ValueError(
                "h3_masked_extension: audio context consumes the whole target latent"
            )
        if int(source_video.shape[2]) < video_steps:
            raise ValueError(
                "h3_masked_extension: source latent has too few video steps for the requested context"
            )
        if int(source_audio.shape[-1]) < audio_steps:
            raise ValueError(
                "h3_masked_extension: source latent has too few audio steps for the requested context"
            )

        if tuple(source_video.shape[1:2] + source_video.shape[3:]) != tuple(
            target_video.shape[1:2] + target_video.shape[3:]
        ):
            raise ValueError(
                "h3_masked_extension: source/target video latent geometry differs: %s vs %s. "
                "Keep chained clips at the same H3 resolution."
                % (tuple(source_video.shape), tuple(target_video.shape))
            )
        if tuple(source_audio.shape[1:3]) != tuple(target_audio.shape[1:3]):
            raise ValueError(
                "h3_masked_extension: source/target audio latent geometry differs: %s vs %s"
                % (tuple(source_audio.shape), tuple(target_audio.shape))
            )

        out_video = target_video.clone()
        out_audio = target_audio.clone()
        out_video[:, :, :video_steps] = source_video[:, :, -video_steps:].to(
            device=out_video.device, dtype=out_video.dtype
        )
        out_audio[..., :audio_steps] = source_audio[..., -audio_steps:].to(
            device=out_audio.device, dtype=out_audio.dtype
        )

        video_mask = torch.ones(
            (1, 1, int(out_video.shape[2]), int(out_video.shape[3]), int(out_video.shape[4])),
            device=out_video.device, dtype=torch.float32,
        )
        audio_mask = torch.ones(
            (1, 1, int(out_audio.shape[2]), int(out_audio.shape[3])),
            device=out_audio.device, dtype=torch.float32,
        )
        video_mask[:, :, :video_steps] = 0.0
        audio_mask[..., :audio_steps] = 0.0

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

        _LOG.info(
            "h3_masked_extension: latent->latent AV continuation protected %d frames = %d video steps / %d audio steps; target %d frames",
            n, video_steps, audio_steps, target_frames,
        )
        return (out, n)


class MiniMaxH3StartMaskedContext:
    """Prepare the first extension context from either a real video or a generated starter clip.

    start_mode=load_video uses the uploaded/decoded source video path.
    start_mode=generate_starter uses a sampled H3 starter clip (t2v or i2v) and
    copies its final AV latent tail directly into the new target. When
    resume_from_extension == 1, the live starter sampler branch is skipped and
    the saved starter checkpoint is loaded from disk instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Fresh target AV latent for Extension 1."
                }),
                "vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE. Used only for load_video mode."
                }),
                "audio_vae": ("VAE", {
                    "tooltip": "MiniMax H3 audio VAE. Used only for load_video mode."
                }),
                "start_mode": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect H3 Extension Start Mode. load_video = extend an uploaded source video; generate_starter = first sample a t2v/i2v starter clip, then extend it."
                }),
                "context_length": ("INT", {
                    "default": 39, "min": 5, "max": 9999,
                    "tooltip": "Protected AV prefix. Uses H3-valid runs 5/22/39/56/...; 39 recommended because it aligns the 24 fps video and 40 Hz audio clocks exactly."
                }),
                "resume_from_extension": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "When start_mode = generate_starter, set 1 to resume Extension 1 from the saved starter checkpoint instead of regenerating the starter clip."
                }),
                "starter_checkpoint_path": ("STRING", {
                    "default": "h3_extension_checkpoints/starter",
                    "tooltip": "Checkpoint path/prefix for the generated starter clip. Used only in generate_starter mode."
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Source-frame FPS. Used only in load_video mode."
                }),
                "crop": (["disabled", "center"], {"default": "disabled"}),
            },
            "optional": {
                "source_frames": ("IMAGE", {
                    "lazy": True,
                    "tooltip": "Decoded source-video frames. Requested only in load_video mode."
                }),
                "source_audio": ("AUDIO", {
                    "lazy": True,
                    "tooltip": "Source-video audio. Requested only in load_video mode."
                }),
                "live_starter_latent": ("LATENT", {
                    "lazy": True,
                    "tooltip": "Live sampled starter H3 AV latent. Requested only in generate_starter mode when not resuming Extension 1."
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "trim_frames")
    FUNCTION = "prepare"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Prepare Extension 1 for a multi-clip masked AV chain. Either use the "
        "uploaded source video as the preserved prefix, or use a sampled H3 "
        "starter clip (t2v/i2v) and continue from its final latent tail."
    )

    @classmethod
    def IS_CHANGED(
        cls, start_mode="load_video", resume_from_extension=0,
        starter_checkpoint_path="h3_extension_checkpoints/starter", **kwargs
    ):
        # Live source-video / live-starter operation should use normal ComfyUI
        # dependency caching. Only explicit Extension-1 resume reads a fixed
        # checkpoint slot that may be replaced between queues.
        if str(start_mode) == "generate_starter" and int(resume_from_extension) == 1:
            from .h3_checkpoint_resume import _checkpoint_disk_token
            return _checkpoint_disk_token(starter_checkpoint_path, 1)
        return "live-start:%s" % str(start_mode)

    def check_lazy_status(
        self, latent, vae, audio_vae, start_mode, context_length,
        resume_from_extension, starter_checkpoint_path, source_fps, crop,
        source_frames=None, source_audio=None, live_starter_latent=None,
    ):
        if str(start_mode) == "load_video":
            needed = []
            if source_frames is None:
                needed.append("source_frames")
            if source_audio is None:
                needed.append("source_audio")
            return needed

        if int(resume_from_extension) == 1:
            return []
        if live_starter_latent is None:
            return ["live_starter_latent"]
        return []

    def prepare(
        self, latent, vae, audio_vae, start_mode="load_video", context_length=39,
        resume_from_extension=0, starter_checkpoint_path="h3_extension_checkpoints/starter",
        source_fps=24.0, crop="disabled", source_frames=None, source_audio=None,
        live_starter_latent=None,
    ):
        mode = str(start_mode)
        if mode == "load_video":
            if source_frames is None or source_audio is None:
                missing = []
                if source_frames is None:
                    missing.append("source_frames")
                if source_audio is None:
                    missing.append("source_audio")
                raise ValueError(
                    "h3_masked_extension: load_video mode needs %s" % ", ".join(missing)
                )
            return MiniMaxH3ExistingVideoMaskedContext().prepare(
                latent, vae, audio_vae, source_frames, source_audio, source_fps,
                context_length, crop,
            )

        if int(resume_from_extension) == 1:
            from .h3_checkpoint_resume import _load_checkpoint, _resolve_checkpoint_path
            ckpt = _resolve_checkpoint_path(starter_checkpoint_path, 1)
            video, audio = _load_checkpoint(ckpt)
            starter = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
            _LOG.info(
                "h3_masked_extension: Extension 1 resumes from saved starter checkpoint %s",
                ckpt,
            )
        else:
            if live_starter_latent is None:
                raise ValueError(
                    "h3_masked_extension: generate_starter mode needs live_starter_latent unless resume_from_extension = 1"
                )
            starter = live_starter_latent

        return MiniMaxH3GeneratedAVMaskedContext().prepare(
            latent, starter, context_length
        )


class MiniMaxH3StartCheckpointMaskedContext:
    """Disk-backed Extension-1 start selector.

    load_video behaves like MiniMaxH3StartMaskedContext. generate_starter waits
    only for the starter checkpoint path and reloads that saved joint AV latent,
    so ComfyUI never has to retain the full starter latent just to preserve the
    dependency between queues.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "start_mode": ("STRING", {"forceInput": True}),
                "context_length": ("INT", {"default": 39, "min": 5, "max": 9999}),
                "resume_from_extension": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "starter_checkpoint_path": ("STRING", {"default": "h3_extension_checkpoints/starter"}),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "crop": (["disabled", "center"], {"default": "disabled"}),
            },
            "optional": {
                "source_frames": ("IMAGE", {"lazy": True}),
                "source_audio": ("AUDIO", {"lazy": True}),
                "starter_checkpoint_signal": ("STRING", {
                    "lazy": True, "forceInput": True,
                    "tooltip": "Exact path from the generated starter H3 Checkpoint Save Path node."
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "trim_frames")
    FUNCTION = "prepare"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Prepare Extension 1 from source video or a disk-backed generated starter checkpoint."
    )

    @classmethod
    def IS_CHANGED(cls, start_mode="load_video", resume_from_extension=0, starter_checkpoint_path="h3_extension_checkpoints/starter", **kwargs):
        if str(start_mode) == "generate_starter" and int(resume_from_extension) == 1:
            from .h3_checkpoint_resume import _checkpoint_disk_token
            return _checkpoint_disk_token(starter_checkpoint_path, 1)
        return "disk-start:%s" % str(start_mode)

    def check_lazy_status(
        self, latent, vae, audio_vae, start_mode, context_length,
        resume_from_extension, starter_checkpoint_path, source_fps, crop,
        source_frames=None, source_audio=None, starter_checkpoint_signal=None,
    ):
        if str(start_mode) == "load_video":
            needed = []
            if source_frames is None:
                needed.append("source_frames")
            if source_audio is None:
                needed.append("source_audio")
            return needed
        if int(resume_from_extension) == 1:
            return []
        if starter_checkpoint_signal is None:
            return ["starter_checkpoint_signal"]
        return []

    def prepare(
        self, latent, vae, audio_vae, start_mode="load_video", context_length=39,
        resume_from_extension=0, starter_checkpoint_path="h3_extension_checkpoints/starter",
        source_fps=24.0, crop="disabled", source_frames=None, source_audio=None,
        starter_checkpoint_signal=None,
    ):
        mode = str(start_mode)
        if mode == "load_video":
            if source_frames is None or source_audio is None:
                missing = []
                if source_frames is None:
                    missing.append("source_frames")
                if source_audio is None:
                    missing.append("source_audio")
                raise ValueError("h3_masked_extension: load_video mode needs %s" % ", ".join(missing))
            return MiniMaxH3ExistingVideoMaskedContext().prepare(
                latent, vae, audio_vae, source_frames, source_audio, source_fps, context_length, crop
            )

        from .h3_checkpoint_resume import _load_checkpoint, _resolve_checkpoint_path
        if int(resume_from_extension) == 1:
            ckpt = _resolve_checkpoint_path(starter_checkpoint_path, 1)
            source = "resume starter checkpoint"
        else:
            if not starter_checkpoint_signal:
                raise ValueError("h3_masked_extension: generate_starter mode needs starter_checkpoint_signal")
            ckpt = _resolve_checkpoint_path(str(starter_checkpoint_signal), 1)
            source = "saved starter checkpoint"
        video, audio = _load_checkpoint(ckpt)
        starter = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
        _LOG.info("h3_masked_extension: Extension 1 continues from %s %s", source, ckpt)
        return MiniMaxH3GeneratedAVMaskedContext().prepare(latent, starter, context_length)


class MiniMaxH3AssembleExtension:
    """Frame/sample-exact assembly for an existing-video continuation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_frames": ("IMAGE", {
                    "tooltip": "Decoded source video frames. They are normalized to the same 24-fps mapping used by the context node."
                }),
                "source_audio": ("AUDIO",),
                "source_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001
                }),
                "continuation_images": ("IMAGE",),
                "continuation_audio": ("AUDIO",),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Use 24 for MiniMax H3. Audio is forced to the exact integer-sample duration implied by each image batch before concatenation."
                }),
                "crop": (["disabled", "center"], {
                    "default": "disabled",
                    "tooltip": "Must match the crop setting used by H3 Existing Video Masked Context so the prepended source and conditioning tail use the identical spatial transform."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "assemble"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Concatenate the canonical source and the trimmed H3 continuation. "
        "No crossfade: both audio pieces are resampled/matched to exact frame "
        "durations first so the join cannot accumulate AV drift."
    )

    def assemble(
        self,
        source_frames,
        source_audio,
        source_fps,
        continuation_images,
        continuation_audio,
        fps=24.0,
        crop="disabled",
    ):
        fps = float(fps)
        if fps <= 0:
            raise ValueError("h3_masked_extension: fps must be > 0")
        if getattr(continuation_images, "ndim", 0) != 4 or int(continuation_images.shape[0]) < 1:
            raise ValueError("h3_masked_extension: continuation_images is empty")
        height = int(continuation_images.shape[1])
        width = int(continuation_images.shape[2])
        source_images = _resample_frames_cfr(source_frames, float(source_fps), fps)
        source_images = _resize_images(source_images, width, height, crop)

        cont_sr = int(continuation_audio["sample_rate"])
        src_wave = _stereo_first_batch(source_audio["waveform"], "source_audio")
        src_wave = _resample_waveform(
            src_wave, int(source_audio["sample_rate"]), cont_sr, "source_audio"
        )
        cont_wave = _stereo_first_batch(
            continuation_audio["waveform"], "continuation_audio"
        )
        cont_wave = _resample_waveform(
            cont_wave,
            int(continuation_audio["sample_rate"]),
            cont_sr,
            "continuation_audio",
        )

        src_frames = int(source_images.shape[0])
        cont_frames = int(continuation_images.shape[0])

        src_want = int(round(src_frames / fps * cont_sr))
        total_want = int(round((src_frames + cont_frames) / fps * cont_sr))
        cont_want = total_want - src_want

        src_wave = _fit_waveform(src_wave, src_want, "source audio")
        cont_wave = _fit_waveform(cont_wave, cont_want, "continuation audio")

        images = torch.cat((source_images, continuation_images), dim=0)
        waveform = torch.cat((src_wave, cont_wave), dim=-1)
        audio = {"waveform": waveform, "sample_rate": cont_sr}

        expected = int(round(int(images.shape[0]) / fps * cont_sr))
        if int(waveform.shape[-1]) != expected:
            raise RuntimeError(
                "h3_masked_extension: internal AV accounting failed: got %d audio "
                "samples, expected %d for %d frames"
                % (int(waveform.shape[-1]), expected, int(images.shape[0]))
            )
        _LOG.info(
            "h3_masked_extension: assembled %d + %d = %d frames; %d audio samples "
            "at %d Hz (drift 0 samples by construction)",
            int(source_images.shape[0]),
            int(continuation_images.shape[0]),
            int(images.shape[0]),
            int(waveform.shape[-1]),
            cont_sr,
        )
        return (images, audio)
