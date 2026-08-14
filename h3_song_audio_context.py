"""MiniMax H3 FL clip prep with exact master-song audio and optional video prefix."""

import logging

import torch

import comfy.nested_tensor
import comfy.utils

try:
    from .h3_compat import ensure_existing_video_compat
except ImportError:
    ensure_existing_video_compat = lambda: True

try:
    import torchaudio
except ImportError:
    torchaudio = None

_LOG = logging.getLogger("h3_motion_context.song_audio")

FPS = 24.0
AUDIO_HZ = 40.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _require_h3_mask_support():
    ensure_existing_video_compat()
    import comfy.model_base

    cls = getattr(comfy.model_base, "MiniMaxH3", None)
    if (
        cls is None
        or "process_denoise_mask" not in cls.__dict__
        or "scale_latent_inpaint" not in cls.__dict__
    ):
        raise RuntimeError(
            "h3_song_audio: H3 AV mask compatibility could not be enabled. Check the ComfyUI console capability report."
        )


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
            "h3_song_audio: expected a MiniMax H3 AV latent, got %r" % type(samples)
        )
    if len(parts) < 2:
        raise ValueError("h3_song_audio: expected joint H3 video+audio latent streams")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "h3_song_audio: video latent must be [B,C,T,H,W], got %s"
            % (tuple(video.shape),)
        )
    if audio.ndim != 4:
        raise ValueError(
            "h3_song_audio: audio latent must be [B,C,2,T], got %s"
            % (tuple(audio.shape),)
        )
    return video, audio


def _resize_images(images, width, height, crop, chunk=32):
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
    source_fps = float(source_fps)
    if source_fps <= 0.0:
        raise ValueError("h3_song_audio: source_fps must be > 0")
    n = int(frame_count)
    if n < 1:
        raise ValueError("h3_song_audio: source video has no frames")
    out_n = max(1, int(round(n * float(target_fps) / source_fps)))
    if out_n == n and abs(source_fps - target_fps) < 1e-6:
        return torch.arange(n, device=device, dtype=torch.long)
    i = torch.arange(out_n, device=device, dtype=torch.float64)
    t = (i + 0.5) / float(target_fps)
    src = torch.round(t * source_fps - 0.5).to(torch.long)
    return src.clamp_(0, n - 1)


def _stereo_first_batch(waveform, label):
    if getattr(waveform, "ndim", 0) != 3:
        raise ValueError(
            "h3_song_audio: %s waveform must be [B,C,L], got %s"
            % (label, tuple(getattr(waveform, "shape", ())))
        )
    waveform = waveform[:1]
    channels = int(waveform.shape[1])
    if channels == 1:
        return waveform.repeat(1, 2, 1)
    if channels == 2:
        return waveform
    raise ValueError(
        "h3_song_audio: %s has %d channels. Downmix multichannel audio to stereo before this node."
        % (label, channels)
    )


def _resample_waveform(waveform, source_sr, target_sr, label):
    source_sr = int(source_sr)
    target_sr = int(target_sr)
    if source_sr == target_sr:
        return waveform
    if torchaudio is None:
        raise RuntimeError(
            "h3_song_audio: %s is %d Hz but %d Hz is required and torchaudio is unavailable"
            % (label, source_sr, target_sr)
        )
    return torchaudio.functional.resample(waveform, source_sr, target_sr)


def _fit_waveform(waveform, want, label, pad=True):
    want = int(want)
    have = int(waveform.shape[-1])
    if have == want:
        return waveform
    if have > want:
        _LOG.info(
            "h3_song_audio: trimming %s by %d samples to exact timeline",
            label,
            have - want,
        )
        return waveform[..., :want]
    if not pad:
        raise ValueError(
            "h3_song_audio: %s is %d samples short of the video timeline" % (label, want - have)
        )
    _LOG.warning(
        "h3_song_audio: %s is %d samples short; padding silence at tail",
        label,
        want - have,
    )
    return torch.nn.functional.pad(waveform, (0, want - have))


def _largest_h3_video_run(n):
    n = int(n)
    if n < 5:
        return 0
    return ((n - 5) // 17) * 17 + 5


def _snap_context_length(requested, available, target_frames):
    cap = min(int(requested), int(available), int(target_frames) - 1)
    run = _largest_h3_video_run(cap)
    if run < 5:
        raise ValueError(
            "h3_song_audio: need at least 5 source frames and a target longer than the preserved prefix"
        )
    if run != int(requested):
        _LOG.warning(
            "h3_song_audio: context_length %d -> exact H3 prefix %d (valid runs are 5, 22, 39, 56, ...; 39/90/141/... align AV clocks)",
            int(requested), run,
        )
    return run


class MiniMaxH3SongMaskedAVContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Target AV latent from MiniMax H3 Image to Video (FL path)."
                }),
                "audio_vae": ("VAE", {
                    "tooltip": "MiniMax H3 audio VAE. Used to encode the master-song slice into the target audio latent."
                }),
                "master_audio": ("AUDIO", {
                    "tooltip": "Full master song or master vocal audio. The node writes the exact requested slice into the H3 latent and protects it from denoising."
                }),
                "clip_start_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 99999.0, "step": 0.001,
                    "tooltip": "Start time of this raw clip within the master song timeline. For clip N, this usually advances by raw_clip_seconds - context_seconds from the previous clip."
                }),
                "context_length": ("INT", {
                    "default": 39, "min": 0, "max": 9999,
                    "tooltip": "Optional preserved visual prefix from the previous clip. Uses exact H3 runs 5/22/39/56/...; 0 disables visual prefixing entirely."
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "FPS of source_frames when visual context is connected."
                }),
                "crop": (["disabled", "center"], {"default": "disabled"}),
            },
            "optional": {
                "vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE. Required only when source_frames is connected."
                }),
                "source_frames": ("IMAGE", {
                    "tooltip": "Optional previous clip frames. The node preserves the final context_length canonical 24fps frames as a protected visual prefix."
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "AUDIO")
    RETURN_NAMES = ("latent", "trim_frames", "clip_audio")
    FUNCTION = "prepare"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Write an exact master-song slice into the target H3 audio latent and protect the entire audio stream from denoising. Optionally also preserve a previous-clip visual prefix."
    )

    def prepare(
        self,
        latent,
        audio_vae,
        master_audio,
        clip_start_seconds=0.0,
        context_length=39,
        source_fps=24.0,
        crop="disabled",
        vae=None,
        source_frames=None,
    ):
        _require_h3_mask_support()

        target_video, target_audio = _streams_from_latent(latent)
        if int(target_video.shape[0]) != 1 or int(target_audio.shape[0]) != 1:
            raise ValueError("h3_song_audio: batch size 1 only")

        target_frames = _pixel_frames(int(target_video.shape[2]))
        expected_audio_steps = int(round(target_frames / FPS * AUDIO_HZ))
        if int(target_audio.shape[-1]) != expected_audio_steps:
            raise RuntimeError(
                "h3_song_audio: target latent has %d audio steps for %d video frames; expected %d on H3's 40 Hz audio grid"
                % (int(target_audio.shape[-1]), target_frames, expected_audio_steps)
            )

        width = int(target_video.shape[4]) * 16
        height = int(target_video.shape[3]) * 16

        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        waveform = _stereo_first_batch(master_audio["waveform"], "master_audio")
        waveform = _resample_waveform(waveform, int(master_audio["sample_rate"]), vae_sr, "master_audio")
        start_sample = int(round(float(clip_start_seconds) * vae_sr))
        if start_sample < 0:
            raise ValueError("h3_song_audio: clip_start_seconds must be >= 0")
        want_samples = int(round(target_frames / FPS * vae_sr))
        end_sample = start_sample + want_samples
        audio_slice = waveform[..., start_sample:end_sample]
        audio_slice = _fit_waveform(audio_slice, want_samples, "master audio slice")

        audio_full = audio_vae.encode(audio_slice.movedim(1, -1))
        if getattr(audio_full, "ndim", 0) != 4:
            raise ValueError(
                "h3_song_audio: audio VAE returned %s, expected [B,C,2,T]"
                % (tuple(getattr(audio_full, "shape", ())),)
            )
        got_audio_steps = int(audio_full.shape[-1])
        if got_audio_steps < expected_audio_steps:
            raise RuntimeError(
                "h3_song_audio: target clip needs %d audio latent steps but the audio VAE produced only %d"
                % (expected_audio_steps, got_audio_steps)
            )
        if got_audio_steps > expected_audio_steps:
            _LOG.warning(
                "h3_song_audio: audio VAE produced %d steps for a %d-step target; keeping the first %d so the slice stays aligned to clip_start_seconds",
                got_audio_steps,
                expected_audio_steps,
                expected_audio_steps,
            )
            audio_full = audio_full[..., :expected_audio_steps]

        out_video = target_video.clone()
        out_audio = target_audio.clone()
        out_audio.copy_(audio_full[:1].to(device=out_audio.device, dtype=out_audio.dtype))

        video_steps = 0
        n = 0
        if source_frames is not None:
            if vae is None:
                raise ValueError("h3_song_audio: vae is required when source_frames is connected")
            if getattr(source_frames, "ndim", 0) != 4 or int(source_frames.shape[0]) < 1:
                raise ValueError("h3_song_audio: source_frames must be IMAGE [N,H,W,C]")
            if int(context_length) <= 0:
                raise ValueError("h3_song_audio: context_length must be > 0 when source_frames is connected")
            idx = _cfr_index_map(int(source_frames.shape[0]), float(source_fps), source_frames.device, FPS)
            available = int(idx.numel())
            n = _snap_context_length(context_length, available, target_frames)
            tail_idx = idx[-n:]
            video_tail = source_frames.index_select(0, tail_idx)
            video_tail = _resize_images(video_tail, width, height, crop)
            video_prefix = vae.encode(video_tail)
            if getattr(video_prefix, "ndim", 0) != 5:
                raise ValueError(
                    "h3_song_audio: video VAE returned %s, expected [B,C,T,H,W]"
                    % (tuple(getattr(video_prefix, "shape", ())),)
                )
            video_steps = int(video_prefix.shape[2])
            covered = _pixel_frames(video_steps)
            if covered != n:
                raise RuntimeError(
                    "h3_song_audio: %d context frames encoded to %d video latent steps covering %d frames; refusing a phase-shifted seam"
                    % (n, video_steps, covered)
                )
            if video_steps >= int(target_video.shape[2]):
                raise ValueError("h3_song_audio: video context consumes the whole target latent")
            vp = video_prefix[:1].to(device=out_video.device, dtype=out_video.dtype)
            if tuple(vp.shape[1:2] + vp.shape[3:]) != tuple(out_video.shape[1:2] + out_video.shape[3:]):
                raise ValueError(
                    "h3_song_audio: encoded video prefix shape %s does not match target %s"
                    % (tuple(vp.shape), tuple(out_video.shape))
                )
            out_video[:, :, :video_steps] = vp

        video_mask = torch.ones(
            (1, 1, int(out_video.shape[2]), int(out_video.shape[3]), int(out_video.shape[4])),
            device=out_video.device,
            dtype=torch.float32,
        )
        if video_steps > 0:
            video_mask[:, :, :video_steps] = 0.0

        audio_mask = torch.zeros(
            (1, 1, int(out_audio.shape[2]), int(out_audio.shape[3])),
            device=out_audio.device,
            dtype=torch.float32,
        )

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

        clip_audio = {"waveform": audio_slice, "sample_rate": vae_sr}
        _LOG.info(
            "h3_song_audio: target %d frames (%.3fs); exact song slice %.3fs -> %.3fs encoded to %d audio latent steps and fully preserved; visual prefix %d frames -> %d video steps",
            target_frames,
            target_frames / FPS,
            float(clip_start_seconds),
            float(clip_start_seconds) + target_frames / FPS,
            expected_audio_steps,
            n,
            video_steps,
        )
        return (out, n, clip_audio)
