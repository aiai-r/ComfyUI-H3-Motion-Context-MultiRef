"""Single-execution low-RAM final H3 video output backed by Video Helper Suite.

The normal ComfyUI IMAGE datatype is a materialized ``torch.Tensor``.  Long H3
runs can therefore require tens of GiB just to hold the final RGB movie before
VHS sees it.  These output nodes keep the frame stream internal instead:

    H3 latent -> decode one clip -> resolve seam -> VHS encoder -> release clip

No custom object is exposed as a ComfyUI IMAGE output. The internal one-shot sequence exists only during the call into VHS_VideoCombine.
"""

from __future__ import annotations

import inspect
import logging

import torch

# Keep this module importable even in lightweight/mock test environments that
# replace the implementation modules with partial stubs.  Runtime helpers are
# resolved only when a streaming node actually executes.
FPS = 24.0

def _ext():
    from . import existing_video_extension as module
    return module

def _music():
    from . import h3_song_audio_context as module
    return module

def _cfr_index_map(*a, **k): return _ext()._cfr_index_map(*a, **k)
def _decode_h3_audio_cpu(*a, **k): return _ext()._decode_h3_audio_cpu(*a, **k)
def _decode_h3_video_cpu(*a, **k): return _ext()._decode_h3_video_cpu(*a, **k)
def _fit_waveform(*a, **k): return _ext()._fit_waveform(*a, **k)
def _conform_waveform_length(*a, **k): return _ext()._conform_waveform_length(*a, **k)
def _pixel_frames(*a, **k): return _ext()._pixel_frames(*a, **k)
def _release_decode_memory(*a, **k): return _ext()._release_decode_memory(*a, **k)
def _resample_waveform(*a, **k): return _ext()._resample_waveform(*a, **k)
def _resize_images(*a, **k): return _ext()._resize_images(*a, **k)
def _snap_av_context_length(*a, **k): return _ext()._snap_context_length(*a, **k)
def _stereo_first_batch(*a, **k): return _ext()._stereo_first_batch(*a, **k)
def _streams_from_latent(*a, **k): return _ext()._streams_from_latent(*a, **k)
def _snap_music_context_length(*a, **k): return _music()._snap_context_length(*a, **k)

def sample_boundary_from_frames(*a, **k):
    from . import h3_timing as module
    return module.sample_boundary_from_frames(*a, **k)

_LOG = logging.getLogger("h3_motion_context.streaming_vhs")


def _rgb_gib(frames, height, width):
    return int(frames) * int(height) * int(width) * 3 * 4 / float(1024 ** 3)


def _resolve_vhs_video_combine():
    """Resolve VHS at execution time without making plugin import order fragile."""
    try:
        import nodes as comfy_nodes
    except Exception as exc:  # pragma: no cover - only reachable in broken Comfy installs
        raise RuntimeError(
            "h3_streaming_vhs: could not import ComfyUI's nodes module"
        ) from exc

    cls = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get("VHS_VideoCombine")
    if cls is None:
        raise RuntimeError(
            "h3_streaming_vhs: VHS_VideoCombine is required. Install/enable "
            "ComfyUI-VideoHelperSuite before using the streaming final output."
        )

    combine = getattr(cls, "combine_video", None)
    if not callable(combine):
        raise RuntimeError(
            "h3_streaming_vhs: the installed VHS_VideoCombine node does not expose "
            "the expected combine_video API. Update ComfyUI-VideoHelperSuite."
        )

    # Fail with a useful message instead of a long TypeError if an older VHS
    # build is missing controls used by the direct-stream nodes. Future/wrapped
    # implementations that accept **kwargs remain compatible.
    try:
        parameters = inspect.signature(combine).parameters
    except (TypeError, ValueError):
        parameters = {}
    if parameters and not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        expected = {
            "frame_rate", "loop_count", "images", "filename_prefix", "format",
            "pingpong", "save_output", "prompt", "extra_pnginfo", "audio",
            "unique_id", "pix_fmt", "crf", "save_metadata", "trim_to_audio",
        }
        missing = sorted(expected.difference(parameters))
        if missing:
            raise RuntimeError(
                "h3_streaming_vhs: the installed VideoHelperSuite is too old for "
                "direct H3 streaming; VHS_VideoCombine.combine_video is missing: "
                + ", ".join(missing)
                + ". Update ComfyUI-VideoHelperSuite."
            )
    return cls


class _OneShotFrameSequence:
    """Sequence facade for VHS that primes once and then streams exactly once.

    VHS currently asks for ``len(images)``, reads ``images[0]`` for output
    dimensions/metadata, and then iterates the image source.  Priming the
    generator in ``__getitem__(0)`` avoids decoding Clip 1 twice.
    """

    def __init__(self, frame_count, generator_factory):
        self._frame_count = int(frame_count)
        self._generator_factory = generator_factory
        self._generator = None
        self._first = None
        self._primed = False
        self._iterated = False

    def __len__(self):
        return self._frame_count

    def _prime(self):
        if self._primed:
            return
        self._generator = iter(self._generator_factory())
        try:
            self._first = next(self._generator)
        except StopIteration as exc:
            raise RuntimeError("h3_streaming_vhs: frame stream produced no frames") from exc
        self._primed = True

    def __getitem__(self, index):
        if int(index) != 0:
            raise IndexError(
                "h3_streaming_vhs: internal frame stream only supports the first-frame probe"
            )
        self._prime()
        return self._first

    def __iter__(self):
        if self._iterated:
            raise RuntimeError("h3_streaming_vhs: frame stream is one-shot")
        self._prime()
        self._iterated = True
        first = self._first
        self._first = None
        yield first
        yield from self._generator


def _yield_segments_and_hold(segments, hold_frames):
    """Yield all but the final ``hold_frames`` and return a detached CPU tail.

    The returned tail is the only RGB history kept for the next seam.  It is
    cloned so it does not keep the much larger decoded clip storage alive.
    """
    segments = [seg for seg in segments if seg is not None and int(seg.shape[0]) > 0]
    total = sum(int(seg.shape[0]) for seg in segments)
    hold = max(0, min(int(hold_frames), total))
    emit = total - hold
    remainders = []

    for seg in segments:
        n = int(seg.shape[0])
        take = min(n, emit)
        if take > 0:
            for frame in seg[:take]:
                yield frame
            emit -= take
        if take < n:
            remainders.append(seg[take:])

    if emit != 0:
        raise RuntimeError("h3_streaming_vhs: internal frame accounting error")
    if hold == 0:
        return None

    if not remainders:
        raise RuntimeError("h3_streaming_vhs: failed to retain requested seam tail")
    if len(remainders) == 1:
        tail = remainders[0].detach().to(device="cpu", dtype=torch.float32).clone()
    else:
        tail = torch.cat(remainders, dim=0).detach().to(device="cpu", dtype=torch.float32)
    if int(tail.shape[0]) != hold:
        raise RuntimeError(
            f"h3_streaming_vhs: retained {int(tail.shape[0])} frames, expected {hold}"
        )
    return tail.contiguous()


def _seam_overlaps(raw_frames, contexts, overlap):
    """Return overlap frames for each seam using the existing assembler rules."""
    overlaps = [0] * len(raw_frames)
    write_frame = int(raw_frames[0])
    overlap = max(0, int(overlap))
    for i in range(1, len(raw_frames)):
        ov = min(overlap, int(contexts[i]), write_frame)
        overlaps[i] = ov
        write_frame += int(raw_frames[i]) - int(contexts[i])
    return overlaps


def _generated_frame_generator(video_vae, videos, raw_frames, contexts, overlap, log_prefix):
    """Stream generated H3 clips with the same seam math as the old full buffer."""
    seam_ovs = _seam_overlaps(raw_frames, contexts, overlap)
    tail = None

    for i, video_latent in enumerate(videos):
        decoded = _decode_h3_video_cpu(video_vae, video_latent)
        if int(decoded.shape[0]) != int(raw_frames[i]):
            raise RuntimeError(
                f"{log_prefix}: Clip {i + 1} video decode produced "
                f"{int(decoded.shape[0])} frames; expected {int(raw_frames[i])}"
            )

        if i == 0:
            segments = [decoded]
        else:
            ctx = int(contexts[i])
            ov = int(seam_ovs[i])
            if ov > 0:
                if tail is None or int(tail.shape[0]) != ov:
                    raise RuntimeError(
                        f"{log_prefix}: seam {i} retained tail mismatch "
                        f"({0 if tail is None else int(tail.shape[0])} != {ov})"
                    )
                dst = decoded[ctx - ov : ctx]
                alpha = torch.linspace(
                    0.0, 1.0, ov + 2, dtype=torch.float32, device="cpu"
                )[1:-1].view(-1, 1, 1, 1)
                tail.mul_(1.0 - alpha).add_(dst * alpha)
                del dst, alpha
                segments = [tail, decoded[ctx:]]
            else:
                segments = [decoded[ctx:]]
                tail = None

        next_hold = int(seam_ovs[i + 1]) if i + 1 < len(videos) else 0
        new_tail = yield from _yield_segments_and_hold(segments, next_hold)
        tail = new_tail
        del decoded, segments, new_tail
        _release_decode_memory()

    if tail is not None:
        raise RuntimeError(f"{log_prefix}: unflushed seam tail after final clip")


def _existing_base_and_generated_extensions_generator(
    video_vae,
    source_frames,
    source_idx,
    width,
    height,
    crop,
    extension_videos,
    raw_frames,
    contexts,
    overlap,
):
    """Stream an existing source followed by generated extension clips."""
    # Build the same seam-overlap plan as [base, ext1, ext2, ...].
    all_frames = [int(source_idx.numel())] + [int(x) for x in raw_frames]
    all_contexts = [0] + [int(x) for x in contexts]
    seam_ovs = _seam_overlaps(all_frames, all_contexts, overlap)
    first_hold = int(seam_ovs[1]) if len(all_frames) > 1 else 0

    # Stream the CFR-resampled/resized source in small chunks.  Only its final
    # seam tail is copied out of the source batch.
    base_frames = int(source_idx.numel())
    emit_upto = base_frames - first_hold
    tail_parts = []
    chunk = 32
    for start in range(0, base_frames, chunk):
        end = min(base_frames, start + chunk)
        ids = source_idx[start:end]
        part = source_frames.index_select(0, ids)
        part = _resize_images(part, width, height, crop).detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        local_emit = max(0, min(int(part.shape[0]), emit_upto - start))
        if local_emit > 0:
            for frame in part[:local_emit]:
                yield frame
        if local_emit < int(part.shape[0]):
            tail_parts.append(part[local_emit:].clone())
        del part

    if first_hold > 0:
        tail = tail_parts[0] if len(tail_parts) == 1 else torch.cat(tail_parts, dim=0)
        tail = tail.contiguous()
        if int(tail.shape[0]) != first_hold:
            raise RuntimeError(
                f"h3_streaming_av: retained {int(tail.shape[0])} source frames, expected {first_hold}"
            )
    else:
        tail = None
    del tail_parts
    _release_decode_memory()

    for ext_i, video_latent in enumerate(extension_videos, start=1):
        decoded = _decode_h3_video_cpu(video_vae, video_latent)
        expected = int(raw_frames[ext_i - 1])
        if int(decoded.shape[0]) != expected:
            raise RuntimeError(
                f"h3_streaming_av: Extension {ext_i} video decode produced "
                f"{int(decoded.shape[0])} frames; expected {expected}"
            )
        ctx = int(contexts[ext_i - 1])
        ov = int(seam_ovs[ext_i])
        if ov > 0:
            if tail is None or int(tail.shape[0]) != ov:
                raise RuntimeError(
                    f"h3_streaming_av: Extension {ext_i} retained tail mismatch"
                )
            dst = decoded[ctx - ov : ctx]
            alpha = torch.linspace(0.0, 1.0, ov + 2, dtype=torch.float32)[1:-1].view(
                -1, 1, 1, 1
            )
            tail.mul_(1.0 - alpha).add_(dst * alpha)
            del dst, alpha
            segments = [tail, decoded[ctx:]]
        else:
            tail = None
            segments = [decoded[ctx:]]

        next_hold = int(seam_ovs[ext_i + 1]) if ext_i + 1 < len(all_frames) else 0
        new_tail = yield from _yield_segments_and_hold(segments, next_hold)
        tail = new_tail
        del decoded, segments, new_tail
        _release_decode_memory()

    if tail is not None:
        raise RuntimeError("h3_streaming_av: unflushed seam tail after final extension")


def _assemble_av_audio(
    audio_vae,
    mode,
    base_frames,
    base_audio_latent,
    source_audio,
    ext_streams,
    raw_frames,
    contexts,
):
    """Existing AV audio assembly, unchanged except separated from RGB output."""
    audio_sr = int(
        getattr(
            audio_vae,
            "audio_sample_rate_output",
            getattr(audio_vae, "audio_sample_rate", 44100),
        )
    )
    final_frames = int(base_frames) + sum(
        int(raw_frames[i]) - int(contexts[i]) for i in range(len(raw_frames))
    )
    total_samples = int(round(final_frames / FPS * audio_sr))
    audio_out = torch.empty((1, 2, total_samples), dtype=torch.float32, device="cpu")

    if mode == "existing_video":
        wave = _stereo_first_batch(source_audio["waveform"], "source_audio")
        wave = _resample_waveform(
            wave, int(source_audio["sample_rate"]), audio_sr, "source_audio"
        )
        want = int(round(int(base_frames) / FPS * audio_sr))
        wave = _conform_waveform_length(wave, want, "source audio").detach().to(
            "cpu", torch.float32
        )
    else:
        wave, got_sr = _decode_h3_audio_cpu(audio_vae, base_audio_latent)
        wave = _stereo_first_batch(wave, "starter audio")
        wave = _resample_waveform(wave, got_sr, audio_sr, "starter audio")
        want = int(round(int(base_frames) / FPS * audio_sr))
        wave = _conform_waveform_length(wave, want, "starter audio")

    audio_out[..., :want].copy_(wave[..., :want])
    sample_pos = want
    del wave
    _release_decode_memory()

    cumulative_frames = int(base_frames)
    for i, (_video_latent, audio_latent) in enumerate(ext_streams):
        wave, got_sr = _decode_h3_audio_cpu(audio_vae, audio_latent)
        wave = _stereo_first_batch(wave, f"Extension {i + 1} audio")
        wave = _resample_waveform(wave, got_sr, audio_sr, f"Extension {i + 1} audio")

        seam_frame = cumulative_frames
        extension_start_frame = seam_frame - int(contexts[i])
        extension_end_frame = extension_start_frame + int(raw_frames[i])
        expected_full = (
            sample_boundary_from_frames(extension_end_frame, audio_sr, FPS)
            - sample_boundary_from_frames(extension_start_frame, audio_sr, FPS)
        )
        wave = _conform_waveform_length(
            wave, expected_full, f"Extension {i + 1} full audio"
        )

        seam_sample = sample_boundary_from_frames(seam_frame, audio_sr, FPS)
        extension_start_sample = sample_boundary_from_frames(
            extension_start_frame, audio_sr, FPS
        )
        cut = seam_sample - extension_start_sample
        if cut >= int(wave.shape[-1]):
            raise ValueError(
                f"h3_streaming_av: Extension {i + 1} audio shorter than protected context"
            )
        wave = wave[..., cut:]

        cumulative_frames += int(raw_frames[i]) - int(contexts[i])
        want_total = int(round(cumulative_frames / FPS * audio_sr))
        want = want_total - sample_pos
        wave = _fit_waveform(wave, want, f"Extension {i + 1} audio", pad=False)
        audio_out[..., sample_pos:want_total].copy_(wave[..., :want])
        sample_pos = want_total
        del wave
        _release_decode_memory()

    if sample_pos != total_samples:
        raise RuntimeError(
            f"h3_streaming_av: wrote {sample_pos} audio samples, expected {total_samples}"
        )
    return {"waveform": audio_out, "sample_rate": audio_sr}


def _vhs_h264_inputs(filename_default, trim_default):
    return {
        "filename_prefix": ("STRING", {"default": filename_default}),
        "pix_fmt": (["yuv420p", "yuv420p10le"], {"default": "yuv420p"}),
        "crf": ("INT", {"default": 19, "min": 0, "max": 100, "step": 1}),
        "save_metadata": ("BOOLEAN", {"default": False}),
        "trim_to_audio": ("BOOLEAN", {"default": bool(trim_default)}),
        "save_output": ("BOOLEAN", {"default": True}),
    }


def _run_vhs_h264(
    frames,
    audio,
    filename_prefix,
    pix_fmt,
    crf,
    save_metadata,
    trim_to_audio,
    save_output,
    prompt,
    extra_pnginfo,
    unique_id,
):
    vhs_cls = _resolve_vhs_video_combine()
    vhs = vhs_cls()

    # IMPORTANT: final filename allocation belongs to VHS.  Do not construct a
    # fixed output path in this module: VHS scans the destination directory and
    # advances its numeric counter for repeated runs using the same prefix.
    # Keeping that responsibility here would risk reintroducing the historical
    # bug where each run overwrote the previous final video.
    result = vhs.combine_video(
        frame_rate=24,
        loop_count=0,
        images=frames,
        filename_prefix=str(filename_prefix),
        format="video/h264-mp4",
        pingpong=False,
        save_output=bool(save_output),
        prompt=prompt,
        extra_pnginfo=extra_pnginfo,
        audio=audio,
        unique_id=unique_id,
        pix_fmt=str(pix_fmt),
        crf=int(crf),
        save_metadata=bool(save_metadata),
        trim_to_audio=bool(trim_to_audio),
    )

    # VHS's own browser-side preview is only installed for nodes whose class
    # name is literally VHS_VideoCombine.  These H3 streaming nodes call VHS
    # internally, so preserve the normal VHS ``gifs`` UI payload and also expose
    # the same saved MP4 through ComfyUI's native PreviewVideo representation
    # (``images`` + ``animated``).  This restores a visible final video preview
    # without materializing the final RGB movie as an IMAGE tensor.
    if isinstance(result, dict):
        ui_payload = result.setdefault("ui", {})
        previews = ui_payload.get("gifs") or []
        if previews:
            preview = previews[0]
            if all(key in preview for key in ("filename", "type")):
                ui_payload["images"] = [{
                    "filename": preview["filename"],
                    "subfolder": preview.get("subfolder", ""),
                    "type": preview["type"],
                }]
                ui_payload["animated"] = (True,)
    return result


class MiniMaxH3StreamLiveExtensionAVToVHS:
    """Stream the final AV Extension timeline directly into VHS H.264 MP4."""

    MAX_EXTENSIONS = 6

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "start_mode": ("STRING", {"forceInput": True}),
            "active_extensions": (
                "INT",
                {"default": 1, "min": 1, "max": cls.MAX_EXTENSIONS},
            ),
            "context_frames": ("INT", {"default": 39, "min": 5, "max": 9999}),
            "video_overlap_frames": (
                "INT",
                {"default": 39, "min": 0, "max": 9999},
            ),
            "source_fps": (
                "FLOAT",
                {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001},
            ),
            "crop": (["disabled", "center"], {"default": "disabled"}),
        }
        required.update(_vhs_h264_inputs("video/masked_av_extension", True))
        optional = {
            "source_frames": ("IMAGE", {"lazy": True}),
            "source_audio": ("AUDIO", {"lazy": True}),
            "starter_latent": ("LATENT", {"lazy": True}),
        }
        for i in range(1, cls.MAX_EXTENSIONS + 1):
            optional[f"extension_{i}"] = ("LATENT", {"lazy": True})
        return {
            "required": required,
            "optional": optional,
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    FUNCTION = "stream_to_vhs"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Low-RAM final AV Extension output. Decodes one H3 clip at a time, keeps "
        "only the effective visual seam tail, streams frames into VHS H.264 MP4, "
        "and preserves the existing exact generated-audio timeline assembly."
    )

    def check_lazy_status(
        self,
        video_vae,
        audio_vae,
        start_mode,
        active_extensions,
        context_frames,
        video_overlap_frames,
        source_fps,
        crop,
        filename_prefix,
        pix_fmt,
        crf,
        save_metadata,
        trim_to_audio,
        save_output,
        source_frames=None,
        source_audio=None,
        starter_latent=None,
        **kwargs,
    ):
        needed = []
        if str(start_mode) == "existing_video":
            if source_frames is None:
                needed.append("source_frames")
            if source_audio is None:
                needed.append("source_audio")
        elif starter_latent is None:
            needed.append("starter_latent")
        count = max(1, min(self.MAX_EXTENSIONS, int(active_extensions)))
        for i in range(1, count + 1):
            if kwargs.get(f"extension_{i}") is None:
                needed.append(f"extension_{i}")
        return needed

    def stream_to_vhs(
        self,
        video_vae,
        audio_vae,
        start_mode="existing_video",
        active_extensions=1,
        context_frames=39,
        video_overlap_frames=39,
        source_fps=24.0,
        crop="disabled",
        filename_prefix="video/masked_av_extension",
        pix_fmt="yuv420p",
        crf=19,
        save_metadata=False,
        trim_to_audio=True,
        save_output=True,
        source_frames=None,
        source_audio=None,
        starter_latent=None,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
        **kwargs,
    ):
        count = max(1, min(self.MAX_EXTENSIONS, int(active_extensions)))
        extension_latents = []
        for i in range(1, count + 1):
            value = kwargs.get(f"extension_{i}")
            if value is None:
                raise ValueError(f"h3_streaming_av: extension_{i} is required")
            extension_latents.append(value)

        ext_streams = [_streams_from_latent(x) for x in extension_latents]
        ext_videos = [video for video, _audio in ext_streams]
        raw_frames = [_pixel_frames(int(video.shape[2])) for video in ext_videos]
        width = int(ext_videos[0].shape[4]) * 16
        height = int(ext_videos[0].shape[3]) * 16
        for video in ext_videos[1:]:
            if int(video.shape[4]) * 16 != width or int(video.shape[3]) * 16 != height:
                raise ValueError(
                    "h3_streaming_av: all H3 extension clips must use one resolution"
                )

        mode = str(start_mode)
        if mode == "existing_video":
            if source_frames is None or source_audio is None:
                raise ValueError(
                    "h3_streaming_av: Existing Video start requires source frames/audio"
                )
            source_idx = _cfr_index_map(
                int(source_frames.shape[0]), float(source_fps), source_frames.device, FPS
            )
            base_frames = int(source_idx.numel())
            base_video_latent = None
            base_audio_latent = None
        else:
            if starter_latent is None:
                raise ValueError("h3_streaming_av: generated start requires starter_latent")
            base_video_latent, base_audio_latent = _streams_from_latent(starter_latent)
            base_frames = _pixel_frames(int(base_video_latent.shape[2]))
            source_idx = None
            if (
                int(base_video_latent.shape[4]) * 16 != width
                or int(base_video_latent.shape[3]) * 16 != height
            ):
                raise ValueError(
                    "h3_streaming_av: starter and extensions must use one resolution"
                )

        contexts = []
        available = base_frames
        for frames in raw_frames:
            ctx = _snap_av_context_length(int(context_frames), available, frames)
            contexts.append(ctx)
            available = frames

        final_frames = base_frames + sum(
            int(raw_frames[i]) - int(contexts[i]) for i in range(count)
        )
        audio = _assemble_av_audio(
            audio_vae,
            mode,
            base_frames,
            base_audio_latent,
            source_audio,
            ext_streams,
            raw_frames,
            contexts,
        )

        if mode == "existing_video":
            factory = lambda: _existing_base_and_generated_extensions_generator(
                video_vae,
                source_frames,
                source_idx,
                width,
                height,
                crop,
                ext_videos,
                raw_frames,
                contexts,
                video_overlap_frames,
            )
        else:
            videos = [base_video_latent] + ext_videos
            all_raw = [base_frames] + raw_frames
            all_contexts = [0] + contexts
            factory = lambda: _generated_frame_generator(
                video_vae,
                videos,
                all_raw,
                all_contexts,
                video_overlap_frames,
                "h3_streaming_av",
            )

        frames = _OneShotFrameSequence(final_frames, factory)
        max_hold = max(_seam_overlaps([base_frames] + raw_frames, [0] + contexts, video_overlap_frames))
        _LOG.info(
            "h3_streaming_av: streaming %d frames from %s + %d extensions into VHS; "
            "old final RGB buffer %.2f GiB is not allocated; max retained seam %d frames (%.2f GiB)",
            final_frames,
            mode,
            count,
            _rgb_gib(final_frames, height, width),
            max_hold,
            _rgb_gib(max_hold, height, width),
        )
        return _run_vhs_h264(
            frames,
            audio,
            filename_prefix,
            pix_fmt,
            crf,
            save_metadata,
            trim_to_audio,
            save_output,
            prompt,
            extra_pnginfo,
            unique_id,
        )


class MiniMaxH3FinalizeVHSOutput:
    """Tiny terminal output sink used to keep final streaming behind clip previews.

    The actual streaming node returns the VHS filenames and UI preview, but is
    intentionally not itself an OUTPUT_NODE.  This sink makes the stream part
    of the executable graph without giving the entire all-clips dependency
    chain the same immediate output priority as each per-clip VHS preview.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"filenames": ("VHS_FILENAMES",)}}

    RETURN_TYPES = ()
    FUNCTION = "finalize"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Execution sink for H3 direct-stream final outputs. Keeps per-clip VHS "
        "previews scheduler-prioritized while preserving the low-RAM final stream."
    )

    def finalize(self, filenames):
        return ()


class MiniMaxH3StreamLiveMusicVideoToVHS:
    """Stream the Music Video timeline directly into VHS H.264 MP4."""

    MAX_CLIPS = 20

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "video_vae": ("VAE",),
            "master_audio": ("AUDIO",),
            "active_clips": (
                "INT",
                {"default": 1, "min": 1, "max": cls.MAX_CLIPS},
            ),
            "context_frames": ("INT", {"default": 39, "min": 5, "max": 9999}),
            "video_overlap_frames": (
                "INT",
                {"default": 39, "min": 0, "max": 9999},
            ),
        }
        required.update(_vhs_h264_inputs("video/h3_music_video", False))
        optional = {}
        for i in range(1, cls.MAX_CLIPS + 1):
            optional[f"clip_{i}"] = ("LATENT", {"lazy": True})
        return {
            "required": required,
            "optional": optional,
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    FUNCTION = "stream_to_vhs"
    # Do not make this node an OUTPUT_NODE.  Its lazy all-clips inputs otherwise
    # compete with each clip's VHS preview branch in ComfyUI's output-priority
    # scheduler and can defer previews until the final assembly.  A tiny
    # MiniMaxH3FinalizeVHSOutput sink after this node keeps it executable while
    # the nearer VAEDecode -> VHS preview branches win scheduler priority.
    OUTPUT_NODE = False
    HAS_INTERMEDIATE_OUTPUT = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Low-RAM final Music Video output. Decodes one H3 clip at a time, keeps "
        "only the effective visual seam tail, streams directly into VHS H.264 MP4, "
        "and muxes the untouched master song."
    )

    def check_lazy_status(
        self,
        video_vae,
        master_audio,
        active_clips,
        context_frames,
        video_overlap_frames,
        filename_prefix,
        pix_fmt,
        crf,
        save_metadata,
        trim_to_audio,
        save_output,
        **kwargs,
    ):
        count = max(1, min(self.MAX_CLIPS, int(active_clips)))
        return [
            f"clip_{i}"
            for i in range(1, count + 1)
            if kwargs.get(f"clip_{i}") is None
        ]

    def stream_to_vhs(
        self,
        video_vae,
        master_audio,
        active_clips=1,
        context_frames=39,
        video_overlap_frames=39,
        filename_prefix="video/h3_music_video",
        pix_fmt="yuv420p",
        crf=19,
        save_metadata=False,
        trim_to_audio=False,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
        **kwargs,
    ):
        count = max(1, min(self.MAX_CLIPS, int(active_clips)))
        latents = []
        for i in range(1, count + 1):
            value = kwargs.get(f"clip_{i}")
            if value is None:
                raise ValueError(f"h3_streaming_music: clip_{i} is required")
            latents.append(value)

        streams = [_streams_from_latent(latent) for latent in latents]
        videos = [video for video, _audio in streams]
        raw_frames = [_pixel_frames(int(video.shape[2])) for video in videos]

        width = int(videos[0].shape[4]) * 16
        height = int(videos[0].shape[3]) * 16
        channels = int(videos[0].shape[1])
        for index, video in enumerate(videos[1:], start=2):
            if (
                int(video.shape[4]) * 16 != width
                or int(video.shape[3]) * 16 != height
                or int(video.shape[1]) != channels
            ):
                raise ValueError(
                    f"h3_streaming_music: Clip {index} resolution/latent channels do not match Clip 1"
                )

        contexts = [0]
        for i in range(1, count):
            contexts.append(
                _snap_music_context_length(
                    int(context_frames), raw_frames[i - 1], raw_frames[i]
                )
            )
        final_frames = raw_frames[0] + sum(
            raw_frames[i] - contexts[i] for i in range(1, count)
        )

        sample_rate = int(master_audio["sample_rate"])
        waveform = master_audio["waveform"]
        if getattr(waveform, "ndim", 0) != 3:
            raise ValueError(
                "h3_streaming_music: master_audio waveform must be [B,C,L], got %s"
                % (tuple(getattr(waveform, "shape", ())),)
            )
        audio = {
            "waveform": waveform[:1].detach().to(device="cpu").contiguous(),
            "sample_rate": sample_rate,
        }

        factory = lambda: _generated_frame_generator(
            video_vae,
            videos,
            raw_frames,
            contexts,
            video_overlap_frames,
            "h3_streaming_music",
        )
        frames = _OneShotFrameSequence(final_frames, factory)
        max_hold = max(_seam_overlaps(raw_frames, contexts, video_overlap_frames))
        _LOG.info(
            "h3_streaming_music: streaming %d clips / %d frames into VHS; untouched master song; "
            "old final RGB buffer %.2f GiB is not allocated; max retained seam %d frames (%.2f GiB)",
            count,
            final_frames,
            _rgb_gib(final_frames, height, width),
            max_hold,
            _rgb_gib(max_hold, height, width),
        )
        return _run_vhs_h264(
            frames,
            audio,
            filename_prefix,
            pix_fmt,
            crf,
            save_metadata,
            trim_to_audio,
            save_output,
            prompt,
            extra_pnginfo,
            unique_id,
        )
