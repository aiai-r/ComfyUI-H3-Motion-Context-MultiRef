import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("h3batchpkg")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(
    "h3batchpkg.h3_streaming_vhs",
    ROOT / "h3_streaming_vhs.py",
)
streaming = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = streaming
spec.loader.exec_module(streaming)


def test_existing_video_batch_output_can_exclude_boundary_source():
    base_frames = 48
    extension_frames = 100
    context_frames = 10
    sample_rate = 24000
    samples_per_frame = sample_rate // 24
    captured = {}

    patches = (
        mock.patch.object(streaming, "_streams_from_latent", lambda value: value),
        mock.patch.object(streaming, "_pixel_frames", lambda _latent_frames: extension_frames),
        mock.patch.object(streaming, "_snap_av_context_length", lambda *_args: context_frames),
        mock.patch.object(streaming, "_assemble_av_audio", lambda *_args: {
            "waveform": torch.zeros((
                1,
                2,
                (base_frames + extension_frames - context_frames) * samples_per_frame,
            )),
            "sample_rate": sample_rate,
        }),
        mock.patch.object(
            streaming,
            "_existing_base_and_generated_extensions_generator",
            lambda *_args: iter(range(base_frames + extension_frames - context_frames)),
        ),
    )

    def fake_vhs(frames, audio, *_args):
        captured["frames"] = len(frames)
        captured["samples"] = int(audio["waveform"].shape[-1])
        return ("saved",)

    vhs_patch = mock.patch.object(streaming, "_run_vhs_h264", fake_vhs)
    video_latent = torch.zeros((1, 1, 1, 1, 1))
    audio_latent = torch.zeros((1, 1, 1))
    source_frames = torch.zeros((base_frames, 1, 1, 3))
    source_audio = {"waveform": torch.zeros((1, 2, base_frames)), "sample_rate": sample_rate}

    with patches[0], patches[1], patches[2], patches[3], patches[4], vhs_patch:
        result = streaming.MiniMaxH3StreamLiveExtensionAVToVHS().stream_to_vhs(
            video_vae=object(),
            audio_vae=object(),
            start_mode="existing_video",
            active_extensions=1,
            context_frames=context_frames,
            video_overlap_frames=context_frames,
            source_fps=24.0,
            crop="disabled",
            include_source_in_output=False,
            source_frames=source_frames,
            source_audio=source_audio,
            extension_1=(video_latent, audio_latent),
        )

    assert result == ("saved",)
    assert captured["frames"] == extension_frames - context_frames
    assert captured["samples"] == (extension_frames - context_frames) * samples_per_frame
