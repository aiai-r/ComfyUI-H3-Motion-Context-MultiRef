"""CPU-only structural regression for exact master-song audio masking."""

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

pkg = types.ModuleType("update2pkg")
pkg.__path__ = [str(ROOT)]
sys.modules["update2pkg"] = pkg

compat = types.ModuleType("update2pkg.h3_compat")
compat.ensure_existing_video_compat = lambda: True
sys.modules["update2pkg.h3_compat"] = compat

comfy = types.ModuleType("comfy")
nested_mod = types.ModuleType("comfy.nested_tensor")
utils_mod = types.ModuleType("comfy.utils")
model_base_mod = types.ModuleType("comfy.model_base")


class NestedTensor:
    def __init__(self, xs):
        self.xs = list(xs)

    def unbind(self):
        return tuple(self.xs)

    @property
    def is_nested(self):
        return True


nested_mod.NestedTensor = NestedTensor


def common_upscale(samples, width, height, method, crop):
    return torch.nn.functional.interpolate(
        samples, size=(height, width), mode="bilinear", align_corners=False
    )


utils_mod.common_upscale = common_upscale


class MiniMaxH3:
    def process_denoise_mask(self, x):
        return x

    def scale_latent_inpaint(self, *args, **kwargs):
        return None


model_base_mod.MiniMaxH3 = MiniMaxH3
comfy.nested_tensor = nested_mod
comfy.utils = utils_mod
comfy.model_base = model_base_mod
sys.modules["comfy"] = comfy
sys.modules["comfy.nested_tensor"] = nested_mod
sys.modules["comfy.utils"] = utils_mod
sys.modules["comfy.model_base"] = model_base_mod


class Functional:
    @staticmethod
    def resample(w, src, dst):
        want = round(w.shape[-1] * dst / src)
        return torch.nn.functional.interpolate(
            w.reshape(-1, 1, w.shape[-1]),
            size=want,
            mode="linear",
            align_corners=False,
        ).reshape(w.shape[0], w.shape[1], want)


ta = types.ModuleType("torchaudio")
ta.functional = Functional
sys.modules["torchaudio"] = ta

# existing_video_extension is a dependency of the new module.
spec_base = importlib.util.spec_from_file_location(
    "update2pkg.existing_video_extension", ROOT / "existing_video_extension.py"
)
base = importlib.util.module_from_spec(spec_base)
sys.modules[spec_base.name] = base
spec_base.loader.exec_module(base)

spec = importlib.util.spec_from_file_location(
    "update2pkg.h3_song_audio_context", ROOT / "h3_song_audio_context.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class VideoVAE:
    def encode(self, frames):
        n = frames.shape[0]
        t = 2 if n <= 5 else ((n - 5) // 17) * 5 + 2
        h, w = frames.shape[1], frames.shape[2]
        return torch.ones((1, 24, t, h // 16, w // 16), dtype=torch.float32) * 0.25


class AudioVAE:
    audio_sample_rate = 32000

    def encode(self, x):
        t = round(x.shape[1] / 32000 * 40)
        return torch.ones((1, 32, 2, t), dtype=torch.float32) * 0.5


def test_master_song_audio_is_fully_preserved_while_only_video_prefix_is_masked():
    # 141 H3 frames -> 42 video latent steps / 235 audio latent steps.
    video = torch.zeros((1, 24, 42, 2, 4))
    audio = torch.zeros((1, 32, 2, 235))
    latent = {"samples": NestedTensor((video, audio))}

    previous_frames = torch.rand((120, 32, 64, 3))
    master_audio = {
        "waveform": torch.rand((1, 2, 32000 * 30)),
        "sample_rate": 32000,
    }

    node = module.MiniMaxH3SongMaskedAVContext()
    out, n, clip_audio = node.prepare(
        latent,
        AudioVAE(),
        master_audio,
        clip_start_seconds=3.25,
        context_length=39,
        source_fps=24.0,
        crop="disabled",
        vae=VideoVAE(),
        source_frames=previous_frames,
    )

    assert n == 39
    ov, oa = out["samples"].unbind()
    vm, am = out["noise_mask"].unbind()

    assert torch.allclose(ov[:, :, :12], torch.full_like(ov[:, :, :12], 0.25))
    assert torch.count_nonzero(ov[:, :, 12:]) == 0
    assert torch.allclose(oa, torch.full_like(oa, 0.5))
    assert vm[:, :, :12].max() == 0 and vm[:, :, 12:].min() == 1
    assert am.max() == 0 and am.min() == 0
    assert clip_audio["sample_rate"] == 32000
    assert clip_audio["waveform"].shape[-1] == round(141 / 24 * 32000)


class FloorAudioVAE:
    """Simulate an encoder that floors temporal output instead of rounding."""
    audio_sample_rate = 32000

    def encode(self, x):
        t = int(x.shape[1] / 32000 * 40)
        return torch.ones((1, 32, 2, t), dtype=torch.float32) * 0.75


def test_master_song_audio_124_frame_rounding_shortfall_is_grid_padded_not_rejected():
    # Native H3 target allocation: 124 / 24 * 40 = 206.666... -> 207 steps.
    # A floor-style audio encoder fed only the exact picture duration would make
    # 206 steps. The node must encode the full rounded audio-grid span instead.
    video = torch.zeros((1, 24, 37, 2, 4))  # 37 video tokens cover 124 frames
    audio = torch.zeros((1, 32, 2, 207))
    latent = {"samples": NestedTensor((video, audio))}

    master_audio = {
        "waveform": torch.rand((1, 2, 32000 * 10)),
        "sample_rate": 32000,
    }

    node = module.MiniMaxH3SongMaskedAVContext()
    out, n, clip_audio = node.prepare(
        latent,
        FloorAudioVAE(),
        master_audio,
        clip_start_seconds=1.0,
        context_length=0,
        source_fps=24.0,
        crop="disabled",
        vae=None,
        source_frames=None,
    )

    _, oa = out["samples"].unbind()
    _, am = out["noise_mask"].unbind()
    assert n == 0
    assert oa.shape[-1] == 207
    assert torch.allclose(oa, torch.full_like(oa, 0.75))
    assert am.max() == 0 and am.min() == 0
    # clip_audio remains exactly picture-duration audio, not the grid lookahead.
    assert clip_audio["waveform"].shape[-1] == round(124 / 24 * 32000)


def test_master_song_clip_audio_uses_absolute_timeline_sample_endpoints():
    # Use the same kind of non-integral 24-fps boundary that appears in later
    # music-video clips. Absolute start/end rounding differs by one sample from
    # rounding the duration independently.
    video = torch.zeros((1, 24, 107, 2, 4))  # 107 video tokens cover 362 frames
    audio = torch.zeros((1, 32, 2, round(362 / 24 * 40)))
    latent = {"samples": NestedTensor((video, audio))}
    master_audio = {
        "waveform": torch.rand((1, 2, 32000 * 40)),
        "sample_rate": 32000,
    }
    start_seconds = 323 / 24
    node = module.MiniMaxH3SongMaskedAVContext()
    _out, _n, clip_audio = node.prepare(
        latent, AudioVAE(), master_audio,
        clip_start_seconds=start_seconds, context_length=0,
        source_fps=24.0, crop="disabled", vae=None, source_frames=None,
    )
    expected = round((start_seconds + 362 / 24) * 32000) - round(start_seconds * 32000)
    assert expected == 482666
    assert clip_audio["waveform"].shape[-1] == expected
