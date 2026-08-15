"""CPU-only structural regression for Update 2 existing-video extension."""

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

# Fake package so relative imports resolve.
pkg = types.ModuleType("update2pkg")
pkg.__path__ = [str(ROOT)]
sys.modules["update2pkg"] = pkg

compat = types.ModuleType("update2pkg.h3_compat")
compat.ensure_existing_video_compat = lambda: True
sys.modules["update2pkg.h3_compat"] = compat

# Minimal comfy package used by existing_video_extension.py.
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

# Fake torchaudio with length-correct resampling.
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

spec = importlib.util.spec_from_file_location(
    "update2pkg.existing_video_extension", ROOT / "existing_video_extension.py"
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


def test_39_frame_prefix_and_exact_assembly():
    # 141 H3 frames -> 42 video latent steps / 235 audio latent steps.
    video = torch.zeros((1, 24, 42, 2, 4))
    audio = torch.zeros((1, 32, 2, 235))
    latent = {"samples": NestedTensor((video, audio))}

    source_frames = torch.rand((120, 32, 64, 3))
    source_audio = {
        "waveform": torch.rand((1, 2, 160000)),
        "sample_rate": 32000,
    }

    node = module.MiniMaxH3ExistingVideoMaskedContext()
    out, n = node.prepare(
        latent,
        VideoVAE(),
        AudioVAE(),
        source_frames,
        source_audio,
        24.0,
        39,
        "disabled",
    )

    assert n == 39
    ov, oa = out["samples"].unbind()
    vm, am = out["noise_mask"].unbind()

    assert ov.shape == (1, 24, 42, 2, 4)
    assert oa.shape == (1, 32, 2, 235)
    assert torch.allclose(ov[:, :, :12], torch.full_like(ov[:, :, :12], 0.25))
    assert torch.count_nonzero(ov[:, :, 12:]) == 0
    assert torch.allclose(oa[..., :65], torch.full_like(oa[..., :65], 0.5))
    assert torch.count_nonzero(oa[..., 65:]) == 0
    assert vm[:, :, :12].max() == 0 and vm[:, :, 12:].min() == 1
    assert am[..., :65].max() == 0 and am[..., 65:].min() == 1

    continuation_images = torch.rand((102, 32, 64, 3))
    continuation_audio = {
        "waveform": torch.rand((1, 2, 136000)),
        "sample_rate": 32000,
    }
    assembler = module.MiniMaxH3AssembleExtension()
    images, joined_audio = assembler.assemble(
        source_frames,
        source_audio,
        24.0,
        continuation_images,
        continuation_audio,
        24.0,
        "disabled",
    )
    assert images.shape[0] == 222
    assert joined_audio["waveform"].shape[-1] == round(222 / 24 * 32000)


def test_generated_latent_av_context_copies_tail_without_reencode():
    # 141 H3 frames -> 42 video steps / 235 audio steps.
    src_video = torch.arange(42, dtype=torch.float32).view(1, 1, 42, 1, 1).repeat(1, 24, 1, 2, 4)
    src_audio = torch.arange(235, dtype=torch.float32).view(1, 1, 1, 235).repeat(1, 32, 2, 1)
    source = {"samples": NestedTensor((src_video, src_audio))}

    dst_video = torch.zeros((1, 24, 42, 2, 4))
    dst_audio = torch.zeros((1, 32, 2, 235))
    target = {"samples": NestedTensor((dst_video, dst_audio))}

    node = module.MiniMaxH3GeneratedAVMaskedContext()
    out, n = node.prepare(target, source, 39)
    assert n == 39
    ov, oa = out["samples"].unbind()
    vm, am = out["noise_mask"].unbind()

    # 39 frames = 12 H3 video latent steps = 65 H3 audio latent steps.
    assert torch.equal(ov[:, :, :12], src_video[:, :, -12:])
    assert torch.equal(oa[..., :65], src_audio[..., -65:])
    assert torch.count_nonzero(ov[:, :, 12:]) == 0
    assert torch.count_nonzero(oa[..., 65:]) == 0
    assert vm[:, :, :12].max() == 0 and vm[:, :, 12:].min() == 1
    assert am[..., :65].max() == 0 and am[..., 65:].min() == 1


def test_start_masked_context_lazy_start_modes_and_live_starter():
    node = module.MiniMaxH3StartMaskedContext()
    assert node.check_lazy_status(
        None, None, None, 'load_video', 39, 0, 'starter', 24.0, 'disabled'
    ) == ['source_frames', 'source_audio']
    assert node.check_lazy_status(
        None, None, None, 'generate_starter', 39, 0, 'starter', 24.0, 'disabled'
    ) == ['live_starter_latent']
    assert node.check_lazy_status(
        None, None, None, 'generate_starter', 39, 1, 'starter', 24.0, 'disabled'
    ) == []

    src_video = torch.arange(42, dtype=torch.float32).view(1, 1, 42, 1, 1).repeat(1, 24, 1, 2, 4)
    src_audio = torch.arange(235, dtype=torch.float32).view(1, 1, 1, 235).repeat(1, 32, 2, 1)
    starter = {'samples': NestedTensor((src_video, src_audio))}
    dst_video = torch.zeros((1, 24, 42, 2, 4))
    dst_audio = torch.zeros((1, 32, 2, 235))
    target = {'samples': NestedTensor((dst_video, dst_audio))}

    out, n = node.prepare(
        target, None, None, 'generate_starter', 39, 0, 'starter', 24.0,
        'disabled', live_starter_latent=starter,
    )
    ov, oa = out['samples'].unbind()
    assert n == 39
    assert torch.equal(ov[:, :, :12], src_video[:, :, -12:])
    assert torch.equal(oa[..., :65], src_audio[..., -65:])
