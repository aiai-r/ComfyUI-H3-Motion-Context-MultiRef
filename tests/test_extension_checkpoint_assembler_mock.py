"""CPU/ffmpeg integration smoke test for streamed multi-clip AV extension assembly."""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def install_modules(tmpdir):
    for name in list(sys.modules):
        if name == "extpkg" or name.startswith("extpkg."):
            sys.modules.pop(name, None)

    pkg = types.ModuleType("extpkg")
    pkg.__path__ = [str(ROOT)]
    sys.modules["extpkg"] = pkg

    # Minimal comfy package for existing_video_extension imports.
    comfy = types.ModuleType("comfy")
    nested = types.ModuleType("comfy.nested_tensor")
    utils = types.ModuleType("comfy.utils")
    model_management = types.ModuleType("comfy.model_management")
    model_base = types.ModuleType("comfy.model_base")

    class NestedTensor:
        def __init__(self, xs): self.xs = list(xs)
        def unbind(self): return tuple(self.xs)
        @property
        def is_nested(self): return True

    nested.NestedTensor = NestedTensor
    utils.common_upscale = lambda x, w, h, method, crop: torch.nn.functional.interpolate(
        x, size=(h, w), mode="bilinear", align_corners=False
    )
    model_management.soft_empty_cache = lambda: None

    class MiniMaxH3:
        def process_denoise_mask(self, x): return x
        def scale_latent_inpaint(self, *args, **kwargs): return None
    model_base.MiniMaxH3 = MiniMaxH3

    comfy.nested_tensor = nested
    comfy.utils = utils
    comfy.model_management = model_management
    comfy.model_base = model_base
    sys.modules["comfy"] = comfy
    sys.modules["comfy.nested_tensor"] = nested
    sys.modules["comfy.utils"] = utils
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.model_base"] = model_base

    # folder_paths output plumbing.
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: str(tmpdir)
    def get_save_image_path(prefix, output_dir):
        p = Path(output_dir) / prefix
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p.parent), p.name, 1, "", prefix
    fp.get_save_image_path = get_save_image_path
    sys.modules["folder_paths"] = fp

    compat = types.ModuleType("extpkg.h3_compat")
    compat.ensure_existing_video_compat = lambda: True
    sys.modules["extpkg.h3_compat"] = compat

    # Load actual timing + extension + checkpoint modules under the fake package.
    for name in ["h3_timing", "existing_video_extension", "h3_checkpoint_resume"]:
        spec = importlib.util.spec_from_file_location(f"extpkg.{name}", ROOT / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return sys.modules["extpkg.h3_checkpoint_resume"]


class VideoVAE:
    def decode(self, latent):
        # Reconstruct the exact H3 pixel-frame count from latent T.
        pattern = (1, 4, 4, 4, 4)
        n = sum(pattern[k % 5] for k in range(int(latent.shape[2])))
        h, w = int(latent.shape[3]) * 16, int(latent.shape[4]) * 16
        # Simple deterministic gradient frames in Comfy IMAGE layout.
        vals = torch.linspace(0.1, 0.9, n).view(n, 1, 1, 1)
        return vals.repeat(1, h, w, 3)


class AudioVAE:
    audio_sample_rate = 32000
    audio_sample_rate_output = 32000
    def decode(self, latent):
        samples = int(round(int(latent.shape[-1]) / 40 * self.audio_sample_rate))
        # VAE decode layout [B, samples, channels].
        t = torch.linspace(-0.2, 0.2, samples).view(1, samples, 1)
        return t.repeat(1, 1, 2)


def test_streamed_extension_checkpoint_assembler_two_clips():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        module = install_modules(tmp)
        prefix = tmp / "h3_extension_checkpoints" / "clip"
        prefix.parent.mkdir(parents=True, exist_ok=True)

        # 141 frames => 42 video latent steps, 235 audio latent steps.
        for i in (1, 2):
            video = torch.zeros((1, 24, 42, 2, 4), dtype=torch.float32) + i
            audio = torch.zeros((1, 32, 2, 235), dtype=torch.float32) + i
            save_file(
                {"video": video, "audio": audio},
                str(prefix) + f"_{i:05d}.safetensors",
            )

        source_frames = torch.rand((60, 32, 64, 3), dtype=torch.float32)
        source_audio = {
            "waveform": torch.rand((1, 2, round(60 / 24 * 32000)), dtype=torch.float32) * 0.1,
            "sample_rate": 32000,
        }

        node = module.MiniMaxH3AssembleExtensionCheckpoints()
        result = node.assemble(
            VideoVAE(), AudioVAE(), source_frames, source_audio, 24.0,
            checkpoint_path=str(prefix), clip_count=2,
            context_frames=39, overlap_frames=39, fps=24.0,
            crop="disabled", assembly_mode="saved_only",
            filename_prefix="video/test_extension", pix_fmt="yuv420p",
            crf=30, trim_to_audio=True,
        )
        if isinstance(result, dict):
            out, frames = result["result"] if isinstance(result, dict) else (result.result if hasattr(result, "result") else result)
        elif hasattr(result, "result"):
            out, frames = result.result
        else:
            out, frames = result
        # source 60 + 2 * (141 - 39) unique extension frames
        assert frames == 264
        assert Path(out).is_file()
        assert Path(out).stat().st_size > 0


def test_streamed_generated_starter_plus_extension_assembler():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        module = install_modules(tmp)
        ext_prefix = tmp / 'h3_extension_checkpoints' / 'clip'
        starter_prefix = tmp / 'h3_extension_checkpoints' / 'starter'
        ext_prefix.parent.mkdir(parents=True, exist_ok=True)

        # Starter + two extension checkpoints, each 141 frames / 235 audio steps.
        starter_video = torch.zeros((1, 24, 42, 2, 4), dtype=torch.float32) + 0.5
        starter_audio = torch.zeros((1, 32, 2, 235), dtype=torch.float32) + 0.5
        save_file({'video': starter_video, 'audio': starter_audio}, str(starter_prefix) + '_00001.safetensors')
        for i in (1, 2):
            video = torch.zeros((1, 24, 42, 2, 4), dtype=torch.float32) + i
            audio = torch.zeros((1, 32, 2, 235), dtype=torch.float32) + i
            save_file({'video': video, 'audio': audio}, str(ext_prefix) + f'_{i:05d}.safetensors')

        node = module.MiniMaxH3AssembleStarterOrExtensionCheckpoints()
        result = node.assemble(
            VideoVAE(), AudioVAE(), start_mode='generate_starter', source_fps=24.0,
            starter_checkpoint_path=str(starter_prefix), checkpoint_path=str(ext_prefix),
            clip_count=2, context_frames=39, overlap_frames=39, fps=24.0,
            crop='disabled', assembly_mode='saved_only',
            filename_prefix='video/test_starter_extension', pix_fmt='yuv420p',
            crf=30, trim_to_audio=True,
        )
        out, frames = result["result"] if isinstance(result, dict) else (result.result if hasattr(result, "result") else result)
        assert frames == 141 + 2 * (141 - 39)
        assert Path(out).is_file()
        assert Path(out).stat().st_size > 0


def test_streamed_music_checkpoint_assembler_and_direct_preview_payload():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        module = install_modules(tmp)
        prefix = tmp / "h3_checkpoints" / "clip"
        prefix.parent.mkdir(parents=True, exist_ok=True)

        for i in (1, 2):
            video = torch.zeros((1, 24, 42, 2, 4), dtype=torch.float32) + i
            audio = torch.zeros((1, 32, 2, 235), dtype=torch.float32) + i
            save_file({"video": video, "audio": audio}, str(prefix) + f"_{i:05d}.safetensors")

        master_audio = {
            "waveform": torch.rand((1, 2, 12 * 32000), dtype=torch.float32) * 0.1,
            "sample_rate": 32000,
        }

        node = module.MiniMaxH3AssembleCheckpoints()
        result = node.assemble(
            VideoVAE(), master_audio,
            checkpoint_path=str(prefix), clip_count=2,
            context_frames=39, overlap_frames=39, fps=24.0,
            assembly_mode="saved_only", filename_prefix="video/test_music",
            pix_fmt="yuv420p", crf=30, trim_to_audio=True,
        )
        assert isinstance(result, dict)
        assert "ui" in result and result["ui"]["gifs"]
        out, frames = result["result"] if isinstance(result, dict) else (result.result if hasattr(result, "result") else result)
        assert frames == 141 + (141 - 39)
        assert Path(out).is_file()
        assert Path(out).stat().st_size > 0
        assert result["ui"]["gifs"][0]["filename"] == Path(out).name


def test_checkpoint_resume_fingerprints_preserve_normal_comfy_cache():
    """Normal live chains must not advertise themselves as changed every queue."""
    import math
    import os
    import time

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        module = install_modules(tmp)
        ext = sys.modules["extpkg.existing_video_extension"]

        save_a = module.MiniMaxH3CheckpointSave.IS_CHANGED(
            filename_prefix="h3_checkpoints/clip", clip_index=1
        )
        save_b = module.MiniMaxH3CheckpointSave.IS_CHANGED(
            filename_prefix="h3_checkpoints/clip", clip_index=1
        )
        assert save_a == save_b
        assert not (isinstance(save_a, float) and math.isnan(save_a))

        live_tail_a = module.MiniMaxH3ResumeTailFrames.IS_CHANGED(
            resume_from_clip=0, next_clip_index=3,
            checkpoint_path="h3_checkpoints/clip"
        )
        live_tail_b = module.MiniMaxH3ResumeTailFrames.IS_CHANGED(
            resume_from_clip=0, next_clip_index=3,
            checkpoint_path="h3_checkpoints/clip"
        )
        assert live_tail_a == live_tail_b == "live-tail:3"

        live_latent_a = module.MiniMaxH3ResumeOrLiveLatent.IS_CHANGED(
            resume_from_clip=0, next_clip_index=3,
            checkpoint_path="h3_extension_checkpoints/clip"
        )
        live_latent_b = module.MiniMaxH3ResumeOrLiveLatent.IS_CHANGED(
            resume_from_clip=0, next_clip_index=3,
            checkpoint_path="h3_extension_checkpoints/clip"
        )
        assert live_latent_a == live_latent_b == "live-latent:3"

        assert ext.MiniMaxH3StartMaskedContext.IS_CHANGED(
            start_mode="generate_starter", resume_from_extension=0,
            starter_checkpoint_path="h3_extension_checkpoints/starter"
        ) == "live-start:generate_starter"

        # Explicit resume mode must still notice when the previous fixed slot is
        # replaced on disk, even though no visible workflow widget changed.
        prefix = tmp / "h3_checkpoints" / "clip"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        ckpt = Path(str(prefix) + "_00002.safetensors")
        save_file({
            "video": torch.zeros((1, 24, 2, 1, 1)),
            "audio": torch.zeros((1, 32, 2, 2)),
        }, str(ckpt))
        first = module.MiniMaxH3ResumeTailFrames.IS_CHANGED(
            resume_from_clip=3, next_clip_index=3, checkpoint_path=str(prefix)
        )
        # Force a distinct mtime even on coarse filesystems.
        st = ckpt.stat()
        os.utime(ckpt, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
        second = module.MiniMaxH3ResumeTailFrames.IS_CHANGED(
            resume_from_clip=3, next_clip_index=3, checkpoint_path=str(prefix)
        )
        assert first != second


def test_disk_backed_path_nodes_cache_only_strings_and_reload_saved_latent():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        module = install_modules(tmp)
        nested = sys.modules["comfy.nested_tensor"]

        video = torch.zeros((1, 24, 2, 1, 1), dtype=torch.float32)
        audio = torch.zeros((1, 32, 2, 2), dtype=torch.float32)
        latent = {"samples": nested.NestedTensor((video, audio))}

        saver = module.MiniMaxH3CheckpointSavePath()
        result = saver.save(latent, filename_prefix="h3_checkpoints/pathonly", clip_index=1)
        assert isinstance(result, tuple) and len(result) == 1
        path = result[0]
        assert isinstance(path, str) and Path(path).is_file()
        assert module.MiniMaxH3CheckpointSavePath.RETURN_TYPES == ("STRING",)

        loader = module.MiniMaxH3CheckpointLoadPath()
        loaded = loader.load(path)[0]
        lv, la = loaded["samples"].unbind()
        assert torch.equal(lv, video)
        assert torch.equal(la, audio)

        tail = module.MiniMaxH3CheckpointTailFrames()
        assert tail.check_lazy_status(VideoVAE(), 0, 2, "h3_checkpoints/pathonly", 5, None) == ["checkpoint_signal"]
        assert tail.check_lazy_status(VideoVAE(), 2, 2, "h3_checkpoints/pathonly", 5, None) == []


def test_persistent_checkpoint_gate_skips_sampler_after_cache_loss_and_invalidates_downstream():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        module = install_modules(tmp)
        nested = sys.modules["comfy.nested_tensor"]

        video = torch.zeros((1, 24, 2, 1, 1), dtype=torch.float32)
        audio = torch.zeros((1, 32, 2, 2), dtype=torch.float32)
        latent = {"samples": nested.NestedTensor((video, audio))}

        prompt = {
            "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": 111}},
            "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "latent_image": ["9", 0]}},
            "9": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"prompt": "clip one"}},
            "20": {"class_type": "MiniMaxH3CheckpointSavePath", "inputs": {"latent": ["11", 0], "filename_prefix": "h3_checkpoints/persist", "clip_index": 1}},
            "30": {"class_type": "RandomNoise", "inputs": {"noise_seed": 222}},
            "31": {"class_type": "MiniMaxH3GeneratedAVMaskedContext", "inputs": {"source_checkpoint": ["20", 0]}},
            "32": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["30", 0], "latent_image": ["31", 0]}},
            "40": {"class_type": "MiniMaxH3CheckpointSavePath", "inputs": {"latent": ["32", 0], "filename_prefix": "h3_checkpoints/persist", "clip_index": 2}},
        }

        gate1 = module.MiniMaxH3CheckpointSavePath()
        assert gate1.check_lazy_status("h3_checkpoints/persist", 1, None, prompt, "20") == ["latent"]
        p1 = gate1.save(latent=latent, filename_prefix="h3_checkpoints/persist", clip_index=1, prompt=prompt, unique_id="20")[0]
        assert Path(p1).is_file()
        # Simulate total Comfy RAM-cache loss: a fresh node instance still skips
        # its sampler because the signature persisted inside the checkpoint.
        gate1b = module.MiniMaxH3CheckpointSavePath()
        assert gate1b.check_lazy_status("h3_checkpoints/persist", 1, None, prompt, "20") == []
        assert gate1b.save(latent=None, filename_prefix="h3_checkpoints/persist", clip_index=1, prompt=prompt, unique_id="20")[0] == p1

        gate2 = module.MiniMaxH3CheckpointSavePath()
        assert gate2.check_lazy_status("h3_checkpoints/persist", 2, None, prompt, "40") == ["latent"]
        p2 = gate2.save(latent=latent, filename_prefix="h3_checkpoints/persist", clip_index=2, prompt=prompt, unique_id="40")[0]
        assert Path(p2).is_file()
        assert module.MiniMaxH3CheckpointSavePath().check_lazy_status(
            "h3_checkpoints/persist", 2, None, prompt, "40"
        ) == []

        # Changing Clip 2's own seed invalidates Clip 2 only.
        own_changed = {k: {"class_type": v["class_type"], "inputs": dict(v["inputs"])} for k, v in prompt.items()}
        own_changed["30"]["inputs"]["noise_seed"] = 223
        assert module.MiniMaxH3CheckpointSavePath().check_lazy_status(
            "h3_checkpoints/persist", 1, None, own_changed, "20"
        ) == []
        assert module.MiniMaxH3CheckpointSavePath().check_lazy_status(
            "h3_checkpoints/persist", 2, None, own_changed, "40"
        ) == ["latent"]

        # Changing Clip 1 propagates into Clip 2's persistent signature because
        # Clip 2's submitted ancestor graph includes Clip 1's checkpoint gate.
        upstream_changed = {k: {"class_type": v["class_type"], "inputs": dict(v["inputs"])} for k, v in prompt.items()}
        upstream_changed["10"]["inputs"]["noise_seed"] = 112
        assert module.MiniMaxH3CheckpointSavePath().check_lazy_status(
            "h3_checkpoints/persist", 1, None, upstream_changed, "20"
        ) == ["latent"]
        assert module.MiniMaxH3CheckpointSavePath().check_lazy_status(
            "h3_checkpoints/persist", 2, None, upstream_changed, "40"
        ) == ["latent"]
