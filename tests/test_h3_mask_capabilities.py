"""Capability detection tests for the PR #15375 model-level compatibility."""

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def install_fake(native, aug15=False):
    for name in list(sys.modules):
        if name == "maskpkg" or name.startswith("maskpkg.") or name == "comfy" or name.startswith("comfy."):
            sys.modules.pop(name, None)

    pkg = types.ModuleType("maskpkg")
    pkg.__path__ = [str(ROOT)]
    sys.modules["maskpkg"] = pkg

    comfy = types.ModuleType("comfy")
    model_base = types.ModuleType("comfy.model_base")
    ldm = types.ModuleType("comfy.ldm")
    minimax = types.ModuleType("comfy.ldm.minimax")
    h3m = types.ModuleType("comfy.ldm.minimax.model")

    class Base:
        pass

    class MiniMaxH3(Base):
        pass

    class MiniMaxH3Model:
        if native:
            def forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, denoise_mask=None, audio_denoise_mask=None, **kwargs):
                return None
            def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, denoise_mask=None, audio_denoise_mask=None, **kwargs):
                return None
        else:
            def forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
                return None
            def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
                return None

    class FinalLayer:
        def forward(self, x, t_emb, video_seg, audio_seg):
            return None

    if native:
        if aug15:
            # PR #15375 head after 989e7a9: process_denoise_mask is gone;
            # token-grid blend alignment happens in scale_latent_inpaint.
            def scale_latent_inpaint(self, sigma, noise, latent_image, x=None, denoise_mask=None, **kwargs):
                return latent_image
            MiniMaxH3.scale_latent_inpaint = scale_latent_inpaint
        else:
            def process_denoise_mask(self, masks):
                return masks
            def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
                return latent_image
            MiniMaxH3.process_denoise_mask = process_denoise_mask
            MiniMaxH3.scale_latent_inpaint = scale_latent_inpaint
        h3m.mask_row_values = lambda *args: None
        h3m._mod_row = lambda vecs, row, dtype: vecs[row]

    model_base.MiniMaxH3 = MiniMaxH3
    # Globals needed to exec compatibility functions; they are not numerically run here.
    model_base.torch = torch
    model_base.utils = types.SimpleNamespace(unpack_latents=lambda *a: [], pack_latents=lambda *a: (None,))
    model_base.comfy = comfy
    h3m.torch = torch
    h3m.MiniMaxH3Model = MiniMaxH3Model
    h3m.FinalLayer = FinalLayer

    minimax.model = h3m
    ldm.minimax = minimax
    comfy.model_base = model_base
    comfy.ldm = ldm

    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_base"] = model_base
    sys.modules["comfy.ldm"] = ldm
    sys.modules["comfy.ldm.minimax"] = minimax
    sys.modules["comfy.ldm.minimax.model"] = h3m

    spec = importlib.util.spec_from_file_location(
        "maskpkg.h3_mask_compat", ROOT / "h3_mask_compat.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, MiniMaxH3, MiniMaxH3Model


def test_native_mask_engine_is_noop():
    module, cls, model_cls = install_fake(native=True)
    before_forward = model_cls.forward
    before_process = cls.process_denoise_mask
    status = module.capability_status()
    assert status["mask_engine_native"]
    assert status["process_denoise_mask_native"]
    assert status["scale_latent_inpaint_native"]
    assert module.ensure_h3_mask_compat()
    assert model_cls.forward is before_forward
    assert cls.process_denoise_mask is before_process



def test_aug15_native_mask_engine_is_noop_without_process_hook():
    module, cls, model_cls = install_fake(native=True, aug15=True)
    before_forward = model_cls.forward
    before_scale = cls.scale_latent_inpaint
    assert "process_denoise_mask" not in cls.__dict__
    status = module.capability_status()
    assert status["mask_engine_native"]
    assert status["native_inpaint_mask_alignment"]
    assert status["mask_model_ready"]
    assert module.ensure_h3_mask_compat()
    assert cls.scale_latent_inpaint is before_scale
    assert model_cls.forward is before_forward
    assert "process_denoise_mask" not in cls.__dict__
    assert module.is_ready()

def test_legacy_mask_engine_gets_compatibility():
    module, cls, model_cls = install_fake(native=False)
    assert not module.capability_status()["mask_engine_complete"]
    assert module.ensure_h3_mask_compat()
    status = module.capability_status()
    assert status["mask_engine_compat"]
    assert status["process_denoise_mask_compat"]
    assert status["scale_latent_inpaint_compat"]
    assert module.is_ready()
