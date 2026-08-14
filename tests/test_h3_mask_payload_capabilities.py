"""Capability/self-retirement tests for H3 AV-mask payload compatibility."""

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def install_fake(native=False, direct_payload=False):
    for name in list(sys.modules):
        if name == "payloadpkg" or name.startswith("payloadpkg.") or name == "comfy" or name.startswith("comfy."):
            sys.modules.pop(name, None)

    pkg = types.ModuleType("payloadpkg")
    pkg.__path__ = [str(ROOT)]
    sys.modules["payloadpkg"] = pkg

    comfy = types.ModuleType("comfy")
    conds = types.ModuleType("comfy.conds")
    model_base = types.ModuleType("comfy.model_base")
    utils = types.ModuleType("comfy.utils")

    class CONDRegular:
        def __init__(self, cond):
            self.cond = cond

    conds.CONDRegular = CONDRegular
    utils.unpack_latents = lambda mask, shapes: mask

    class MiniMaxH3:
        if direct_payload:
            def extra_conds(self, **kwargs):
                return {
                    "denoise_mask": object(),
                    "audio_denoise_mask": object(),
                }
        else:
            # Deliberately contains neither native output-key string. This
            # simulates a future refactor that delegates extraction elsewhere.
            def extra_conds(self, **kwargs):
                return {}

    if native:
        def process_timestep(self, timestep, x=None, denoise_mask=None, audio_denoise_mask=None, **kwargs):
            return timestep
        def process_denoise_mask(self, masks):
            return masks
        def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
            return latent_image
        MiniMaxH3.process_timestep = process_timestep
        MiniMaxH3.process_denoise_mask = process_denoise_mask
        MiniMaxH3.scale_latent_inpaint = scale_latent_inpaint

    model_base.MiniMaxH3 = MiniMaxH3
    comfy.conds = conds
    comfy.model_base = model_base
    comfy.utils = utils

    sys.modules["comfy"] = comfy
    sys.modules["comfy.conds"] = conds
    sys.modules["comfy.model_base"] = model_base
    sys.modules["comfy.utils"] = utils

    spec = importlib.util.spec_from_file_location(
        "payloadpkg.h3_mask_payload_compat", ROOT / "h3_mask_payload_compat.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, MiniMaxH3


def test_native_mask_payload_is_noop_after_refactor():
    module, cls = install_fake(native=True, direct_payload=False)
    before = cls.extra_conds
    status = module.capability_status()
    assert status["native_av_mask_payload"]
    assert status["native_h3_mask_hooks"]
    assert not status["native_payload_direct"]
    assert module.ensure_av_mask_payload_compat()
    assert cls.extra_conds is before
    assert not module.capability_status()["wrapper_present"]


def test_direct_native_payload_is_noop():
    module, cls = install_fake(native=False, direct_payload=True)
    before = cls.extra_conds
    assert module.capability_status()["native_payload_direct"]
    assert module.ensure_av_mask_payload_compat()
    assert cls.extra_conds is before


def test_legacy_payload_gets_lazy_wrapper():
    module, cls = install_fake(native=False, direct_payload=False)
    before = cls.extra_conds
    assert not module.capability_status()["native_av_mask_payload"]
    assert module.ensure_av_mask_payload_compat()
    assert cls.extra_conds is not before
    assert module.capability_status()["wrapper_present"]
