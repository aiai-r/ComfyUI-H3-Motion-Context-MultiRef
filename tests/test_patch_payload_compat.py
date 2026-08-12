"""Mock tests for the single-owner MiniMaxH3.extra_conds compatibility wrapper."""

import functools
import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(cls):
    # Clear prior fake modules and package module.
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

    class Cond:
        def __init__(self, cond):
            self.cond = cond

    conds.CONDRegular = Cond
    conds.CONDConstant = Cond
    model_base.MiniMaxH3 = cls
    utils.unpack_latents = lambda packed, shapes: packed

    comfy.conds = conds
    comfy.model_base = model_base
    comfy.utils = utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.conds"] = conds
    sys.modules["comfy.model_base"] = model_base
    sys.modules["comfy.utils"] = utils

    spec = importlib.util.spec_from_file_location(
        "payloadpkg.patch_payload", ROOT / "patch_payload.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, Cond


def test_merge_masks_idempotence_and_rewrap():
    class BuggyMiniMaxH3:
        def extra_conds(self, **kwargs):
            payload = {}
            keyframes = kwargs.get("minimax_keyframes")
            refs = kwargs.get("minimax_refs")
            if keyframes is not None:
                payload["cond_video_latents"] = [kf["latent"] for kf in keyframes]
            if refs is not None:
                payload["cond_video_latents"] = [r["latent"] for r in refs if "latent" in r]
                payload["cond_audio_latents"] = [r["audio_latent"] for r in refs if r.get("audio_latent") is not None]
            return {"minimax_payload": Cond(payload)}

    # Cond is looked up when method executes; fill it after module setup.
    module, Cond = load_module(BuggyMiniMaxH3)

    assert module.apply_patch(require_merge=True, require_av_masks=True)
    wrapped_once = BuggyMiniMaxH3.extra_conds
    assert module.apply_patch(require_merge=True, require_av_masks=True)
    assert BuggyMiniMaxH3.extra_conds is wrapped_once

    kf_v = torch.tensor([1.0])
    ref_v = torch.tensor([2.0])
    ref_a = torch.tensor([3.0])
    video_mask = torch.cat((torch.zeros(1), torch.ones(1))).reshape(1, 1, 2, 1, 1)
    audio_mask = torch.cat((torch.zeros(1), torch.ones(1))).reshape(1, 1, 1, 2)

    out = BuggyMiniMaxH3().extra_conds(
        minimax_keyframes=[{"latent": kf_v}],
        minimax_refs=[{"latent": ref_v}, {"audio_latent": ref_a}],
        minimax_frame_count=39,
        denoise_mask=(video_mask, audio_mask),
        latent_shapes=[video_mask.shape, audio_mask.shape],
    )
    payload = out["minimax_payload"].cond
    assert payload["cond_video_latents"] == [kf_v, ref_v]
    assert payload["cond_audio_latents"] == [ref_a]
    assert payload["frame_count"] == 39
    assert "denoise_mask" in out
    assert "audio_denoise_mask" in out

    # Simulate a destructive third-party replacement after our first install.
    def replacement(self, **kwargs):
        return {"minimax_payload": Cond({})}

    BuggyMiniMaxH3.extra_conds = replacement
    assert not module.is_applied()
    assert module.apply_patch(require_merge=True, require_av_masks=True)
    assert module.is_applied()
    assert BuggyMiniMaxH3.extra_conds is not replacement
    assert module.is_outermost()

    # A later well-behaved custom node can still wrap our method and then mutate
    # the result.  ensure/apply must reassert our post-processor on the outside.
    prior = BuggyMiniMaxH3.extra_conds

    @functools.wraps(prior)
    def outer_third_party(self, **kwargs):
        out = prior(self, **kwargs)
        out["minimax_payload"].cond["cond_video_latents"] = [torch.tensor([99.0])]
        return out

    BuggyMiniMaxH3.extra_conds = outer_third_party
    # functools.wraps copied our marker attribute, but this is not our code.
    assert module.is_applied()
    assert not module.is_outermost()
    assert module.apply_patch(require_merge=True, require_av_masks=True)
    assert module.is_outermost()

    out2 = BuggyMiniMaxH3().extra_conds(
        minimax_keyframes=[{"latent": kf_v}],
        minimax_refs=[{"latent": ref_v}, {"audio_latent": ref_a}],
        minimax_frame_count=39,
        denoise_mask=(video_mask, audio_mask),
        latent_shapes=[video_mask.shape, audio_mask.shape],
    )
    assert out2["minimax_payload"].cond["cond_video_latents"] == [kf_v, ref_v]


def test_native_capabilities_are_left_unwrapped():
    class NativeMiniMaxH3:
        def extra_conds(self, **kwargs):
            # Deliberately written in the source forms detected by Update 2.
            payload = {}
            keyframes = kwargs.get("minimax_keyframes") or []
            refs = kwargs.get("minimax_refs") or []
            kf_video = [kf["latent"] for kf in keyframes if kf.get("latent") is not None]
            ref_video = [ref["latent"] for ref in refs if ref.get("latent") is not None]
            payload["cond_video_latents"] = kf_video + ref_video
            denoise_mask = kwargs.get("denoise_mask")
            unpack_latents = denoise_mask
            out = {"minimax_payload": Cond(payload)}
            if unpack_latents is not None:
                out["denoise_mask"] = unpack_latents
                out["audio_denoise_mask"] = unpack_latents
            return out

    module, Cond = load_module(NativeMiniMaxH3)
    original = NativeMiniMaxH3.extra_conds
    status = module.capability_status()
    assert status["native_keyframe_ref_merge"]
    assert status["native_av_mask_payload"]
    assert module.apply_patch(require_merge=True, require_av_masks=True)
    assert NativeMiniMaxH3.extra_conds is original
    assert not module.is_applied()


def test_av_mask_payload_preserves_batch_dimension():
    class BuggyMiniMaxH3:
        def extra_conds(self, **kwargs):
            return {"minimax_payload": Cond({})}

    module, Cond = load_module(BuggyMiniMaxH3)
    assert module.apply_patch(require_merge=False, require_av_masks=True)

    video_mask = torch.ones(2, 1, 2, 1, 1)
    audio_mask = torch.ones(2, 1, 1, 2)
    video_mask[0, :, 0] = 0
    video_mask[1, :, 1] = 0
    audio_mask[0, :, :, 0] = 0
    audio_mask[1, :, :, 1] = 0

    out = BuggyMiniMaxH3().extra_conds(
        denoise_mask=(video_mask, audio_mask),
        latent_shapes=[video_mask.shape, audio_mask.shape],
    )
    assert tuple(out["denoise_mask"].cond.shape) == (2, 1, 2, 1, 1)
    assert tuple(out["audio_denoise_mask"].cond.shape) == (2, 1, 1, 2)
