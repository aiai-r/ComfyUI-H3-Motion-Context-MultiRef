"""Regression coverage for Issue #7: behavioral #15439 detection and conditional ref requirements."""

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


class _CondConstant:
    def __init__(self, cond):
        self.cond = cond


def _install_compat(extra_conds_impl):
    for name in list(sys.modules):
        if name == "compatpkg" or name.startswith("compatpkg.") or name == "comfy" or name.startswith("comfy."):
            sys.modules.pop(name, None)

    pkg = types.ModuleType("compatpkg")
    pkg.__path__ = [str(ROOT)]
    sys.modules["compatpkg"] = pkg

    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    minimax = types.ModuleType("comfy.ldm.minimax")
    mm = types.ModuleType("comfy.ldm.minimax.model")
    model_base = types.ModuleType("comfy.model_base")
    conds = types.ModuleType("comfy.conds")
    conds.CONDConstant = _CondConstant

    class PackedLayout:
        def __init__(self, text_t, video_t, h, w, audio_t, keyframes=None, refs=None):
            expected = 8.0 + (5.0 / 3.0) * 3.0
            self.segments = [(0, 1, "cond"), (1, 3, "cond_audio")]
            self.position_ids = torch.tensor([[expected], [expected], [expected + 1.0]], dtype=torch.float64)

    mm.PackedLayout = PackedLayout
    mm.FRAME_RESCALE = 5.0 / 3.0

    class MiniMaxH3:
        extra_conds = extra_conds_impl

    model_base.MiniMaxH3 = MiniMaxH3

    comfy.ldm = ldm
    comfy.conds = conds
    ldm.minimax = minimax
    minimax.model = mm
    sys.modules["comfy"] = comfy
    sys.modules["comfy.ldm"] = ldm
    sys.modules["comfy.ldm.minimax"] = minimax
    sys.modules["comfy.ldm.minimax.model"] = mm
    sys.modules["comfy.model_base"] = model_base
    sys.modules["comfy.conds"] = conds

    spec = importlib.util.spec_from_file_location("compatpkg.h3_compat", ROOT / "h3_compat.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _functional_merge(self, **kwargs):
    # Deliberately written unlike upstream's current source expression. This
    # would have fooled the old inspect.getsource substring detector.
    payload = {}
    keyframes = kwargs.get("minimax_keyframes") or []
    refs = kwargs.get("minimax_refs") or []
    videos = [x["latent"] for x in keyframes if x.get("latent") is not None]
    audios = [x["audio_latent"] for x in keyframes if x.get("audio_latent") is not None]
    videos.extend(x["latent"] for x in refs if x.get("latent") is not None)
    audios.extend(x["audio_latent"] for x in refs if x.get("audio_latent") is not None)
    payload["cond_video_latents"] = videos
    payload["cond_audio_latents"] = audios
    return {"minimax_payload": _CondConstant(payload)}


def _broken_ref_overwrite(self, **kwargs):
    refs = kwargs.get("minimax_refs") or []
    payload = {
        "cond_video_latents": [x["latent"] for x in refs if x.get("latent") is not None],
        "cond_audio_latents": [x["audio_latent"] for x in refs if x.get("audio_latent") is not None],
    }
    return {"minimax_payload": _CondConstant(payload)}


def test_behavioral_merge_probe_accepts_functional_refactor():
    compat = _install_compat(_functional_merge)
    status = compat.native_guide_status()
    assert status["ref_aware_arbitrary_guides"] is True
    assert status["guide_audio_segment"] is True
    assert status["native_keyframe_ref_merge"] is True
    assert status["native_keyframe_ref_audio_merge"] is True


def test_behavioral_merge_probe_rejects_overwrite_semantics():
    compat = _install_compat(_broken_ref_overwrite)
    status = compat.native_guide_status()
    assert status["native_keyframe_ref_merge"] is False
    assert status["native_keyframe_ref_audio_merge"] is False


def test_simple_motion_context_does_not_require_unreflected_ref_merge():
    compat = _install_compat(_functional_merge)
    compat.native_guide_status = lambda: {
        "ref_aware_arbitrary_guides": True,
        "guide_audio_segment": True,
        "native_keyframe_ref_merge": False,
        "native_keyframe_ref_audio_merge": False,
    }
    assert compat.ensure_motion_context_compat([[None, {}]]) is True


def test_image_ref_requires_only_video_merge():
    compat = _install_compat(_functional_merge)
    compat.native_guide_status = lambda: {
        "ref_aware_arbitrary_guides": True,
        "guide_audio_segment": True,
        "native_keyframe_ref_merge": True,
        "native_keyframe_ref_audio_merge": False,
    }
    conditioning = [[None, {"minimax_refs": [{"kind": "image", "latent": torch.zeros(1)}]}]]
    assert compat.ensure_motion_context_compat(conditioning) is True


def test_image_ref_fails_if_video_merge_is_missing():
    compat = _install_compat(_functional_merge)
    compat.native_guide_status = lambda: {
        "ref_aware_arbitrary_guides": True,
        "guide_audio_segment": True,
        "native_keyframe_ref_merge": False,
        "native_keyframe_ref_audio_merge": True,
    }
    conditioning = [[None, {"minimax_refs": [{"kind": "image", "latent": torch.zeros(1)}]}]]
    try:
        compat.ensure_motion_context_compat(conditioning)
    except RuntimeError as exc:
        assert "native_keyframe_ref_merge" in str(exc)
        assert "video=True" in str(exc)
    else:
        raise AssertionError("expected missing video-ref merge to fail")


def test_audio_ref_requires_audio_merge():
    compat = _install_compat(_functional_merge)
    compat.native_guide_status = lambda: {
        "ref_aware_arbitrary_guides": True,
        "guide_audio_segment": True,
        "native_keyframe_ref_merge": True,
        "native_keyframe_ref_audio_merge": False,
    }
    conditioning = [[None, {"minimax_refs": [{"kind": "audio", "audio_latent": torch.zeros(1)}]}]]
    try:
        compat.ensure_motion_context_compat(conditioning)
    except RuntimeError as exc:
        assert "native_keyframe_ref_audio_merge" in str(exc)
        assert "audio=True" in str(exc)
    else:
        raise AssertionError("expected missing audio-ref merge to fail")
