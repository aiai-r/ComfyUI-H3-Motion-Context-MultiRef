"""Capability orchestration for H3 Motion Context on current ComfyUI.

Classic Motion Context / MultiRef uses native ComfyUI guide support introduced
by PR #15439 and never monkey-patches H3 core.  The existing-video masked
extension is a separate feature: it still needs PR #15375-equivalent AV mask
support when the live ComfyUI build does not provide that support natively.
"""

from __future__ import annotations

import inspect
import logging

import torch

_LOG = logging.getLogger("h3_motion_context")
_LOGGED = set()


def _log_once(key, message, *args):
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    _LOG.info(message, *args)


def native_guide_status():
    """Probe the live H3 core for the #15439 capabilities we rely on.

    This is intentionally behavioral for PackedLayout: a guide is placed at a
    nonzero target frame while an ordinary Ref2VA image ref is present.  The
    guide's video and audio condition segments must both land relative to the
    *post-ref target origin*.
    """
    import comfy.ldm.minimax.model as mm
    import comfy.model_base as model_base

    out = {
        "packed_layout_available": False,
        "ref_aware_arbitrary_guides": False,
        "guide_audio_segment": False,
        "native_keyframe_ref_merge": False,
        "native_keyframe_ref_audio_merge": False,
    }

    layout_cls = getattr(mm, "PackedLayout", None)
    if layout_cls is not None:
        out["packed_layout_available"] = True
        try:
            kf = {
                "resolved_frame_index": 3,
                "latent": torch.zeros((1, 24, 1, 2, 2)),
                "audio_latent": torch.zeros((1, 32, 2, 2)),
            }
            refs = [{"kind": "image", "latent_h": 2, "latent_w": 2}]
            layout = layout_cls(7, 7, 2, 2, 16, keyframes=[kf], refs=refs)
            cond_spans = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
            aud_spans = [(a, b) for a, b, kind in layout.segments if kind == "cond_audio"]
            # One image ref advances target origin by exactly 1.0.
            expected = 8.0 + float(mm.FRAME_RESCALE) * 3.0
            if cond_spans:
                a, b = cond_spans[0]
                vals = layout.position_ids[a:b, 0]
                out["ref_aware_arbitrary_guides"] = bool(torch.allclose(
                    vals, torch.full_like(vals, expected), atol=1e-9, rtol=0.0))
            if aud_spans:
                a, b = aud_spans[0]
                vals = layout.position_ids[a:b, 0]
                # Stereo audio rows begin at the guide's cond_t.
                rt = 2
                out["guide_audio_segment"] = bool(
                    torch.allclose(vals[:rt], torch.tensor([expected, expected + 1.0], dtype=vals.dtype), atol=1e-9, rtol=0.0)
                )
        except Exception as exc:
            out["packed_layout_error"] = repr(exc)

    cls = getattr(model_base, "MiniMaxH3", None)
    fn = getattr(cls, "extra_conds", None) if cls is not None else None
    if fn is not None:
        try:
            src = " ".join(inspect.getsource(fn).split())
        except (OSError, TypeError):
            src = ""
        out["native_keyframe_ref_merge"] = (
            'payload.get("cond_video_latents", []) +' in src
            or "payload.get('cond_video_latents', []) +" in src
        )
        out["native_keyframe_ref_audio_merge"] = (
            'payload.get("cond_audio_latents", []) +' in src
            or "payload.get('cond_audio_latents', []) +" in src
        )
    return out


def ensure_motion_context_compat():
    """Require native #15439 guide support; never patch H3 core."""
    status = native_guide_status()
    required = (
        status.get("ref_aware_arbitrary_guides"),
        status.get("guide_audio_segment"),
        status.get("native_keyframe_ref_merge"),
        status.get("native_keyframe_ref_audio_merge"),
    )
    if not all(required):
        raise RuntimeError(
            "h3_motion_context: this build requires the native MiniMax H3 "
            "guide/MultiRef support from ComfyUI PR #15439. Capability report: %r"
            % status
        )
    _log_once(
        "native_15439",
        "h3_motion_context: native ComfyUI H3 guide + MultiRef core detected; "
        "no Motion Context core patch installed",
    )
    return True


def ensure_existing_video_compat():
    """Enable #15375-equivalent support only for the masked extension node."""
    from .h3_mask_compat import ensure_h3_mask_compat
    from .h3_mask_payload_compat import ensure_av_mask_payload_compat

    ensure_h3_mask_compat()
    ensure_av_mask_payload_compat()
    _log_once(
        "existing_video_ready",
        "h3_motion_context: existing-video H3 AV-mask support ready "
        "(native #15375 capabilities are preferred automatically)",
    )
    return True


def ensure_hybrid_compat():
    """Research helper: native guide support plus optional masked extension."""
    ensure_motion_context_compat()
    ensure_existing_video_compat()
    return True


def capability_report():
    from .h3_mask_compat import capability_status as mask_capability_status
    from .h3_mask_payload_compat import capability_status as mask_payload_status
    return {
        "native_guide": native_guide_status(),
        "mask": mask_capability_status(),
        "mask_payload": mask_payload_status(),
    }
