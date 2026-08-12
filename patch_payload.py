"""Capability-aware MiniMax H3 payload compatibility.

This module is the *only* file in this node pack allowed to wrap
``MiniMaxH3.extra_conds``.

It provides two independent compatibility features when current ComfyUI does
not already provide them natively:

1. Motion Context / MultiRef payload composition: keyframe video latents and
   ordinary Ref2VA latents must coexist instead of the refs branch overwriting
   the keyframe list.
2. MiniMax H3 AV denoise-mask payload extraction from ComfyUI PR #15375:
   unpack a nested video/audio mask into the two model conditions consumed by
   the H3 diffusion model.

The wrapper always calls the live upstream implementation first and only
post-processes its returned conditions. It therefore inherits future ComfyUI
changes instead of copying ``extra_conds`` wholesale.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Iterable

import torch

import comfy.conds
import comfy.model_base as model_base
import comfy.utils as utils

_LOG = logging.getLogger("h3_motion_context")
_MARKER = "_h3_motion_context_payload_compat_v2"
_REQUIRED: set[str] = set()
_ACTIVE: set[str] = set()


def _walk_wrapped(fn):
    """Yield a wrapper chain without looping on malformed __wrapped__ links."""
    seen = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        yield fn
        fn = getattr(fn, "__wrapped__", None)


def _is_our_wrapper(fn) -> bool:
    """Identify our actual wrapper, not a third-party wrapper that copied attrs.

    ``functools.wraps`` copies the wrapped function's ``__dict__`` by default,
    so a marker attribute alone can leak onto an outer wrapper.  The code object
    still points at this module, which makes the pair a much stronger identity.
    """
    code = getattr(fn, "__code__", None)
    return bool(
        getattr(fn, _MARKER, False)
        and code is not None
        and code.co_name == "wrapper"
        and code.co_filename == __file__
    )


def _contains_our_patch(fn) -> bool:
    return any(_is_our_wrapper(item) for item in _walk_wrapped(fn))


def _outermost_is_ours(fn) -> bool:
    return _is_our_wrapper(fn)


def _source_chain(fn) -> Iterable[str]:
    for item in _walk_wrapped(fn):
        if _is_our_wrapper(item):
            continue
        try:
            yield inspect.getsource(item)
        except (OSError, TypeError):
            continue


def _native_av_mask_payload(fn) -> bool:
    """Detect #15375-equivalent mask extraction in the live upstream chain."""
    for src in _source_chain(fn):
        if (
            "audio_denoise_mask" in src
            and "denoise_mask" in src
            and "unpack_latents" in src
        ):
            return True
    return False


def _native_keyframe_ref_merge(fn) -> bool:
    """Best-effort structural detection of a fixed keyframe/ref video merge.

    The current buggy implementation assigns cond_video_latents separately in
    the keyframe and ref branches. A future fixed implementation is considered
    native if its source clearly appends/concatenates an existing list or builds
    one expression from both keyframes and refs. If source inspection is not
    available we conservatively keep our idempotent post-processor enabled.
    """
    for src in _source_chain(fn):
        compact = " ".join(src.split())
        if "cond_video_latents" not in compact:
            continue
        # Known/likely fixed forms: existing-list + refs, +=, or explicit
        # keyframe+ref construction in one expression.
        if (
            'payload.get("cond_video_latents", []) +' in compact
            or "payload.get('cond_video_latents', []) +" in compact
            or 'payload["cond_video_latents"] +=' in compact
            or "payload['cond_video_latents'] +=" in compact
        ):
            return True
        if (
            "keyframes" in compact
            and "refs" in compact
            and "kf_video" in compact
            and "ref_video" in compact
            and "kf_video + ref_video" in compact
        ):
            return True
    return False


def capability_status():
    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        return {
            "available": False,
            "native_keyframe_ref_merge": False,
            "native_av_mask_payload": False,
            "wrapper_present": False,
        }
    fn = cls.extra_conds
    return {
        "available": True,
        "native_keyframe_ref_merge": _native_keyframe_ref_merge(fn),
        "native_av_mask_payload": _native_av_mask_payload(fn),
        "wrapper_present": _contains_our_patch(fn),
    }


def _payload_from_output(out):
    cond = out.get("minimax_payload") if isinstance(out, dict) else None
    payload = getattr(cond, "cond", None) if cond is not None else None
    return payload if isinstance(payload, dict) else None


def _repair_keyframe_ref_merge(out, kwargs):
    keyframes = kwargs.get("minimax_keyframes") or []
    refs = kwargs.get("minimax_refs") or []
    if not keyframes or not refs:
        return

    payload = _payload_from_output(out)
    if payload is None:
        raise RuntimeError(
            "h3_motion_context: MiniMaxH3.extra_conds returned no accessible "
            "minimax_payload while keyframes and refs are both active. "
            "Ref/Motion Context conditioning cannot be composed safely."
        )

    kf_video = [kf["latent"] for kf in keyframes if kf.get("latent") is not None]
    ref_video = [ref["latent"] for ref in refs if ref.get("latent") is not None]
    payload["cond_video_latents"] = kf_video + ref_video

    # Future-safe: current Motion Context audio lives in refs, but native H3 may
    # eventually allow audio-bearing keyframes/guides. Preserve both in layout
    # order instead of erasing one source.
    kf_audio = [
        kf["audio_latent"]
        for kf in keyframes
        if kf.get("audio_latent") is not None
    ]
    ref_audio = [
        ref["audio_latent"]
        for ref in refs
        if ref.get("audio_latent") is not None
    ]
    if kf_audio or ref_audio:
        payload["cond_audio_latents"] = kf_audio + ref_audio

    frame_count = kwargs.get("minimax_frame_count")
    if frame_count is not None:
        payload["frame_count"] = frame_count


def _add_av_mask_conditions(out, kwargs):
    # Native #15375-equivalent ComfyUI already owns these keys. Never replace
    # native conditions when they are present.
    if not isinstance(out, dict):
        return
    have_video = "denoise_mask" in out
    have_audio = "audio_denoise_mask" in out
    if have_video and have_audio:
        return

    denoise_mask = kwargs.get("denoise_mask")
    latent_shapes = kwargs.get("latent_shapes")
    if denoise_mask is None or latent_shapes is None or len(latent_shapes) < 2:
        return

    masks = utils.unpack_latents(denoise_mask, latent_shapes)
    if len(masks) < 2:
        return

    if not have_video and torch.amin(masks[0]).item() < 1.0 - 1e-3:
        out["denoise_mask"] = comfy.conds.CONDRegular(
            masks[0][:, :1].clone()
        )
    if not have_audio and torch.amin(masks[1]).item() < 1.0 - 1e-3:
        out["audio_denoise_mask"] = comfy.conds.CONDRegular(
            masks[1][:, :1].clone()
        )


def _make_wrapper(base):
    @functools.wraps(base, updated=())
    def wrapper(self, **kwargs):
        out = base(self, **kwargs)
        if "merge" in _ACTIVE:
            _repair_keyframe_ref_merge(out, kwargs)
        if "av_masks" in _ACTIVE:
            _add_av_mask_conditions(out, kwargs)
        return out

    setattr(wrapper, _MARKER, True)
    wrapper._h3_motion_context_payload_base = base
    return wrapper


def apply_patch(*, require_merge=True, require_av_masks=False):
    """Enable only compatibility features missing from the live ComfyUI build.

    Repeated calls are safe. If another custom node destructively replaces
    ``MiniMaxH3.extra_conds`` after us, a later call wraps the new live method
    instead of trusting stale module-local state.
    """
    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        _LOG.warning("h3_motion_context: MiniMaxH3.extra_conds not found")
        return False

    current = cls.extra_conds
    native_merge = _native_keyframe_ref_merge(current)
    native_masks = _native_av_mask_payload(current)

    # Record what this process has actually requested, then recompute which of
    # those features are still missing from the *live* implementation.  This
    # makes the post-processor go dormant if a hot-swapped implementation gains
    # native support; after a normal ComfyUI restart no wrapper is installed at
    # all when everything is native.
    if require_merge:
        _REQUIRED.add("merge")
    if require_av_masks:
        _REQUIRED.add("av_masks")

    wanted = set()
    if "merge" in _REQUIRED and not native_merge:
        wanted.add("merge")
    if "av_masks" in _REQUIRED and not native_masks:
        wanted.add("av_masks")
    _ACTIVE.clear()
    _ACTIVE.update(wanted)

    if not _ACTIVE:
        return True

    # Our compatibility must be the final post-processor.  Merely finding an
    # older copy deeper in the wrapper chain is not enough: an outer custom-node
    # wrapper could mutate the payload after it.  Re-wrap the live top-level
    # method when necessary; the repair itself is idempotent.
    if _outermost_is_ours(current):
        return True

    had_inner = _contains_our_patch(current)
    cls.extra_conds = _make_wrapper(current)
    features = ", ".join(sorted(_ACTIVE))
    if had_inner:
        _LOG.warning(
            "h3_motion_context: MiniMaxH3.extra_conds was wrapped after our "
            "compatibility layer; reasserted it as the final post-processor (%s)",
            features,
        )
    else:
        _LOG.info("h3_motion_context: payload compatibility enabled (%s)", features)
    return True


def is_applied():
    cls = getattr(model_base, "MiniMaxH3", None)
    return bool(
        cls and hasattr(cls, "extra_conds") and _contains_our_patch(cls.extra_conds)
    )


def is_outermost():
    cls = getattr(model_base, "MiniMaxH3", None)
    return bool(
        cls and hasattr(cls, "extra_conds") and _outermost_is_ours(cls.extra_conds)
    )
