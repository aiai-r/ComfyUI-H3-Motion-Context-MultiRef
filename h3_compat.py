"""Lazy compatibility orchestration for H3 Motion Context Update 2."""

from __future__ import annotations

import logging

from .patch_layout import apply_patch as apply_layout_patch
from .patch_payload import apply_patch as apply_payload_patch
from .patch_payload import capability_status as payload_capability_status
from .h3_mask_compat import ensure_h3_mask_compat
from .h3_mask_compat import capability_status as mask_capability_status

_LOG = logging.getLogger("h3_motion_context")
_LOGGED = set()


def _log_once(key, message, *args):
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    _LOG.info(message, *args)


def ensure_motion_context_compat():
    """Enable only compatibility needed by classic Motion Context/MultiRef."""
    if not apply_layout_patch():
        raise RuntimeError(
            "h3_motion_context: could not enable Motion Context timeline/layout "
            "compatibility. Check the ComfyUI console self-test error."
        )
    if not apply_payload_patch(require_merge=True, require_av_masks=False):
        raise RuntimeError(
            "h3_motion_context: could not enable keyframe/ref payload composition."
        )
    return True


def ensure_existing_video_compat():
    """Enable only compatibility needed by the existing-MP4 masked extension."""
    ensure_h3_mask_compat()
    if not apply_payload_patch(require_merge=False, require_av_masks=True):
        raise RuntimeError(
            "h3_motion_context: could not enable H3 AV-mask payload compatibility."
        )
    _log_once(
        "existing_video_ready",
        "h3_motion_context: existing-video H3 AV-mask support ready "
        "(native capabilities are preferred automatically)",
    )
    return True


def ensure_hybrid_compat():
    """Enable both families for research graphs combining the mechanisms."""
    ensure_motion_context_compat()
    ensure_existing_video_compat()
    return True


def capability_report():
    """Return a diagnostic snapshot without applying new compatibility."""
    return {
        "payload": payload_capability_status(),
        "mask": mask_capability_status(),
    }
