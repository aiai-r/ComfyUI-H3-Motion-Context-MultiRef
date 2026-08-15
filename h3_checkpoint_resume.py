"""Crash-safe MiniMax H3 clip checkpoints, lazy resume, and RAM-safe assembly.

The music-video workflow can save every finished H3 joint AV latent to a fixed
clip slot.  A lazy resume node can then load the previous finished clip from
disk without evaluating the upstream generation tree, decode only the visual
tail required by the next H3 continuation, and continue from there.

Final assembly intentionally never constructs the complete movie as a ComfyUI
IMAGE batch.  Saved clip latents are loaded and VAE-decoded one at a time; the
same linear overlap used by KJNodes ImageBatchExtendWithOverlap is applied in
float space and completed frames are streamed directly to ffmpeg.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import comfy.model_management as _model_management
except (ImportError, ModuleNotFoundError, AttributeError):
    _model_management = None

try:
    import comfy.nested_tensor as _nested_tensor
except (ImportError, ModuleNotFoundError, AttributeError):
    _nested_tensor = None

import folder_paths

try:
    from comfy_api.latest import io as _comfy_io, ui as _comfy_ui
except (ImportError, ModuleNotFoundError, AttributeError):
    _comfy_io = _comfy_ui = None

try:
    from safetensors import safe_open as _st_safe_open
    from safetensors.torch import load_file as _st_load, save_file as _st_save
except ImportError:  # ComfyUI normally ships safetensors.
    _st_safe_open = _st_load = _st_save = None

_LOG = logging.getLogger("h3_motion_context.checkpoint")

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _pixel_frames(latent_t: int) -> int:
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


def _streams_from_latent(latent) -> Tuple[torch.Tensor, torch.Tensor]:
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3_checkpoint: expected a MiniMax H3 joint AV latent, got %r"
            % type(samples)
        )
    if len(parts) < 2:
        raise ValueError("h3_checkpoint: H3 latent is missing its audio stream")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "h3_checkpoint: video latent must be [B,C,T,H,W], got %s"
            % (tuple(video.shape),)
        )
    if audio.ndim != 4:
        raise ValueError(
            "h3_checkpoint: audio latent must be [B,C,2,T], got %s"
            % (tuple(audio.shape),)
        )
    return video, audio


def _checkpoint_save_path(filename_prefix: str, clip_index: int) -> str:
    prefix = (filename_prefix or "").strip().strip('"').strip("'")
    if not prefix:
        prefix = "h3_checkpoints/clip"
    folder, filename, _counter, _subfolder, _prefix = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory()
    )
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "%s_%05d.safetensors" % (filename, int(clip_index)))


def _resolve_checkpoint_path(path_or_prefix: str, clip_index: int) -> str:
    """Resolve a checkpoint file, directory, or prefix to one fixed clip slot."""
    p = (path_or_prefix or "").strip().strip('"').strip("'")
    if not p:
        p = "h3_checkpoints/clip"
    idx = int(clip_index)
    if idx < 1:
        raise ValueError("h3_checkpoint: clip_index must be >= 1")

    roots = [p]
    if not os.path.isabs(p):
        roots.append(os.path.join(folder_paths.get_output_directory(), p))

    tried = []
    for root in roots:
        tried.append(root)
        if os.path.isfile(root):
            return root
        if os.path.isdir(root):
            endings = (
                "_%05d.safetensors" % idx,
                "_clip%03d.safetensors" % idx,  # older repo naming
            )
            files = [
                os.path.join(root, name)
                for name in os.listdir(root)
                if name.endswith(endings)
            ]
            if files:
                return max(files, key=os.path.getmtime)
        candidate = "%s_%05d.safetensors" % (root, idx)
        tried.append(candidate)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "h3_checkpoint: no checkpoint found for clip %d. Tried: %s"
        % (idx, ", ".join(tried))
    )


def _checkpoint_disk_token(path_or_prefix: str, clip_index: int) -> str:
    """Stable cache token for a checkpoint that this node reads from disk.

    Live graph operation should be invalidated by normal upstream dependencies,
    not by an always-changing token. Resume mode is different: the fixed slot can
    be atomically overwritten between queues without any graph input changing, so
    include file identity/mtime only while a node is actually reading that slot.
    """
    try:
        path = _resolve_checkpoint_path(path_or_prefix, int(clip_index))
        st = os.stat(path)
        return "checkpoint:%s:%d:%d" % (path, int(st.st_mtime_ns), int(st.st_size))
    except Exception:
        return "checkpoint-missing:%s:%d" % (str(path_or_prefix), int(clip_index))



def _canonical_signature_value(value):
    """Convert prompt constants into deterministic JSON-safe signature data."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return value
    if isinstance(value, dict):
        return {str(k): _canonical_signature_value(value[k]) for k in sorted(value, key=lambda x: str(x))}
    if isinstance(value, (list, tuple)):
        return [_canonical_signature_value(v) for v in value]
    return {"type": type(value).__name__, "repr": repr(value)}


def _prompt_node(prompt, node_id):
    if not isinstance(prompt, dict):
        return None
    return prompt.get(str(node_id), prompt.get(node_id))


def _is_prompt_link(prompt, value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], int)
        and _prompt_node(prompt, value[0]) is not None
    )


def _loader_file_token(class_type: str, input_name: str, value):
    """Add inexpensive identity for user input media without hashing model files."""
    if not isinstance(value, str):
        return None
    if class_type not in {"LoadImage", "LoadAudio", "VHS_LoadVideo", "VHS_LoadAudio", "VHS_LoadImages"}:
        return None
    if input_name not in {"image", "audio", "video", "directory", "path"}:
        return None
    candidates = []
    try:
        get_annotated = getattr(folder_paths, "get_annotated_filepath", None)
        if get_annotated is not None:
            candidates.append(get_annotated(value))
    except Exception:
        pass
    try:
        inp = getattr(folder_paths, "get_input_directory", lambda: None)()
        if inp:
            candidates.append(os.path.join(inp, value))
    except Exception:
        pass
    candidates.append(value)
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            try:
                st = os.stat(candidate)
                return {"path": os.path.abspath(candidate), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
            except OSError:
                continue
    return {"name": value}


def _generation_signature_from_link(prompt, link):
    """Hash the complete submitted ancestor graph feeding one generation result.

    This is deliberately independent of ComfyUI's RAM cache.  It represents the
    actual API-prompt topology/constants that can affect a clip, so it remains
    usable after cache eviction or a process restart.
    """
    memo = {}
    visiting = set()

    def sig_node(node_id):
        key = str(node_id)
        if key in memo:
            return memo[key]
        if key in visiting:
            return "cycle:" + key
        node = _prompt_node(prompt, node_id)
        if node is None:
            return "missing:" + key
        visiting.add(key)
        class_type = str(node.get("class_type", ""))
        payload = {"class_type": class_type, "inputs": {}}
        inputs = node.get("inputs", {}) or {}
        for name in sorted(inputs):
            value = inputs[name]
            if _is_prompt_link(prompt, value):
                payload["inputs"][name] = {
                    "link_output": int(value[1]),
                    "upstream": sig_node(value[0]),
                }
            else:
                item = {"value": _canonical_signature_value(value)}
                file_token = _loader_file_token(class_type, str(name), value)
                if file_token is not None:
                    item["file"] = file_token
                payload["inputs"][name] = item
        visiting.remove(key)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        memo[key] = digest
        return digest

    if not _is_prompt_link(prompt, link):
        payload = {"unlinked": _canonical_signature_value(link)}
    else:
        payload = {"source": sig_node(link[0]), "output": int(link[1])}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _gate_generation_signature(prompt, unique_id, input_name="live_latent"):
    node = _prompt_node(prompt, unique_id)
    if node is None:
        return None
    link = (node.get("inputs", {}) or {}).get(input_name)
    if link is None:
        return None
    return _generation_signature_from_link(prompt, link)


def _checkpoint_metadata(path: str) -> dict:
    if _st_safe_open is None or not os.path.isfile(path):
        return {}
    try:
        with _st_safe_open(path, framework="pt", device="cpu") as f:
            return dict(f.metadata() or {})
    except Exception:
        return {}


def _checkpoint_signature_matches(path: str, signature: Optional[str]) -> bool:
    if not signature or not os.path.isfile(path):
        return False
    return _checkpoint_metadata(path).get("generation_signature") == signature

def _load_checkpoint(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if _st_load is None:
        raise RuntimeError("h3_checkpoint: safetensors is unavailable")
    data = _st_load(path, device="cpu")
    if "video" not in data or "audio" not in data:
        raise ValueError(
            "h3_checkpoint: %s is not an H3 AV checkpoint (video/audio missing)"
            % path
        )
    video, audio = data["video"], data["audio"]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    return video, audio


def _load_checkpoint_video(path: str) -> torch.Tensor:
    if _st_safe_open is None:
        return _load_checkpoint(path)[0]
    with _st_safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        if "video" not in keys:
            raise ValueError(
                "h3_checkpoint: %s is not an H3 AV checkpoint (video missing)"
                % path
            )
        video = f.get_tensor("video")
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "h3_checkpoint: saved video latent must be [B,C,T,H,W], got %s"
            % (tuple(video.shape),)
        )
    return video


def _load_checkpoint_audio(path: str) -> torch.Tensor:
    if _st_safe_open is None:
        return _load_checkpoint(path)[1]
    with _st_safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        if "audio" not in keys:
            raise ValueError(
                "h3_checkpoint: %s is not an H3 AV checkpoint (audio missing)"
                % path
            )
        audio = f.get_tensor("audio")
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError(
            "h3_checkpoint: saved audio latent must be [B,C,2,T], got %s"
            % (tuple(audio.shape),)
        )
    return audio


def _decoded_audio(audio_vae, audio_latent: torch.Tensor) -> dict:
    waveform = audio_vae.decode(audio_latent).movedim(-1, 1)
    std = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    waveform = waveform / std
    sr = int(
        getattr(
            audio_vae, "audio_sample_rate_output",
            getattr(audio_vae, "audio_sample_rate", 44100),
        )
    )
    return {"waveform": waveform.detach().cpu(), "sample_rate": sr}


def _append_f32_audio(fileobj, waveform: torch.Tensor) -> int:
    x = waveform.detach().cpu()
    if getattr(x, "ndim", 0) != 3 or int(x.shape[0]) < 1:
        raise ValueError(
            "h3_checkpoint: audio waveform must be [B,C,L], got %s"
            % (tuple(getattr(x, "shape", ())),)
        )
    x = x[:1]
    channels = int(x.shape[1])
    if channels == 1:
        x = x.repeat(1, 2, 1)
        channels = 2
    elif channels > 2:
        x = x[:, :2]
        channels = 2
    interleaved = x[0].transpose(0, 1).contiguous().numpy().astype("<f4", copy=False)
    fileobj.write(interleaved.tobytes(order="C"))
    return int(x.shape[-1])


def _decoded_images(vae, video_latent: torch.Tensor) -> torch.Tensor:
    images = vae.decode(video_latent)
    if getattr(images, "ndim", 0) == 5 and int(images.shape[0]) == 1:
        images = images[0]
    if getattr(images, "ndim", 0) != 4:
        raise ValueError(
            "h3_checkpoint: video VAE decode returned %s; expected 4-D frames"
            % (tuple(getattr(images, "shape", ())),)
        )
    # Stock Comfy IMAGE is [N,H,W,C]. Be tolerant of [N,C,H,W].
    if int(images.shape[-1]) in (3, 4):
        images = images[..., :3]
    elif int(images.shape[1]) in (3, 4):
        images = images.movedim(1, -1)[..., :3]
    else:
        raise ValueError(
            "h3_checkpoint: cannot infer RGB channel axis from decoded shape %s"
            % (tuple(images.shape),)
        )
    return images.detach().cpu().contiguous()


def _write_rgb24_frames(proc, images: torch.Tensor, chunk: int = 8) -> int:
    """Quantize only at the final video encode boundary and stream in chunks."""
    count = int(images.shape[0])
    for start in range(0, count, max(1, int(chunk))):
        part = images[start : start + chunk].detach().cpu().clamp(0.0, 1.0)
        arr = torch.round(part * 255.0).to(torch.uint8).numpy()
        proc.stdin.write(arr.tobytes(order="C"))
        del part, arr
    return count


def _write_f32_audio(path: str, audio: dict) -> Tuple[int, int]:
    waveform = audio["waveform"]
    sr = int(audio["sample_rate"])
    if getattr(waveform, "ndim", 0) != 3:
        raise ValueError(
            "h3_checkpoint: master_audio waveform must be [B,C,L], got %s"
            % (tuple(getattr(waveform, "shape", ())),)
        )
    x = waveform[:1].detach().cpu()
    channels = int(x.shape[1])
    if channels < 1:
        raise ValueError("h3_checkpoint: master_audio has no channels")
    if channels > 2:
        _LOG.warning(
            "h3_checkpoint: master audio has %d channels; using the first two for MP4 output",
            channels,
        )
        x = x[:, :2]
        channels = 2
    interleaved = x[0].transpose(0, 1).contiguous().numpy().astype("<f4", copy=False)
    interleaved.tofile(path)
    return sr, channels


def _ui_video_preview_entry(path: str, fps: float, media_format: str = "video/h264-mp4") -> dict:
    output_dir = os.path.abspath(folder_paths.get_output_directory())
    abs_path = os.path.abspath(path)
    try:
        rel = os.path.relpath(abs_path, output_dir)
    except ValueError:
        rel = os.path.basename(abs_path)
    rel = rel.replace("\\", "/")
    subfolder = os.path.dirname(rel).replace("\\", "/")
    if subfolder == ".":
        subfolder = ""
    filename = os.path.basename(abs_path)
    return {
        "filename": filename,
        "subfolder": subfolder,
        "type": "output",
        "format": media_format,
        "frame_rate": float(fps),
        "workflow": os.path.splitext(filename)[0] + ".png",
        "fullpath": abs_path,
    }


def _ui_video_preview(path: str, fps: float, media_format: str = "video/h264-mp4") -> dict:
    return {"gifs": [_ui_video_preview_entry(path, fps, media_format)]}


def _final_video_node_output(path: str, result: tuple, fps: float):
    """Return current ComfyUI's native video preview UI for an existing MP4.

    The MP4 is already on disk; PreviewVideo only publishes that saved result to
    the frontend and never decodes the complete movie back into an IMAGE batch.
    """
    if _comfy_io is not None and _comfy_ui is not None:
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        abs_path = os.path.abspath(path)
        try:
            rel = os.path.relpath(abs_path, output_dir).replace("\\", "/")
        except ValueError:
            rel = os.path.basename(abs_path)
        filename = os.path.basename(rel)
        subfolder = os.path.dirname(rel).replace("\\", "/")
        if subfolder == ".":
            subfolder = ""
        return _comfy_io.NodeOutput(
            *result,
            ui=_comfy_ui.PreviewVideo([
                _comfy_ui.SavedResult(filename, subfolder, _comfy_io.FolderType.output)
            ]),
        )
    # Compatibility fallback for older frontends.
    return {"ui": _ui_video_preview(path, fps), "result": result}


class MiniMaxH3CheckpointSave:
    """Atomically save a finished H3 sampler latent and pass it downstream."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Finished H3 sampler output. The node saves its joint video/audio streams and passes the latent through unchanged."
                }),
                "filename_prefix": ("STRING", {
                    "default": "h3_checkpoints/clip",
                    "tooltip": "Path prefix under ComfyUI/output. Fixed clip slots are written as prefix_00001.safetensors, prefix_00002.safetensors, ..."
                }),
                "clip_index": ("INT", {
                    "default": 1, "min": 1, "max": 9999,
                    "tooltip": "Fixed checkpoint slot for this clip. Re-running the same clip atomically replaces only its own slot."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "checkpoint_path")
    FUNCTION = "save"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Crash-safe per-clip H3 checkpoint. Saves the sampler's joint AV latent "
        "to a fixed safetensors slot, then passes the latent onward."
    )

    @classmethod
    def IS_CHANGED(cls, filename_prefix="h3_checkpoints/clip", clip_index=1, **kwargs):
        # Do not force a checkpoint writer to run on every queue. Normal ComfyUI
        # dependency tracking already invalidates this node when the sampler
        # latent changes. A stable fingerprint lets unchanged earlier clips stay
        # cached when only a later seed/prompt is edited.
        return "checkpoint-save:%s:%d" % (str(filename_prefix), int(clip_index))

    def save(self, latent, filename_prefix="h3_checkpoints/clip", clip_index=1):
        if _st_save is None:
            raise RuntimeError("h3_checkpoint: safetensors is unavailable")
        video, audio = _streams_from_latent(latent)
        path = _checkpoint_save_path(filename_prefix, int(clip_index))
        tmp = path + ".tmp"
        tensors = {
            "video": video.detach().cpu().contiguous(),
            "audio": audio.detach().cpu().contiguous(),
        }
        metadata = {
            "format": "h3_motion_context_checkpoint_v1",
            "clip_index": str(int(clip_index)),
            "video_frames": str(_pixel_frames(int(video.shape[2]))),
        }
        try:
            _st_save(tensors, tmp, metadata=metadata)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        _LOG.info(
            "h3_checkpoint: saved clip %d -> %s (video %s, audio %s)",
            int(clip_index), path, tuple(video.shape), tuple(audio.shape),
        )
        del tensors
        return (latent, path)


class MiniMaxH3CheckpointSavePath:
    """Persistent lazy generation gate backed by a fixed H3 checkpoint slot.

    The sampler latent is lazy.  Before asking ComfyUI to evaluate it, this node
    hashes the submitted ancestor graph and compares that signature with metadata
    stored inside the existing safetensors checkpoint.  An exact match returns the
    saved path immediately; a mismatch requests the sampler, atomically overwrites
    the slot, and records the new signature.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {
                    "default": "h3_checkpoints/clip",
                }),
                "clip_index": ("INT", {
                    "default": 1, "min": 1, "max": 9999,
                }),
            },
            "optional": {
                "latent": ("LATENT", {
                    "lazy": True,
                    "tooltip": "Finished H3 sampler output. Requested only when this clip's persistent generation signature changed or its checkpoint is missing.",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("checkpoint_path",)
    FUNCTION = "save"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Persistent disk-backed H3 generation gate. If the saved checkpoint's "
        "generation signature still matches this sampler's complete submitted "
        "ancestor graph, the sampler branch is not requested at all."
    )

    @classmethod
    def IS_CHANGED(cls, filename_prefix="h3_checkpoints/clip", clip_index=1, **kwargs):
        # Keep the gate itself cacheable. If its tiny path output is evicted,
        # check_lazy_status re-validates the checkpoint against its persistent
        # graph signature before deciding whether the expensive sampler is needed.
        return "persistent-checkpoint-gate:%s:%d:v2" % (str(filename_prefix), int(clip_index))

    def _state(self, filename_prefix, clip_index, prompt, unique_id):
        path = _checkpoint_save_path(filename_prefix, int(clip_index))
        signature = _gate_generation_signature(prompt, unique_id, "latent")
        return path, signature

    def check_lazy_status(
        self, filename_prefix="h3_checkpoints/clip", clip_index=1,
        latent=None, prompt=None, unique_id=None
    ):
        path, signature = self._state(filename_prefix, clip_index, prompt, unique_id)
        if _checkpoint_signature_matches(path, signature):
            _LOG.info(
                "h3_checkpoint: Clip %d signature unchanged; reuse %s without sampling",
                int(clip_index), path,
            )
            return []
        if latent is None:
            return ["latent"]
        return []

    def save(
        self, latent=None, filename_prefix="h3_checkpoints/clip", clip_index=1,
        prompt=None, unique_id=None
    ):
        if _st_save is None:
            raise RuntimeError("h3_checkpoint: safetensors is unavailable")
        path, signature = self._state(filename_prefix, clip_index, prompt, unique_id)

        if latent is None:
            if _checkpoint_signature_matches(path, signature):
                return (path,)
            raise RuntimeError(
                "h3_checkpoint: persistent gate for Clip %d needs generation because "
                "the checkpoint is missing/stale, but its lazy latent was not evaluated"
                % int(clip_index)
            )

        video, audio = _streams_from_latent(latent)
        tmp = path + ".tmp"
        tensors = {
            "video": video.detach().cpu().contiguous(),
            "audio": audio.detach().cpu().contiguous(),
        }
        metadata = {
            "format": "h3_motion_context_checkpoint_v2",
            "clip_index": str(int(clip_index)),
            "video_frames": str(_pixel_frames(int(video.shape[2]))),
            "generation_signature": str(signature or ""),
        }
        try:
            _st_save(tensors, tmp, metadata=metadata)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        _LOG.info(
            "h3_checkpoint: generated + saved Clip %d -> %s (signature %s)",
            int(clip_index), path, (signature or "")[:12],
        )
        del tensors
        return (path,)


class MiniMaxH3CheckpointLoadPath:
    """Load an exact checkpoint path emitted by CheckpointSavePath."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Exact path from H3 Checkpoint Save Path."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "load"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = "Load a joint H3 AV latent from an exact saved checkpoint path."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The linked SavePath ancestry fingerprints the generation settings.
        # If this large latent is evicted, re-executing only reloads disk; it
        # never forces the upstream sampler when the tiny SavePath output is cached.
        return "checkpoint-load-linked-path-v1"

    def load(self, checkpoint_path):
        path = _resolve_checkpoint_path(str(checkpoint_path), 1)
        video, audio = _load_checkpoint(path)
        global _nested_tensor
        if _nested_tensor is None:
            import comfy.nested_tensor as _runtime_nested_tensor
            _nested_tensor = _runtime_nested_tensor
        return ({"samples": _nested_tensor.NestedTensor((video, audio))},)


class MiniMaxH3CheckpointTailFrames:
    """Disk-backed previous-clip tail selector with explicit crash resume."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "resume_from_clip": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "next_clip_index": ("INT", {"default": 2, "min": 2, "max": 9999}),
                "checkpoint_path": ("STRING", {"default": "h3_checkpoints/clip"}),
                "context_length": ("INT", {"default": 39, "min": 1, "max": 9999}),
            },
            "optional": {
                "checkpoint_signal": ("STRING", {
                    "lazy": True, "forceInput": True,
                    "tooltip": "Exact previous-clip path from H3 Checkpoint Save Path. Normal operation depends only on this tiny cached string."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("tail_frames", "source")
    FUNCTION = "select"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Load the previous saved clip and decode only its tail. Normal chaining "
        "uses the previous SavePath output; explicit resume can resolve the fixed slot directly."
    )

    @classmethod
    def IS_CHANGED(cls, resume_from_clip=0, next_clip_index=2, checkpoint_path="h3_checkpoints/clip", **kwargs):
        if int(resume_from_clip) == int(next_clip_index):
            return _checkpoint_disk_token(checkpoint_path, int(next_clip_index) - 1)
        return "disk-tail:%d" % int(next_clip_index)

    def check_lazy_status(self, vae, resume_from_clip, next_clip_index, checkpoint_path, context_length, checkpoint_signal=None):
        if int(resume_from_clip) == int(next_clip_index):
            return []
        if checkpoint_signal is None:
            return ["checkpoint_signal"]
        return []

    def select(self, vae, resume_from_clip, next_clip_index, checkpoint_path="h3_checkpoints/clip", context_length=39, checkpoint_signal=None):
        next_idx = int(next_clip_index)
        prev_idx = next_idx - 1
        if int(resume_from_clip) == next_idx:
            path = _resolve_checkpoint_path(checkpoint_path, prev_idx)
            source = "resume checkpoint clip %d: %s" % (prev_idx, path)
        else:
            if not checkpoint_signal:
                raise ValueError("h3_checkpoint: Clip %d needs checkpoint_signal from saved Clip %d" % (next_idx, prev_idx))
            path = _resolve_checkpoint_path(str(checkpoint_signal), prev_idx)
            source = "saved clip %d: %s" % (prev_idx, path)
        video = _load_checkpoint_video(path)
        images = _decoded_images(vae, video)
        total = int(images.shape[0])
        want = min(max(1, int(context_length)), total)
        tail = images[-want:].detach().clone().contiguous()
        del images, video
        gc.collect()
        _LOG.info("h3_checkpoint: %s -> returned final %d frames for Clip %d", source, want, next_idx)
        return (tail, source)


class MiniMaxH3ResumeCheckpointLatent:
    """Disk-backed latent continuation selector with explicit crash resume."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resume_from_clip": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "next_clip_index": ("INT", {"default": 2, "min": 2, "max": 9999}),
                "checkpoint_path": ("STRING", {"default": "h3_extension_checkpoints/clip"}),
            },
            "optional": {
                "checkpoint_signal": ("STRING", {
                    "lazy": True, "forceInput": True,
                    "tooltip": "Exact previous-clip path from H3 Checkpoint Save Path."
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("source_latent", "source")
    FUNCTION = "select"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Load the previous generated H3 AV latent from its saved checkpoint. "
        "Only a tiny path dependency is cached between clips."
    )

    @classmethod
    def IS_CHANGED(cls, resume_from_clip=0, next_clip_index=2, checkpoint_path="h3_extension_checkpoints/clip", **kwargs):
        if int(resume_from_clip) == int(next_clip_index):
            return _checkpoint_disk_token(checkpoint_path, int(next_clip_index) - 1)
        return "disk-latent:%d" % int(next_clip_index)

    def check_lazy_status(self, resume_from_clip, next_clip_index, checkpoint_path, checkpoint_signal=None):
        if int(resume_from_clip) == int(next_clip_index):
            return []
        if checkpoint_signal is None:
            return ["checkpoint_signal"]
        return []

    def select(self, resume_from_clip, next_clip_index, checkpoint_path="h3_extension_checkpoints/clip", checkpoint_signal=None):
        next_idx = int(next_clip_index)
        prev_idx = next_idx - 1
        if int(resume_from_clip) == next_idx:
            path = _resolve_checkpoint_path(checkpoint_path, prev_idx)
            source = "resume checkpoint clip %d: %s" % (prev_idx, path)
        else:
            if not checkpoint_signal:
                raise ValueError("h3_checkpoint: Clip %d needs checkpoint_signal from saved Clip %d" % (next_idx, prev_idx))
            path = _resolve_checkpoint_path(str(checkpoint_signal), prev_idx)
            source = "saved clip %d: %s" % (prev_idx, path)
        video, audio = _load_checkpoint(path)
        global _nested_tensor
        if _nested_tensor is None:
            import comfy.nested_tensor as _runtime_nested_tensor
            _nested_tensor = _runtime_nested_tensor
        latent = {"samples": _nested_tensor.NestedTensor((video, audio))}
        _LOG.info("h3_checkpoint: Clip %d continues from %s", next_idx, source)
        return (latent, source)


class MiniMaxH3CheckpointLoad:
    """Load a checkpoint as a normal decodable H3 joint AV LATENT."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {
                    "default": "h3_checkpoints/clip",
                    "tooltip": "Checkpoint prefix, checkpoint directory, or exact .safetensors file."
                }),
                "clip_index": ("INT", {
                    "default": 1, "min": 1, "max": 9999,
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "resolved_path")
    FUNCTION = "load"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = "Load a saved H3 clip checkpoint as a normal decodable joint AV latent."

    @classmethod
    def IS_CHANGED(cls, checkpoint_path, clip_index=1):
        try:
            path = _resolve_checkpoint_path(checkpoint_path, clip_index)
            return "%s:%d" % (path, os.stat(path).st_mtime_ns)
        except Exception:
            return float("NaN")

    def load(self, checkpoint_path="h3_checkpoints/clip", clip_index=1):
        path = _resolve_checkpoint_path(checkpoint_path, int(clip_index))
        video, audio = _load_checkpoint(path)
        global _nested_tensor
        if _nested_tensor is None:
            import comfy.nested_tensor as _runtime_nested_tensor
            _nested_tensor = _runtime_nested_tensor
        latent = {"samples": _nested_tensor.NestedTensor((video, audio))}
        _LOG.info("h3_checkpoint: loaded clip %d <- %s", int(clip_index), path)
        return (latent, path)


class MiniMaxH3ResumeTailFrames:
    """Lazy live/checkpoint selector that emits only the previous visual tail.

    If resume_from_clip equals next_clip_index, the live_latent input is NOT
    evaluated. The node instead loads checkpoint next_clip_index-1 from disk,
    decodes it internally, clones only the requested tail frames, and releases
    the full decoded batch before returning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE used to decode the previous clip."
                }),
                "resume_from_clip": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "0/1 = normal run from Clip 1. To recover after a crash in Clip N, set this to N; the Clip N selector loads checkpoint N-1 and lazily skips the entire earlier generation tree."
                }),
                "next_clip_index": ("INT", {
                    "default": 2, "min": 2, "max": 9999,
                    "tooltip": "The continuation clip this selector feeds. Set once per node (2 for Clip 2, 3 for Clip 3, ...)."
                }),
                "checkpoint_path": ("STRING", {
                    "default": "h3_checkpoints/clip",
                    "tooltip": "Same checkpoint prefix used by H3 Checkpoint Save."
                }),
                "context_length": ("INT", {
                    "default": 39, "min": 1, "max": 9999,
                    "tooltip": "Number of decoded tail frames returned to the song/masked-video context node."
                }),
            },
            "optional": {
                "live_latent": ("LATENT", {
                    "lazy": True,
                    "tooltip": "Previous clip's live checkpointed sampler latent. This branch is not evaluated when resume_from_clip equals next_clip_index."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("tail_frames", "source")
    FUNCTION = "select"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Decode only the previous clip tail needed for continuation. In resume "
        "mode, lazy evaluation cuts the upstream graph and loads the previous "
        "finished clip from disk instead."
    )

    @classmethod
    def IS_CHANGED(
        cls, resume_from_clip=0, next_clip_index=2,
        checkpoint_path="h3_checkpoints/clip", **kwargs
    ):
        # In normal live mode, upstream graph dependencies determine whether this
        # node changed. Only resume-from-disk mode needs to watch the fixed slot
        # for an external/atomic checkpoint replacement.
        if int(resume_from_clip) == int(next_clip_index):
            return _checkpoint_disk_token(checkpoint_path, int(next_clip_index) - 1)
        return "live-tail:%d" % int(next_clip_index)

    def check_lazy_status(
        self,
        vae,
        resume_from_clip,
        next_clip_index,
        checkpoint_path,
        context_length,
        live_latent=None,
    ):
        if int(resume_from_clip) == int(next_clip_index):
            return []
        if live_latent is None:
            return ["live_latent"]
        return []

    def select(
        self,
        vae,
        resume_from_clip,
        next_clip_index,
        checkpoint_path="h3_checkpoints/clip",
        context_length=39,
        live_latent=None,
    ):
        next_idx = int(next_clip_index)
        resume_idx = int(resume_from_clip)
        if resume_idx == next_idx:
            prev_idx = next_idx - 1
            path = _resolve_checkpoint_path(checkpoint_path, prev_idx)
            video = _load_checkpoint_video(path)
            source = "checkpoint clip %d: %s" % (prev_idx, path)
            _LOG.info(
                "h3_checkpoint: RESUME Clip %d from saved Clip %d; live upstream branch skipped",
                next_idx, prev_idx,
            )
        else:
            if live_latent is None:
                raise ValueError(
                    "h3_checkpoint: Clip %d needs live_latent unless resume_from_clip is exactly %d"
                    % (next_idx, next_idx)
                )
            video, _audio = _streams_from_latent(live_latent)
            source = "live clip %d" % (next_idx - 1)

        images = _decoded_images(vae, video)
        total = int(images.shape[0])
        want = min(max(1, int(context_length)), total)
        # Clone the slice so it owns a small storage. Otherwise a view would keep
        # the full decoded clip allocation alive in Comfy's cache.
        tail = images[-want:].detach().clone().contiguous()
        del images, video
        gc.collect()
        _LOG.info(
            "h3_checkpoint: %s -> returned only final %d frames for Clip %d",
            source, want, next_idx,
        )
        return (tail, source)


class MiniMaxH3ResumeOrLiveLatent:
    """Lazy previous-clip selector for latent-space continuation chains.

    When resume_from_clip equals next_clip_index, the live branch is not
    evaluated and checkpoint next_clip_index-1 is loaded from disk. Otherwise
    the previous live checkpointed sampler latent is passed through.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resume_from_clip": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "0/1 = normal run. To resume at Clip N, set N; this selector loads saved Clip N-1 and skips its live upstream branch."
                }),
                "next_clip_index": ("INT", {
                    "default": 2, "min": 2, "max": 9999,
                }),
                "checkpoint_path": ("STRING", {
                    "default": "h3_extension_checkpoints/clip",
                    "tooltip": "Same prefix used by H3 Checkpoint Save."
                }),
            },
            "optional": {
                "live_latent": ("LATENT", {
                    "lazy": True,
                    "tooltip": "Previous clip's live checkpointed sampler latent. Skipped when resuming exactly at next_clip_index."
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("source_latent", "source")
    FUNCTION = "select"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Lazy live/checkpoint selector for latent-to-latent H3 continuation. "
        "Allows multi-clip masked-AV extension chains to resume at clip boundaries "
        "without decoding the previous clip."
    )

    @classmethod
    def IS_CHANGED(
        cls, resume_from_clip=0, next_clip_index=2,
        checkpoint_path="h3_extension_checkpoints/clip", **kwargs
    ):
        if int(resume_from_clip) == int(next_clip_index):
            return _checkpoint_disk_token(checkpoint_path, int(next_clip_index) - 1)
        return "live-latent:%d" % int(next_clip_index)

    def check_lazy_status(
        self, resume_from_clip, next_clip_index, checkpoint_path, live_latent=None
    ):
        if int(resume_from_clip) == int(next_clip_index):
            return []
        if live_latent is None:
            return ["live_latent"]
        return []

    def select(
        self, resume_from_clip, next_clip_index,
        checkpoint_path="h3_extension_checkpoints/clip", live_latent=None
    ):
        next_idx = int(next_clip_index)
        if int(resume_from_clip) == next_idx:
            prev_idx = next_idx - 1
            path = _resolve_checkpoint_path(checkpoint_path, prev_idx)
            video, audio = _load_checkpoint(path)
            global _nested_tensor
            if _nested_tensor is None:
                import comfy.nested_tensor as _runtime_nested_tensor
                _nested_tensor = _runtime_nested_tensor
            latent = {"samples": _nested_tensor.NestedTensor((video, audio))}
            source = "checkpoint clip %d: %s" % (prev_idx, path)
            _LOG.info(
                "h3_checkpoint: RESUME latent-space Clip %d from saved Clip %d; live upstream branch skipped",
                next_idx, prev_idx,
            )
            return (latent, source)

        if live_latent is None:
            raise ValueError(
                "h3_checkpoint: Clip %d needs live_latent unless resume_from_clip is exactly %d"
                % (next_idx, next_idx)
            )
        return (live_latent, "live clip %d" % (next_idx - 1))


class MiniMaxH3CheckpointTrigger:
    """Lazy-select the checkpoint belonging to the last active clip.

    This keeps the final assembler generic for 1..20 clip workflows. Only the
    selected checkpoint dependency is evaluated; the other clip branches remain
    untouched.
    """

    MAX_CLIPS = 20

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "checkpoint_%d" % i: ("STRING", {
                "lazy": True,
                "forceInput": True,
                "tooltip": "Checkpoint path output from Clip %d's H3 Checkpoint Save node." % i,
            })
            for i in range(1, cls.MAX_CLIPS + 1)
        }
        return {
            "required": {
                "clip_count": ("INT", {
                    "default": 20, "min": 1, "max": cls.MAX_CLIPS,
                    "tooltip": "Only the checkpoint input for this last active clip is evaluated."
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("completion_checkpoint",)
    FUNCTION = "select"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Lazy final-generation dependency selector. Requests only the saved "
        "checkpoint for the configured last active clip."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def check_lazy_status(self, clip_count, **kwargs):
        idx = max(1, min(self.MAX_CLIPS, int(clip_count)))
        name = "checkpoint_%d" % idx
        if kwargs.get(name) is None:
            return [name]
        return []

    def select(self, clip_count, **kwargs):
        idx = max(1, min(self.MAX_CLIPS, int(clip_count)))
        name = "checkpoint_%d" % idx
        path = kwargs.get(name)
        if not path:
            raise ValueError(
                "h3_checkpoint: final trigger did not receive Clip %d checkpoint" % idx
            )
        return (path,)


class MiniMaxH3AssembleCheckpoints:
    """Sequentially decode and assemble saved H3 clips without a giant IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE. Saved clip latents are decoded one at a time."
                }),
                "master_audio": ("AUDIO", {
                    "tooltip": "Original master song used as the final soundtrack. H3-generated clip audio is ignored."
                }),
                "checkpoint_path": ("STRING", {
                    "default": "h3_checkpoints/clip",
                    "tooltip": "Checkpoint prefix used by H3 Checkpoint Save."
                }),
                "clip_count": ("INT", {
                    "default": 15, "min": 1, "max": 9999,
                    "tooltip": "Number of sequential saved clip slots to assemble."
                }),
                "context_frames": ("INT", {
                    "default": 39, "min": 0, "max": 9999,
                    "tooltip": "Effective duplicated visual context at the start of Clips 2+."
                }),
                "overlap_frames": ("INT", {
                    "default": 39, "min": 0, "max": 9999,
                    "tooltip": "Linear visual crossfade. If shorter than context_frames, the older duplicated context is discarded before blending, matching H3 Motion Context Trim + KJNodes."
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                }),
                "assembly_mode": (["after_generation", "saved_only"], {
                    "default": "after_generation",
                    "tooltip": "after_generation lazily waits for the final checkpoint dependency. saved_only ignores that dependency and assembles existing checkpoint files only, useful after restarting ComfyUI following a final-assembly crash."
                }),
                "filename_prefix": ("STRING", {
                    "default": "video/motion_context",
                }),
                "pix_fmt": (["yuv420p", "yuv444p"], {"default": "yuv420p"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51}),
                "trim_to_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "completion_checkpoint": ("STRING", {
                    "lazy": True,
                    "forceInput": True,
                    "tooltip": "Connect the final H3 Checkpoint Save path here. In after_generation mode this creates the dependency that guarantees all clip checkpoints exist before assembly; saved_only mode lazily skips it."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("video_path", "frames_written")
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "RAM-safe final music-video assembly from per-clip H3 checkpoints. "
        "Decodes one clip at a time, reproduces KJNodes source-side linear_blend "
        "semantics exactly, streams RGB frames to ffmpeg, and muxes the original "
        "master song."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def check_lazy_status(
        self,
        vae,
        master_audio,
        checkpoint_path,
        clip_count,
        context_frames,
        overlap_frames,
        fps,
        assembly_mode,
        filename_prefix,
        pix_fmt,
        crf,
        trim_to_audio,
        completion_checkpoint=None,
    ):
        if assembly_mode == "after_generation" and completion_checkpoint is None:
            return ["completion_checkpoint"]
        return []

    def assemble(
        self,
        vae,
        master_audio,
        checkpoint_path="h3_checkpoints/clip",
        clip_count=20,
        context_frames=39,
        overlap_frames=39,
        fps=24.0,
        assembly_mode="after_generation",
        filename_prefix="video/motion_context",
        pix_fmt="yuv420p",
        crf=19,
        trim_to_audio=True,
        completion_checkpoint=None,
    ):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("h3_checkpoint: ffmpeg not found on PATH")

        count = int(clip_count)
        context = max(0, int(context_frames))
        overlap = max(0, int(overlap_frames))
        fps = float(fps)
        if count < 1:
            raise ValueError("h3_checkpoint: clip_count must be >= 1")
        if fps <= 0:
            raise ValueError("h3_checkpoint: fps must be > 0")
        if overlap > context and count > 1:
            _LOG.warning(
                "h3_checkpoint: overlap_frames %d exceeds context_frames %d; clamping overlap to context",
                overlap, context,
            )
            overlap = context

        # Resolve every slot before opening ffmpeg. This makes missing/incomplete
        # checkpoints fail immediately without creating a partial output file.
        paths = [_resolve_checkpoint_path(checkpoint_path, i) for i in range(1, count + 1)]
        if assembly_mode == "after_generation" and completion_checkpoint:
            _LOG.info("h3_checkpoint: final generation dependency completed: %s", completion_checkpoint)

        folder, filename, counter, _subfolder, _prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        os.makedirs(folder, exist_ok=True)
        out_path = os.path.join(folder, "%s_%05d-audio.mp4" % (filename, counter))

        tempdir = tempfile.mkdtemp(prefix="h3_checkpoint_assembly_")
        audio_raw = os.path.join(tempdir, "master.f32le")
        audio_sr, audio_channels = _write_f32_audio(audio_raw, master_audio)

        first_video = _load_checkpoint_video(paths[0])
        first_images = _decoded_images(vae, first_video)
        height = int(first_images.shape[1])
        width = int(first_images.shape[2])

        cmd = [
            ffmpeg,
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", "%dx%d" % (width, height),
            "-r", "%.9g" % fps,
            "-i", "-",
            "-f", "f32le",
            "-ar", str(audio_sr),
            "-ac", str(audio_channels),
            "-i", audio_raw,
            "-c:v", "libx264",
            "-pix_fmt", str(pix_fmt),
            "-crf", str(int(crf)),
            "-c:a", "aac",
            "-b:a", "320k",
        ]
        if bool(trim_to_audio):
            cmd.append("-shortest")
        cmd.append(out_path)

        _LOG.info(
            "h3_checkpoint: assembling %d clips sequentially at %dx%d %.6g fps; context=%d overlap=%d -> %s",
            count, width, height, fps, context, overlap, out_path,
        )
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        frames_written = 0
        pending_tail: Optional[torch.Tensor] = None
        try:
            if count == 1 or overlap == 0:
                if count == 1:
                    frames_written += _write_rgb24_frames(proc, first_images)
                else:
                    # No blend: keep the full first clip. Later clips still drop
                    # their duplicated context_frames before appending.
                    frames_written += _write_rgb24_frames(proc, first_images)
            else:
                if int(first_images.shape[0]) <= overlap:
                    raise ValueError(
                        "h3_checkpoint: Clip 1 has %d frames, not enough for %d-frame overlap"
                        % (int(first_images.shape[0]), overlap)
                    )
                frames_written += _write_rgb24_frames(proc, first_images[:-overlap])
                pending_tail = first_images[-overlap:].detach().clone().contiguous()

            del first_images, first_video
            gc.collect()

            for clip_idx in range(2, count + 1):
                video = _load_checkpoint_video(paths[clip_idx - 1])
                images = _decoded_images(vae, video)
                if int(images.shape[1]) != height or int(images.shape[2]) != width:
                    raise ValueError(
                        "h3_checkpoint: Clip %d decoded at %s; expected %dx%d"
                        % (clip_idx, tuple(images.shape[1:3]), height, width)
                    )
                total = int(images.shape[0])
                ctx = min(context, max(0, total - 1))
                ov = min(overlap, ctx)

                if overlap == 0:
                    # Equivalent to hard-dropping the duplicated prefix.
                    frames_written += _write_rgb24_frames(proc, images[ctx:])
                else:
                    if pending_tail is None:
                        raise RuntimeError("h3_checkpoint: internal overlap state is empty")
                    ov = min(ov, int(pending_tail.shape[0]))
                    if ov < 1:
                        frames_written += _write_rgb24_frames(proc, pending_tail)
                        frames_written += _write_rgb24_frames(proc, images[ctx:])
                        pending_tail = None
                    else:
                        # Match KJNodes ImageBatchExtendWithOverlap exactly:
                        # source side, linear_blend, alpha excludes both 0 and 1.
                        blend_start = ctx - ov
                        blend_dst = images[blend_start:ctx]
                        blend_src = pending_tail[-ov:]
                        alpha = torch.linspace(
                            0.0, 1.0, ov + 2,
                            device=blend_src.device,
                            dtype=blend_src.dtype,
                        )[1:-1].view(-1, 1, 1, 1)
                        blended = (1.0 - alpha) * blend_src + alpha * blend_dst
                        frames_written += _write_rgb24_frames(proc, blended)
                        del blended, alpha, blend_src, blend_dst

                        suffix = images[ctx:]
                        if clip_idx < count:
                            if int(suffix.shape[0]) <= overlap:
                                raise ValueError(
                                    "h3_checkpoint: Clip %d has only %d new frames after context; not enough for %d-frame next overlap"
                                    % (clip_idx, int(suffix.shape[0]), overlap)
                                )
                            frames_written += _write_rgb24_frames(proc, suffix[:-overlap])
                            pending_tail = suffix[-overlap:].detach().clone().contiguous()
                        else:
                            frames_written += _write_rgb24_frames(proc, suffix)
                            pending_tail = None

                del images, video
                gc.collect()

            if pending_tail is not None:
                frames_written += _write_rgb24_frames(proc, pending_tail)
                pending_tail = None

            proc.stdin.close()
            stderr = proc.stderr.read()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(
                    "h3_checkpoint: ffmpeg failed with code %d:\n%s"
                    % (rc, stderr.decode("utf-8", errors="replace"))
                )
        except Exception:
            try:
                if proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except OSError:
                pass
            raise
        finally:
            try:
                proc.stderr.close()
            except Exception:
                pass
            try:
                os.remove(audio_raw)
                os.rmdir(tempdir)
            except OSError:
                pass
            gc.collect()
            # Only assembly is done at this point; releasing VAE scratch/cache is
            # preferable to retaining it after a long multi-clip job.
            try:
                global _model_management
                if _model_management is None:
                    try:
                        import comfy.model_management as _runtime_model_management
                        _model_management = _runtime_model_management
                    except (ImportError, ModuleNotFoundError, AttributeError):
                        _model_management = None
                if _model_management is not None:
                    _model_management.soft_empty_cache()
            except Exception:
                pass

        _LOG.info(
            "h3_checkpoint: final assembly wrote %d frames; output %s",
            frames_written, out_path,
        )
        return _final_video_node_output(out_path, (out_path, frames_written), fps)

class MiniMaxH3AssembleExtensionCheckpoints:
    """RAM-safe source-video + generated H3 AV checkpoint assembly."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE. Checkpoints are decoded one at a time."
                }),
                "audio_vae": ("VAE", {
                    "tooltip": "MiniMax H3 audio VAE used to decode each checkpoint's generated audio."
                }),
                "source_frames": ("IMAGE", {
                    "tooltip": "Original source-video frames. They are normalized to the configured output fps."
                }),
                "source_audio": ("AUDIO", {
                    "tooltip": "Original source-video audio. It is followed by the newly generated audio from each extension clip."
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                }),
                "checkpoint_path": ("STRING", {
                    "default": "h3_extension_checkpoints/clip",
                    "tooltip": "Checkpoint prefix used by H3 Checkpoint Save."
                }),
                "clip_count": ("INT", {
                    "default": 6, "min": 1, "max": 9999,
                }),
                "context_frames": ("INT", {
                    "default": 39, "min": 5, "max": 9999,
                    "tooltip": "Requested preserved AV prefix for every generated extension. The first clip clamps against source-video length; later clips clamp against the previous H3 clip."
                }),
                "overlap_frames": ("INT", {
                    "default": 39, "min": 0, "max": 9999,
                    "tooltip": "Linear visual blend at every source/clip and clip/clip seam. Audio drops the duplicated protected prefix and concatenates sample-exactly."
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                }),
                "crop": (["disabled", "center"], {
                    "default": "disabled",
                    "tooltip": "Must match the crop mode used by H3 Existing Video Masked Context for the first source-video prefix."
                }),
                "assembly_mode": (["after_generation", "saved_only"], {
                    "default": "after_generation",
                    "tooltip": "after_generation lazily waits for the last active checkpoint. saved_only assembles already-saved checkpoints after a restart."
                }),
                "filename_prefix": ("STRING", {
                    "default": "video/masked_av_extension",
                }),
                "pix_fmt": (["yuv420p", "yuv444p"], {"default": "yuv420p"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51}),
                "trim_to_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "completion_checkpoint": ("STRING", {
                    "lazy": True,
                    "forceInput": True,
                    "tooltip": "Connect H3 Checkpoint Final Trigger here. saved_only mode lazily skips this dependency."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("video_path", "frames_written")
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "RAM-safe multi-clip existing-video extension assembler. Streams the "
        "canonical source video plus saved H3 extension checkpoints to ffmpeg, "
        "linearly blends every visual seam, and concatenates generated H3 audio "
        "after removing each duplicated protected AV prefix."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def check_lazy_status(
        self, video_vae, audio_vae, source_frames, source_audio, source_fps,
        checkpoint_path, clip_count, context_frames, overlap_frames, fps, crop,
        assembly_mode, filename_prefix, pix_fmt, crf, trim_to_audio,
        completion_checkpoint=None,
    ):
        if assembly_mode == "after_generation" and completion_checkpoint is None:
            return ["completion_checkpoint"]
        return []

    def assemble(
        self, video_vae, audio_vae, source_frames, source_audio, source_fps,
        checkpoint_path="h3_extension_checkpoints/clip", clip_count=6,
        context_frames=39, overlap_frames=39, fps=24.0, crop="disabled",
        assembly_mode="after_generation", filename_prefix="video/masked_av_extension",
        pix_fmt="yuv420p", crf=19, trim_to_audio=True, completion_checkpoint=None,
    ):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("h3_checkpoint: ffmpeg not found on PATH")
        count = int(clip_count)
        if count < 1:
            raise ValueError("h3_checkpoint: clip_count must be >= 1")
        fps = float(fps)
        if fps <= 0:
            raise ValueError("h3_checkpoint: fps must be > 0")
        overlap = max(0, int(overlap_frames))
        requested_context = max(5, int(context_frames))
        if getattr(source_frames, "ndim", 0) != 4 or int(source_frames.shape[0]) < 1:
            raise ValueError("h3_checkpoint: source_frames must be IMAGE [N,H,W,C]")
        if source_audio is None:
            raise ValueError("h3_checkpoint: source_audio is required for AV extension assembly")

        from .existing_video_extension import (
            _cfr_index_map, _resize_images, _stereo_first_batch,
            _resample_waveform, _fit_waveform, _snap_context_length,
        )

        paths = [_resolve_checkpoint_path(checkpoint_path, i) for i in range(1, count + 1)]
        if assembly_mode == "after_generation" and completion_checkpoint:
            _LOG.info("h3_checkpoint: extension final dependency completed: %s", completion_checkpoint)

        source_idx = _cfr_index_map(
            int(source_frames.shape[0]), float(source_fps), source_frames.device, fps
        )
        source_count = int(source_idx.numel())
        raw_frames = []
        first_shape = None
        for path in paths:
            video = _load_checkpoint_video(path)
            if first_shape is None:
                first_shape = tuple(video.shape)
            elif tuple(video.shape[1:2] + video.shape[3:]) != tuple(
                first_shape[1:2] + first_shape[3:]
            ):
                raise ValueError(
                    "h3_checkpoint: chained extension checkpoints use different resolutions; keep every H3 clip on one canvas"
                )
            raw_frames.append(_pixel_frames(int(video.shape[2])))
            del video
        width = int(first_shape[4]) * 16
        height = int(first_shape[3]) * 16

        contexts = []
        available = source_count
        for frames in raw_frames:
            ctx = _snap_context_length(requested_context, available, frames)
            contexts.append(ctx)
            available = frames

        tempdir = tempfile.mkdtemp(prefix="h3_extension_assembly_")
        audio_raw = os.path.join(tempdir, "extension.f32le")
        audio_sr = int(
            getattr(
                audio_vae, "audio_sample_rate_output",
                getattr(audio_vae, "audio_sample_rate", 44100),
            )
        )
        audio_channels = 2
        total_samples = 0
        cumulative_frames = source_count
        try:
            with open(audio_raw, "wb") as af:
                src_wave = _stereo_first_batch(source_audio["waveform"], "source_audio")
                src_wave = _resample_waveform(
                    src_wave, int(source_audio["sample_rate"]), audio_sr, "source_audio"
                )
                src_want = int(round(source_count / fps * audio_sr))
                src_wave = _fit_waveform(src_wave, src_want, "source audio")
                total_samples += _append_f32_audio(af, src_wave)
                del src_wave

                for i, path in enumerate(paths):
                    audio_latent = _load_checkpoint_audio(path)
                    decoded = _decoded_audio(audio_vae, audio_latent)
                    wave = _stereo_first_batch(decoded["waveform"], "generated audio")
                    wave = _resample_waveform(
                        wave, int(decoded["sample_rate"]), audio_sr, "generated audio"
                    )
                    cut = int(round(contexts[i] / fps * audio_sr))
                    if cut >= int(wave.shape[-1]):
                        raise ValueError(
                            "h3_checkpoint: Clip %d decoded audio is shorter than its %d-frame protected prefix"
                            % (i + 1, contexts[i])
                        )
                    wave = wave[..., cut:]
                    unique_frames = raw_frames[i] - contexts[i]
                    cumulative_frames += unique_frames
                    want_total = int(round(cumulative_frames / fps * audio_sr))
                    want = want_total - total_samples
                    wave = _fit_waveform(wave, want, "Clip %d generated audio" % (i + 1))
                    total_samples += _append_f32_audio(af, wave)
                    del wave, decoded, audio_latent
                    gc.collect()

            expected_samples = int(round(cumulative_frames / fps * audio_sr))
            if total_samples != expected_samples:
                raise RuntimeError(
                    "h3_checkpoint: extension audio accounting failed: wrote %d samples, expected %d"
                    % (total_samples, expected_samples)
                )

            folder, filename, counter, _subfolder, _prefix = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory()
            )
            os.makedirs(folder, exist_ok=True)
            out_path = os.path.join(folder, "%s_%05d-audio.mp4" % (filename, counter))

            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "rgb24", "-s", "%dx%d" % (width, height),
                "-r", "%.9g" % fps, "-i", "-",
                "-f", "f32le", "-ar", str(audio_sr), "-ac", str(audio_channels),
                "-i", audio_raw,
                "-c:v", "libx264", "-pix_fmt", str(pix_fmt), "-crf", str(int(crf)),
                "-c:a", "aac", "-b:a", "320k",
            ]
            if bool(trim_to_audio):
                cmd.append("-shortest")
            cmd.append(out_path)
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )

            frames_written = 0
            pending_tail = None
            try:
                first_ov = min(overlap, contexts[0], source_count)
                source_body_n = source_count - first_ov if first_ov > 0 else source_count
                chunk = 32
                for start in range(0, source_body_n, chunk):
                    ids = source_idx[start : min(source_body_n, start + chunk)]
                    part = source_frames.index_select(0, ids)
                    part = _resize_images(part, width, height, crop)
                    frames_written += _write_rgb24_frames(proc, part)
                    del part
                if first_ov > 0:
                    ids = source_idx[source_count - first_ov :]
                    tail = source_frames.index_select(0, ids)
                    tail = _resize_images(tail, width, height, crop)
                    pending_tail = tail.detach().cpu().clone().contiguous()
                    del tail

                for i, path in enumerate(paths):
                    video = _load_checkpoint_video(path)
                    images = _decoded_images(video_vae, video)
                    ctx = contexts[i]
                    ov = min(overlap, ctx)
                    if ov > 0:
                        if pending_tail is None:
                            raise RuntimeError("h3_checkpoint: extension seam tail is empty")
                        ov = min(ov, int(pending_tail.shape[0]))
                        blend_dst = images[ctx - ov : ctx]
                        blend_src = pending_tail[-ov:]
                        alpha = torch.linspace(
                            0.0, 1.0, ov + 2,
                            dtype=blend_src.dtype, device=blend_src.device,
                        )[1:-1].view(-1, 1, 1, 1)
                        blended = (1.0 - alpha) * blend_src + alpha * blend_dst
                        frames_written += _write_rgb24_frames(proc, blended)
                        del blended, alpha, blend_src, blend_dst
                    elif pending_tail is not None:
                        frames_written += _write_rgb24_frames(proc, pending_tail)

                    suffix = images[ctx:]
                    if i < count - 1 and overlap > 0:
                        next_ov = min(overlap, contexts[i + 1], int(suffix.shape[0]) - 1)
                        if next_ov < 1:
                            raise ValueError(
                                "h3_checkpoint: Clip %d has too few unique frames for the next visual overlap"
                                % (i + 1)
                            )
                        frames_written += _write_rgb24_frames(proc, suffix[:-next_ov])
                        pending_tail = suffix[-next_ov:].detach().clone().contiguous()
                    else:
                        frames_written += _write_rgb24_frames(proc, suffix)
                        pending_tail = None
                    del suffix, images, video
                    gc.collect()

                if pending_tail is not None:
                    frames_written += _write_rgb24_frames(proc, pending_tail)
                    pending_tail = None

                proc.stdin.close()
                stderr = proc.stderr.read()
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(
                        "h3_checkpoint: ffmpeg failed with code %d:\n%s"
                        % (rc, stderr.decode("utf-8", errors="replace"))
                    )
            except Exception:
                try:
                    if proc.stdin is not None and not proc.stdin.closed:
                        proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    if os.path.exists(out_path):
                        os.remove(out_path)
                except OSError:
                    pass
                raise
            finally:
                try:
                    proc.stderr.close()
                except Exception:
                    pass

            expected_frames = source_count + sum(
                raw_frames[i] - contexts[i] for i in range(count)
            )
            if frames_written != expected_frames:
                raise RuntimeError(
                    "h3_checkpoint: extension video accounting failed: wrote %d frames, expected %d"
                    % (frames_written, expected_frames)
                )
            _LOG.info(
                "h3_checkpoint: extension assembly wrote %d frames / %d audio samples -> %s",
                frames_written, total_samples, out_path,
            )
            return _final_video_node_output(out_path, (out_path, frames_written), fps)
        finally:
            try:
                os.remove(audio_raw)
                os.rmdir(tempdir)
            except OSError:
                pass
            gc.collect()
            try:
                global _model_management
                if _model_management is None:
                    try:
                        import comfy.model_management as _runtime_model_management
                        _model_management = _runtime_model_management
                    except (ImportError, ModuleNotFoundError, AttributeError):
                        _model_management = None
                if _model_management is not None:
                    _model_management.soft_empty_cache()
            except Exception:
                pass


class MiniMaxH3AssembleStarterOrExtensionCheckpoints:
    """RAM-safe assembly for a starter t2v/i2v clip plus masked AV extensions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE. Starter/extension checkpoints are decoded one at a time."
                }),
                "audio_vae": ("VAE", {
                    "tooltip": "MiniMax H3 audio VAE used to decode generated starter/extension audio."
                }),
                "start_mode": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect H3 Extension Start Mode. load_video = start from an uploaded source video; generate_starter = start from a generated H3 starter checkpoint (t2v/i2v)."
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                }),
                "starter_checkpoint_path": ("STRING", {
                    "default": "h3_extension_checkpoints/starter",
                    "tooltip": "Checkpoint prefix used by the generated t2v/i2v starter clip. Used only in generate_starter mode."
                }),
                "checkpoint_path": ("STRING", {
                    "default": "h3_extension_checkpoints/clip",
                    "tooltip": "Checkpoint prefix used by H3 Checkpoint Save for extension clips."
                }),
                "clip_count": ("INT", {
                    "default": 6, "min": 1, "max": 9999,
                }),
                "context_frames": ("INT", {
                    "default": 39, "min": 5, "max": 9999,
                    "tooltip": "Requested preserved AV prefix for every generated extension. The first extension clamps against the chosen chain start (source video or starter clip); later clips clamp against the previous H3 clip."
                }),
                "overlap_frames": ("INT", {
                    "default": 39, "min": 0, "max": 9999,
                    "tooltip": "Linear visual blend at every seam. Audio drops the duplicated protected prefix and concatenates sample-exactly."
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                }),
                "crop": (["disabled", "center"], {
                    "default": "disabled",
                    "tooltip": "Used only in load_video mode; must match the crop mode used by H3 Start Masked Context / H3 Existing Video Masked Context."
                }),
                "assembly_mode": (["after_generation", "saved_only"], {
                    "default": "after_generation",
                    "tooltip": "after_generation lazily waits for the last active extension checkpoint. saved_only assembles already-saved checkpoints after a restart."
                }),
                "filename_prefix": ("STRING", {
                    "default": "video/masked_av_extension",
                }),
                "pix_fmt": (["yuv420p", "yuv444p"], {"default": "yuv420p"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51}),
                "trim_to_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "source_frames": ("IMAGE", {
                    "lazy": True,
                    "tooltip": "Original source-video frames. Requested only in load_video mode."
                }),
                "source_audio": ("AUDIO", {
                    "lazy": True,
                    "tooltip": "Original source-video audio. Requested only in load_video mode."
                }),
                "completion_checkpoint": ("STRING", {
                    "lazy": True,
                    "forceInput": True,
                    "tooltip": "Connect H3 Checkpoint Final Trigger here. saved_only mode lazily skips this dependency."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("video_path", "frames_written")
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "RAM-safe multi-clip assembler for the new masked-AV extension chain. "
        "Starts from either an uploaded source video or a generated H3 starter "
        "checkpoint (t2v/i2v), then streams saved extension checkpoints to "
        "ffmpeg, linearly blends every visual seam, and concatenates audio "
        "without duplicating the protected AV prefixes."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def check_lazy_status(
        self, video_vae, audio_vae, start_mode, source_fps, starter_checkpoint_path,
        checkpoint_path, clip_count, context_frames, overlap_frames, fps, crop,
        assembly_mode, filename_prefix, pix_fmt, crf, trim_to_audio,
        source_frames=None, source_audio=None, completion_checkpoint=None,
    ):
        needed = []
        if assembly_mode == "after_generation" and completion_checkpoint is None:
            needed.append("completion_checkpoint")
        if str(start_mode) == "load_video":
            if source_frames is None:
                needed.append("source_frames")
            if source_audio is None:
                needed.append("source_audio")
        return needed

    def assemble(
        self, video_vae, audio_vae, start_mode="load_video", source_fps=24.0,
        starter_checkpoint_path="h3_extension_checkpoints/starter",
        checkpoint_path="h3_extension_checkpoints/clip", clip_count=6, context_frames=39,
        overlap_frames=39, fps=24.0, crop="disabled", assembly_mode="after_generation",
        filename_prefix="video/masked_av_extension", pix_fmt="yuv420p", crf=19,
        trim_to_audio=True, source_frames=None, source_audio=None,
        completion_checkpoint=None,
    ):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("h3_checkpoint: ffmpeg not found on PATH")
        count = int(clip_count)
        if count < 1:
            raise ValueError("h3_checkpoint: clip_count must be >= 1")
        fps = float(fps)
        if fps <= 0:
            raise ValueError("h3_checkpoint: fps must be > 0")
        overlap = max(0, int(overlap_frames))
        requested_context = max(5, int(context_frames))

        from .existing_video_extension import (
            _cfr_index_map, _resize_images, _stereo_first_batch,
            _resample_waveform, _fit_waveform, _snap_context_length, _pixel_frames,
        )

        paths = [_resolve_checkpoint_path(checkpoint_path, i) for i in range(1, count + 1)]
        if assembly_mode == "after_generation" and completion_checkpoint:
            _LOG.info("h3_checkpoint: extension final dependency completed: %s", completion_checkpoint)

        raw_frames = []
        first_shape = None
        for path in paths:
            video = _load_checkpoint_video(path)
            if first_shape is None:
                first_shape = tuple(video.shape)
            elif tuple(video.shape[1:2] + video.shape[3:]) != tuple(first_shape[1:2] + first_shape[3:]):
                raise ValueError(
                    "h3_checkpoint: chained extension checkpoints use different resolutions; keep every H3 clip on one canvas"
                )
            raw_frames.append(_pixel_frames(int(video.shape[2])))
            del video

        mode = str(start_mode)
        source_idx = None
        starter_path = None
        starter_source_count = None
        starter_video_shape = None
        if mode == "load_video":
            if getattr(source_frames, "ndim", 0) != 4 or int(source_frames.shape[0]) < 1:
                raise ValueError("h3_checkpoint: source_frames must be IMAGE [N,H,W,C]")
            if source_audio is None:
                raise ValueError("h3_checkpoint: source_audio is required in load_video mode")
            source_idx = _cfr_index_map(int(source_frames.shape[0]), float(source_fps), source_frames.device, fps)
            source_count = int(source_idx.numel())
            width = int(first_shape[4]) * 16
            height = int(first_shape[3]) * 16
        else:
            starter_path = _resolve_checkpoint_path(starter_checkpoint_path, 1)
            starter_video = _load_checkpoint_video(starter_path)
            starter_video_shape = tuple(starter_video.shape)
            if tuple(starter_video.shape[1:2] + starter_video.shape[3:]) != tuple(first_shape[1:2] + first_shape[3:]):
                raise ValueError(
                    "h3_checkpoint: starter checkpoint and extension checkpoints use different resolutions; keep every H3 clip on one canvas"
                )
            starter_source_count = _pixel_frames(int(starter_video.shape[2]))
            source_count = starter_source_count
            width = int(starter_video.shape[4]) * 16
            height = int(starter_video.shape[3]) * 16
            del starter_video

        contexts = []
        available = source_count
        for frames in raw_frames:
            ctx = _snap_context_length(requested_context, available, frames)
            contexts.append(ctx)
            available = frames

        tempdir = tempfile.mkdtemp(prefix="h3_starter_extension_assembly_")
        audio_raw = os.path.join(tempdir, "extension.f32le")
        audio_sr = int(getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 44100)))
        audio_channels = 2
        total_samples = 0
        cumulative_frames = source_count
        try:
            with open(audio_raw, "wb") as af:
                if mode == "load_video":
                    src_wave = _stereo_first_batch(source_audio["waveform"], "source_audio")
                    src_wave = _resample_waveform(src_wave, int(source_audio["sample_rate"]), audio_sr, "source_audio")
                    src_want = int(round(source_count / fps * audio_sr))
                    src_wave = _fit_waveform(src_wave, src_want, "source audio")
                    total_samples += _append_f32_audio(af, src_wave)
                    del src_wave
                else:
                    starter_audio_latent = _load_checkpoint_audio(starter_path)
                    starter_decoded = _decoded_audio(audio_vae, starter_audio_latent)
                    start_wave = _stereo_first_batch(starter_decoded["waveform"], "starter_audio")
                    start_wave = _resample_waveform(start_wave, int(starter_decoded["sample_rate"]), audio_sr, "starter_audio")
                    start_want = int(round(source_count / fps * audio_sr))
                    start_wave = _fit_waveform(start_wave, start_want, "starter audio")
                    total_samples += _append_f32_audio(af, start_wave)
                    del start_wave, starter_decoded, starter_audio_latent

                for i, path in enumerate(paths):
                    audio_latent = _load_checkpoint_audio(path)
                    decoded = _decoded_audio(audio_vae, audio_latent)
                    wave = _stereo_first_batch(decoded["waveform"], "generated audio")
                    wave = _resample_waveform(wave, int(decoded["sample_rate"]), audio_sr, "generated audio")
                    cut = int(round(contexts[i] / fps * audio_sr))
                    if cut >= int(wave.shape[-1]):
                        raise ValueError(
                            "h3_checkpoint: Clip %d decoded audio is shorter than its %d-frame protected prefix"
                            % (i + 1, contexts[i])
                        )
                    wave = wave[..., cut:]
                    unique_frames = raw_frames[i] - contexts[i]
                    cumulative_frames += unique_frames
                    want_total = int(round(cumulative_frames / fps * audio_sr))
                    want = want_total - total_samples
                    wave = _fit_waveform(wave, want, "Clip %d generated audio" % (i + 1))
                    total_samples += _append_f32_audio(af, wave)
                    del wave, decoded, audio_latent
                    gc.collect()

            expected_samples = int(round(cumulative_frames / fps * audio_sr))
            if total_samples != expected_samples:
                raise RuntimeError(
                    "h3_checkpoint: extension audio accounting failed: wrote %d samples, expected %d"
                    % (total_samples, expected_samples)
                )

            folder, filename, counter, _subfolder, _prefix = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory()
            )
            os.makedirs(folder, exist_ok=True)
            out_path = os.path.join(folder, "%s_%05d-audio.mp4" % (filename, counter))

            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "rgb24", "-s", "%dx%d" % (width, height),
                "-r", "%.9g" % fps, "-i", "-",
                "-f", "f32le", "-ar", str(audio_sr), "-ac", str(audio_channels),
                "-i", audio_raw,
                "-c:v", "libx264", "-pix_fmt", str(pix_fmt), "-crf", str(int(crf)),
                "-c:a", "aac", "-b:a", "320k",
            ]
            if bool(trim_to_audio):
                cmd.append("-shortest")
            cmd.append(out_path)
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            frames_written = 0
            pending_tail = None
            try:
                first_ov = min(overlap, contexts[0], source_count)
                source_body_n = source_count - first_ov if first_ov > 0 else source_count

                if mode == "load_video":
                    chunk = 32
                    for start in range(0, source_body_n, chunk):
                        ids = source_idx[start : min(source_body_n, start + chunk)]
                        part = source_frames.index_select(0, ids)
                        part = _resize_images(part, width, height, crop)
                        frames_written += _write_rgb24_frames(proc, part)
                        del part
                    if first_ov > 0:
                        ids = source_idx[source_count - first_ov :]
                        tail = source_frames.index_select(0, ids)
                        tail = _resize_images(tail, width, height, crop)
                        pending_tail = tail.detach().cpu().clone().contiguous()
                        del tail
                else:
                    starter_video = _load_checkpoint_video(starter_path)
                    starter_images = _decoded_images(video_vae, starter_video)
                    del starter_video
                    if source_body_n > 0:
                        frames_written += _write_rgb24_frames(proc, starter_images[:source_body_n])
                    if first_ov > 0:
                        tail = starter_images[source_count - first_ov :]
                        pending_tail = tail.detach().cpu().clone().contiguous()
                    del starter_images

                for i, path in enumerate(paths):
                    video = _load_checkpoint_video(path)
                    images = _decoded_images(video_vae, video)
                    ctx = contexts[i]
                    ov = min(overlap, ctx)
                    if ov > 0:
                        if pending_tail is None:
                            raise RuntimeError("h3_checkpoint: extension seam tail is empty")
                        ov = min(ov, int(pending_tail.shape[0]))
                        blend_dst = images[ctx - ov : ctx]
                        blend_src = pending_tail[-ov:]
                        alpha = torch.linspace(0.0, 1.0, ov + 2, dtype=blend_src.dtype, device=blend_src.device)[1:-1].view(-1, 1, 1, 1)
                        blended = (1.0 - alpha) * blend_src + alpha * blend_dst
                        frames_written += _write_rgb24_frames(proc, blended)
                        del blended, alpha, blend_src, blend_dst
                    elif pending_tail is not None:
                        frames_written += _write_rgb24_frames(proc, pending_tail)

                    suffix = images[ctx:]
                    if i < count - 1 and overlap > 0:
                        next_ov = min(overlap, contexts[i + 1], int(suffix.shape[0]) - 1)
                        if next_ov < 1:
                            raise ValueError(
                                "h3_checkpoint: Clip %d has too few unique frames for the next visual overlap" % (i + 1)
                            )
                        frames_written += _write_rgb24_frames(proc, suffix[:-next_ov])
                        pending_tail = suffix[-next_ov:].detach().clone().contiguous()
                    else:
                        frames_written += _write_rgb24_frames(proc, suffix)
                        pending_tail = None
                    del suffix, images, video
                    gc.collect()

                if pending_tail is not None:
                    frames_written += _write_rgb24_frames(proc, pending_tail)
                    pending_tail = None

                proc.stdin.close()
                stderr = proc.stderr.read()
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(
                        "h3_checkpoint: ffmpeg failed with code %d\n%s" % (rc, stderr.decode("utf-8", errors="replace"))
                    )
            except Exception:
                try:
                    if proc.stdin is not None and not proc.stdin.closed:
                        proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    if os.path.exists(out_path):
                        os.remove(out_path)
                except OSError:
                    pass
                raise
            finally:
                try:
                    proc.stderr.close()
                except Exception:
                    pass

            expected_frames = source_count + sum(raw_frames[i] - contexts[i] for i in range(count))
            if frames_written != expected_frames:
                raise RuntimeError(
                    "h3_checkpoint: extension video accounting failed: wrote %d frames, expected %d"
                    % (frames_written, expected_frames)
                )
            _LOG.info(
                "h3_checkpoint: start-mode %s assembly wrote %d frames / %d audio samples -> %s",
                mode, frames_written, total_samples, out_path,
            )
            return _final_video_node_output(out_path, (out_path, frames_written), fps)
        finally:
            try:
                os.remove(audio_raw)
                os.rmdir(tempdir)
            except OSError:
                pass
            gc.collect()
            try:
                global _model_management
                if _model_management is None:
                    try:
                        import comfy.model_management as _runtime_model_management
                        _model_management = _runtime_model_management
                    except (ImportError, ModuleNotFoundError, AttributeError):
                        _model_management = None
                if _model_management is not None:
                    _model_management.soft_empty_cache()
            except Exception:
                pass


class MiniMaxH3PreviewCheckpointVideo:
    """Decode one saved H3 checkpoint and write a small preview MP4."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE used to decode the saved clip."
                }),
                "audio_vae": ("VAE", {
                    "tooltip": "MiniMax H3 audio VAE used to decode the saved clip audio."
                }),
                "checkpoint_prefix": ("STRING", {
                    "default": "h3_extension_checkpoints/clip",
                    "tooltip": "Checkpoint prefix used by H3 Checkpoint Save. The node resolves clip_N from here when no exact checkpoint path is connected."
                }),
                "clip_index": ("INT", {
                    "default": 1, "min": 1, "max": 9999,
                }),
                "active_through": ("INT", {
                    "default": 1, "min": 1, "max": 9999,
                    "tooltip": "Skip preview generation when this clip index is above the current active extension count."
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                }),
                "filename_prefix": ("STRING", {
                    "default": "video/h3_extension_preview",
                }),
                "pix_fmt": (["yuv420p", "yuv444p"], {"default": "yuv420p"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51}),
                "trim_to_audio": ("BOOLEAN", {"default": True}),
                "missing_behavior": (["skip", "error"], {"default": "skip"}),
            },
            "optional": {
                "checkpoint_path": ("STRING", {
                    "lazy": True,
                    "forceInput": True,
                    "tooltip": "Optional exact path from H3 Checkpoint Save. If connected, this exact file is previewed instead of resolving checkpoint_prefix + clip_index."
                }),
                "ready_signal": ("STRING", {
                    "lazy": True,
                    "forceInput": True,
                    "tooltip": "Optional dependency-only signal. Connect H3 Checkpoint Final Trigger here so previews wait until the active clip chain has finished saving."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("video_path", "frames_written")
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Decode one saved H3 checkpoint and encode a preview MP4. Intended for "
        "per-extension previews in the masked-AV extension workflow. It only "
        "decodes one clip at a time and skips inactive extension slots."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def preview(
        self, video_vae, audio_vae, checkpoint_prefix="h3_extension_checkpoints/clip",
        clip_index=1, active_through=1, fps=24.0, filename_prefix="video/h3_extension_preview",
        pix_fmt="yuv420p", crf=19, trim_to_audio=True, missing_behavior="skip",
        checkpoint_path=None, ready_signal=None,
    ):
        clip_index = int(clip_index)
        if clip_index > int(active_through):
            return ("", 0)

        if checkpoint_path:
            try:
                path = _resolve_checkpoint_path(checkpoint_path, clip_index)
            except Exception:
                path = str(checkpoint_path)
        else:
            path = _resolve_checkpoint_path(checkpoint_prefix, clip_index)

        if not os.path.exists(path):
            if str(missing_behavior) == "skip":
                _LOG.info("h3_checkpoint: preview clip %d skipped because %s does not exist", clip_index, path)
                return ("", 0)
            raise FileNotFoundError(path)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("h3_checkpoint: ffmpeg not found on PATH")

        video = _load_checkpoint_video(path)
        images = _decoded_images(video_vae, video)
        audio_latent = _load_checkpoint_audio(path)
        audio = _decoded_audio(audio_vae, audio_latent)
        width = int(images.shape[2])
        height = int(images.shape[1])
        frames_written = int(images.shape[0])

        tempdir = tempfile.mkdtemp(prefix="h3_preview_clip_")
        audio_raw = os.path.join(tempdir, f"clip_{clip_index:05d}.f32le")
        try:
            audio_sr, audio_channels = _write_f32_audio(audio_raw, audio)
            folder, filename, counter, _subfolder, _prefix = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory()
            )
            os.makedirs(folder, exist_ok=True)
            out_path = os.path.join(folder, "%s_%05d.mp4" % (filename, counter))
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "rgb24", "-s", "%dx%d" % (width, height),
                "-r", "%.9g" % float(fps), "-i", "-",
                "-f", "f32le", "-ar", str(audio_sr), "-ac", str(audio_channels),
                "-i", audio_raw,
                "-c:v", "libx264", "-pix_fmt", str(pix_fmt), "-crf", str(int(crf)),
                "-c:a", "aac", "-b:a", "320k",
            ]
            if bool(trim_to_audio):
                cmd.append("-shortest")
            cmd.append(out_path)
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                _write_rgb24_frames(proc, images)
                proc.stdin.close()
                stderr = proc.stderr.read()
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(
                        "h3_checkpoint: ffmpeg preview failed with code %d\n%s" % (rc, stderr.decode("utf-8", errors="replace"))
                    )
            except Exception:
                try:
                    if proc.stdin is not None and not proc.stdin.closed:
                        proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    if os.path.exists(out_path):
                        os.remove(out_path)
                except OSError:
                    pass
                raise
            finally:
                try:
                    proc.stderr.close()
                except Exception:
                    pass

            _LOG.info("h3_checkpoint: preview clip %d wrote %d frames -> %s", clip_index, frames_written, out_path)
            return _final_video_node_output(out_path, (out_path, frames_written), fps)
        finally:
            try:
                os.remove(audio_raw)
                os.rmdir(tempdir)
            except OSError:
                pass
            del audio, audio_latent, images, video
            gc.collect()
            try:
                global _model_management
                if _model_management is None:
                    try:
                        import comfy.model_management as _runtime_model_management
                        _model_management = _runtime_model_management
                    except (ImportError, ModuleNotFoundError, AttributeError):
                        _model_management = None
                if _model_management is not None:
                    _model_management.soft_empty_cache()
            except Exception:
                pass


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3CheckpointSave": MiniMaxH3CheckpointSave,
    "MiniMaxH3CheckpointSavePath": MiniMaxH3CheckpointSavePath,
    "MiniMaxH3CheckpointLoadPath": MiniMaxH3CheckpointLoadPath,
    "MiniMaxH3CheckpointTailFrames": MiniMaxH3CheckpointTailFrames,
    "MiniMaxH3ResumeCheckpointLatent": MiniMaxH3ResumeCheckpointLatent,
    "MiniMaxH3CheckpointLoad": MiniMaxH3CheckpointLoad,
    "MiniMaxH3ResumeTailFrames": MiniMaxH3ResumeTailFrames,
    "MiniMaxH3ResumeOrLiveLatent": MiniMaxH3ResumeOrLiveLatent,
    "MiniMaxH3CheckpointTrigger": MiniMaxH3CheckpointTrigger,
    "MiniMaxH3AssembleCheckpoints": MiniMaxH3AssembleCheckpoints,
    "MiniMaxH3AssembleExtensionCheckpoints": MiniMaxH3AssembleExtensionCheckpoints,
    "MiniMaxH3AssembleStarterOrExtensionCheckpoints": MiniMaxH3AssembleStarterOrExtensionCheckpoints,
    "MiniMaxH3PreviewCheckpointVideo": MiniMaxH3PreviewCheckpointVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3CheckpointSave": "H3 Checkpoint Save",
    "MiniMaxH3CheckpointSavePath": "H3 Checkpoint Save Path (Disk-Backed)",
    "MiniMaxH3CheckpointLoadPath": "H3 Checkpoint Load Path",
    "MiniMaxH3CheckpointTailFrames": "H3 Checkpoint Tail Frames",
    "MiniMaxH3ResumeCheckpointLatent": "H3 Resume / Saved AV Latent",
    "MiniMaxH3CheckpointLoad": "H3 Checkpoint Load",
    "MiniMaxH3ResumeTailFrames": "H3 Resume / Live Tail Frames",
    "MiniMaxH3ResumeOrLiveLatent": "H3 Resume / Live AV Latent",
    "MiniMaxH3CheckpointTrigger": "H3 Checkpoint Final Trigger",
    "MiniMaxH3AssembleCheckpoints": "H3 Assemble Checkpoints",
    "MiniMaxH3AssembleExtensionCheckpoints": "H3 Assemble Extension Checkpoints",
    "MiniMaxH3AssembleStarterOrExtensionCheckpoints": "H3 Assemble Starter + Extension Checkpoints",
    "MiniMaxH3PreviewCheckpointVideo": "H3 Preview Checkpoint Video",
}
