# MODIFIED FORK NOTICE: native-core revision 2026-08-13. Classic Motion Context
# now uses ComfyUI PR #15439 guide + MultiRef support without core monkey-patches.
# Original project by NikoDemon80. See MODIFICATIONS.md. GPL-3.0.

"""Pin previous-clip motion at the head of an H3 clip.

Wire it between a stock H3 conditioning node and the sampler:

    MiniMaxH3ImageToVideo (or the t2v path)
        -> H3 Motion Context
        -> guider / sampler

Two axes to test, both cheap.

encode_mode
  frames  one VAE call per frame, each pinned as its own cond block. The
          model sees N snapshots at N instants.
  video   one VAE call for the whole run. The H3 video VAE has latent_dim
          3, so it reads the batch axis as time and compresses the run
          into fewer latent steps (5 pixel frames -> 2 steps, 22 -> 7).
          Each step becomes one cond block, so the motion between frames
          lives inside the latent instead of being implied across separate
          stills. Far fewer rows and one VAE load.

anchor_mode
  head    native ComfyUI guide at target frame 0. The guided head is returned
          in the generated clip, so trim that many frames before concatenating.
  before  legacy research mode from the old PackedLayout monkey-patch; rejected
          on the native-core path because native guides live on the target timeline.
"""

import json
import logging
import os

import comfy.utils
import folder_paths
import node_helpers

try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
except ImportError:  # ComfyUI always ships safetensors; belt and braces
    _st_load = _st_save = None

from .h3_compat import ensure_motion_context_compat
from .existing_video_extension import (
    MiniMaxH3ExistingVideoMaskedContext,
    MiniMaxH3AssembleExtension,
)
from .h3_masked_bridge import MiniMaxH3MaskedAVBridge
from .h3_song_audio_context import MiniMaxH3SongMaskedAVContext
from .h3_auto_crop32 import MiniMaxH3CropTo32
from .h3_timing import crossfade_plan

try:
    import torchaudio
except ImportError:
    torchaudio = None

_LOG = logging.getLogger("h3_motion_context")

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24  # H3's native rate; audio latents run at 40 Hz, hence FRAME_RESCALE 5/3
FRAME_RESCALE = 5.0 / 3.0
AUDIO_HZ = 40.0

# H3's video VAE has exact full-run lengths 5, 22, 39, 56, ... (17k+5).
# Motion Context accepts ANY exact frame count. Native runs use one temporal VAE
# encode; off-grid requests automatically fall back to the existing exact
# per-frame/still conditioning path instead of silently shortening the context.
# 39/90/141/... also land exactly on H3's 40 Hz audio clock.


def _ensure_h3_native_guide_support():
    """Verify native #15439 guide/MultiRef support; never patch H3 core."""
    ensure_motion_context_compat()


def _pixel_frames(latent_t):
    """Pixel frames covered by latent_t latent steps."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _step_offsets(latent_t):
    """Pixel-frame index at which each latent step begins."""
    out, acc = [], 0
    for k in range(latent_t):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]; matches the stock helper
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _encode_tail_audio(audio_vae, audio, seconds):
    """Encode the last `seconds` of a clip's audio with the H3 audio VAE.

    Returns ([1, 32, 2, T] latent, T) where T counts 40 Hz latent steps,
    matching what the layout calls ref_audio_t.
    """
    waveform = audio["waveform"]  # [B, C, L]
    sr = int(audio["sample_rate"])
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        if torchaudio is None:
            raise RuntimeError(
                "h3_motion_context: context_audio is %d Hz but the VAE wants %d Hz "
                "and torchaudio is not available to resample." % (sr, vae_sr))
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    want = int(round(seconds * vae_sr))
    have = int(waveform.shape[-1])
    if have < want:
        _LOG.warning("h3_motion_context: context_audio is %.3fs, shorter than the "
                     "%.3fs of pinned video. Pinning what there is.",
                     have / vae_sr, seconds)
    else:
        waveform = waveform[..., have - want:]
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, int(z.shape[-1])


def _streams_from_latent(latent):
    """Unpack an H3 AV latent into its contained streams.

    NestedTensor.__getitem__ broadcasts the index into every contained
    tensor rather than selecting one, so samples[0] would strip the batch
    dimension off both streams. unbind() returns the pair.
    """
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3_motion_context: expected a MiniMax H3 AV latent (a nested "
            "video/audio pair), got %r" % type(samples))
    if not parts:
        raise ValueError("h3_motion_context: AV latent contains no streams")
    return parts


def _video_from_latent(latent):
    """Pull the video stream out of an H3 AV latent."""
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:  # unbatched [C,T,H,W]
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError("h3_motion_context: expected video latent [B,C,T,H,W], "
                         "got shape %s" % (tuple(video.shape),))
    return video


def _audio_tail_from_latent(latent, a_frames):
    """Slice the last `a_frames` worth of audio steps straight out of a
    generated H3 latent, skipping the decode -> re-encode round trip.

    Returns (tail latent [1, C, 2, rt], rt, overhang) where rt counts
    40 Hz latent steps and overhang is the fraction of a step by which the
    clip's audio grid extends past its last pixel frame. H3 rounds the
    audio grid UP (124 frames want 206.67 steps, the layout allocates
    207), so the latent's final step reaches ~overhang/40 s beyond the
    last frame. The decoded-audio path never sees this because match_tail
    cuts it; on this path the caller compensates the placement with it,
    so the pinned content lands exactly where its samples actually sit.
    """
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError(
            "h3_motion_context: context_latent has no audio stream. Wire the "
            "sampler output of an H3 AV graph, not a video-only latent.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:  # unbatched [C,2,T]
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError("h3_motion_context: expected audio latent [B,C,2,T], "
                         "got shape %s" % (tuple(audio.shape),))
    total_t = int(audio.shape[-1])
    frames = _pixel_frames(int(video.shape[2]))
    overhang = total_t - FRAME_RESCALE * frames
    if not (0.0 <= overhang < 1.0):
        _LOG.warning(
            "h3_motion_context: context_latent audio grid is unexpected "
            "(%d steps for %d frames); assuming no overhang.", total_t, frames)
        overhang = 0.0
    rt = int(round(a_frames / float(FPS) * AUDIO_HZ))
    if rt > total_t:
        _LOG.warning("h3_motion_context: asked for %d audio steps, the latent "
                     "has %d. Pinning all of it.", rt, total_t)
        rt = total_t
    if rt < 1:
        raise ValueError("h3_motion_context: audio window is empty")
    tail = audio[:1, ..., total_t - rt:].clone()
    return tail, rt, float(overhang)


class MiniMaxH3MotionContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "context_frames": ("IMAGE",),
                "context_length": ("INT", {
                    "default": 39, "min": 1, "max": 9999,
                    "tooltip": "Any frame count. 39 recommended; 90 for long context. "
                               "39/90/141/... align exactly to H3 video+audio timing."}),
                "encode_mode": (["video", "frames"], {
                    "default": "video",
                    "tooltip": "video: one VAE call, motion lives inside the "
                               "latent, fewer rows. frames: one call per frame, "
                               "each pinned as a separate still."}),
                "anchor_mode": (["head", "before"], {
                    "default": "head",
                    "tooltip": "head uses native ComfyUI guide semantics at frame 0. "
                               "before is a legacy mode and is rejected on native core."}),
                "crop": (["disabled", "center"], {"default": "disabled"}),
                "audio_context_length": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "Frames of tail audio to pin. 0 follows "
                               "context_length; timeline mode end-aligns it "
                               "with the visual context."}),
                "audio_mode": (["timeline", "ref"], {
                    "default": "timeline",
                    "tooltip": "timeline: pinned audio gets coordinates on "
                               "this clip's own timeline, end-aligned with "
                               "the pinned video, so the model reads it as "
                               "this clip's sound so far and continues it. "
                               "ref: stock placement in a span before the "
                               "clip, which the model imitates (similar "
                               "music, not phase-locked) rather than "
                               "continues."}),
            },
            "optional": {
                "context_latent": ("LATENT", {
                    "tooltip": "Previous clip's SAMPLER OUTPUT latent (the same "
                               "one you wire into the decode nodes). When "
                               "supplied, the pinned audio is sliced straight "
                               "from it, skipping the decode/re-encode round "
                               "trip that dulls sound a little more at every "
                               "link of a chain. Takes priority over "
                               "context_audio; audio_vae is not needed on "
                               "this path."}),
                "audio_vae": ("VAE", {
                    "tooltip": "H3 audio VAE. Supply with context_audio to carry "
                               "the previous clip's tail sound across the join. "
                               "Not needed when context_latent is wired."}),
                "context_audio": ("AUDIO", {
                    "tooltip": "Audio of the previous clip. The tail matching the "
                               "pinned frames is encoded and pinned alongside "
                               "them. Ignored when context_latent is wired."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "trim_frames")
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Pin a run of consecutive frames from a previous clip as "
                   "never-denoised conditioning rows, so the model reads real "
                   "motion instead of guessing it from a single still.")

    def apply(self, conditioning, vae, latent, context_frames, context_length,
              encode_mode, anchor_mode, crop, audio_context_length=0,
              audio_mode="timeline", context_latent=None, audio_vae=None,
              context_audio=None):
        _ensure_h3_native_guide_support()

        if anchor_mode != "head":
            raise ValueError(
                "h3_motion_context: anchor_mode='before' belonged to the old "
                "PackedLayout monkey-patch and is intentionally unsupported on "
                "native ComfyUI guides. Use head mode and trim the duplicated head."
            )

        video = _video_from_latent(latent)
        latent_t = int(video.shape[2])
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(latent_t)

        available = int(context_frames.shape[0])
        n = min(int(context_length), available)
        if n < 1:
            raise ValueError("h3_motion_context: context_frames is empty")
        if n < context_length:
            _LOG.warning(
                "h3_motion_context: only %d frames supplied, guiding with %d",
                available, n)
        if n >= frame_count:
            raise ValueError(
                "h3_motion_context: asked to guide %d frames in a %d-frame clip; "
                "the guide must leave room for newly generated frames."
                % (n, frame_count))

        # Native #15439 guide semantics: a valid H3 clip is ONE keyframe whose
        # latent may contain many temporal tokens. PackedLayout assigns one
        # native cond segment with the model's own _video_grid(). Off-grid
        # requests fall back to native arbitrary-position still guides.
        tail = _resize(context_frames[available - n:], width, height, crop)
        native_video_run = (n == 1) or (n >= 5 and (n - 5) % 17 == 0)
        keyframes = []
        if encode_mode == "video" and native_video_run:
            enc = vae.encode(tail)
            if getattr(enc, "ndim", 0) != 5:
                raise ValueError(
                    "h3_motion_context: guide video encoded to %s; expected "
                    "[B,C,T,H,W]" % (tuple(getattr(enc, "shape", ())),))
            covered = _pixel_frames(int(enc.shape[2]))
            if covered != n:
                raise RuntimeError(
                    "h3_motion_context: exact %d-frame guide encoded to a latent "
                    "covering %d frames; refusing a phase-shifted guide." % (n, covered))
            keyframes.append({"resolved_frame_index": 0, "latent": enc})
            guide_kind = "native clip"
            span = covered
        else:
            if encode_mode == "video":
                _LOG.info(
                    "h3_motion_context: %d-frame context is off the native H3 "
                    "clip grid; using native per-frame guides", n)
            for i in range(n):
                enc = vae.encode(tail[i:i + 1])
                if getattr(enc, "ndim", 0) != 5 or int(enc.shape[2]) != 1:
                    raise ValueError(
                        "h3_motion_context: frame %d encoded to %s; expected one "
                        "H3 still latent" % (i, tuple(getattr(enc, "shape", ()))))
                keyframes.append({"resolved_frame_index": i, "latent": enc})
            guide_kind = "native still sequence"
            span = n

        ref_audio_t = 0
        a_frames = 0
        audio_src = "off"
        if context_latent is not None or context_audio is not None:
            a_frames = int(audio_context_length) or span
            if a_frames < 1:
                raise ValueError("h3_motion_context: audio context window is empty")
            if context_latent is not None:
                if context_audio is not None:
                    _LOG.info(
                        "h3_motion_context: both context_latent and context_audio "
                        "wired; using latent audio to avoid a VAE round trip")
                audio_latent, ref_audio_t, _unused_overhang = _audio_tail_from_latent(
                    context_latent, a_frames)
                audio_src = "latent"
            else:
                if audio_vae is None:
                    raise ValueError(
                        "h3_motion_context: context_audio supplied without audio_vae")
                audio_latent, ref_audio_t = _encode_tail_audio(
                    audio_vae, context_audio, a_frames / float(FPS))
                audio_src = "vae"

            if audio_mode == "timeline":
                if a_frames > span:
                    raise ValueError(
                        "h3_motion_context: native timeline audio longer than the "
                        "visual guide would need to start before target frame 0. "
                        "Use audio_context_length <= context_length or ref mode.")
                audio_start = int(span - a_frames)
                # Native #15439 audio guide: cond_audio rows share the guide's
                # target-relative time origin. Attach to an existing keyframe at
                # that index when possible, otherwise add an audio-only guide.
                holder = next(
                    (kf for kf in keyframes
                     if int(kf.get("resolved_frame_index", -999999)) == audio_start),
                    None)
                if holder is None:
                    holder = {"resolved_frame_index": audio_start}
                    keyframes.append(holder)
                holder["audio_latent"] = audio_latent
            else:
                # Keep an explicit stock-reference mode for research/backward
                # compatibility; native timeline mode should be preferred.
                ref = {
                    "kind": "audio",
                    "ref_audio_t": ref_audio_t,
                    "audio_latent": audio_latent,
                }
                conditioning = node_helpers.conditioning_set_values(
                    conditioning, {"minimax_refs": [ref]}, append=True)

        # Match stock MiniMaxH3AddGuide behavior: preserve any keyframes already
        # present, then append our guide(s). Ref2VA refs live separately in
        # minimax_refs and current ComfyUI merges both payload families natively.
        existing = list(conditioning[0][1].get("minimax_keyframes", []))
        existing.extend(keyframes)
        out = node_helpers.conditioning_set_values(
            conditioning, {"minimax_keyframes": existing})

        trim = span
        _LOG.info(
            "h3_motion_context: native #15439 %s/head, %d frames at target 0, "
            "%d guide block(s), %d-frame target at %dx%d, trim %d, audio %s",
            guide_kind, n, len(keyframes), frame_count, width, height, trim,
            ("%d frames -> %d latent steps (%.3fs) from %s as native cond_audio"
             % (a_frames, ref_audio_t, ref_audio_t / AUDIO_HZ, audio_src))
            if ref_audio_t and audio_mode == "timeline"
            else ("%d frames -> %d latent steps from %s as ordinary ref"
                  % (a_frames, ref_audio_t, audio_src))
            if ref_audio_t else "off")
        return (out, trim)


class MiniMaxH3MotionContextTrim:
    """Drop the pinned head off a decoded clip, picture and sound together.

    The pinned frames occupy the start of the delivered timeline, so they
    have to come off before concatenating. Trimming only the images would
    leave the audio a full trim_frames longer than the video, and muxing
    those puts the whole soundtrack ahead of the picture by trim_frames/24
    seconds. At 5 frames that is 208ms, silent on ambience but squarely
    offbeat on anything with a pulse.

    So this takes both streams and removes the same span from each: whole
    frames from the images, the matching number of samples from the
    waveform. Wire trim_frames from the motion context node so the count
    follows whatever the encoder actually produced.

    The tail needs the same treatment for a different reason. H3's audio
    latent runs at 40 Hz against 24 fps picture, and FRAME_RESCALE is 5/3,
    so a 124 frame clip wants 206.67 audio steps and the layout rounds up
    to 207. Every clip therefore ships about 8.3 ms more sound than
    picture. Concatenate two and the second seam is out by 16.7 ms, three
    and it is 25 ms, and the error grows without bound down a chain. It
    reads as a faint dampening at the first join and a short click at
    later ones. Truncating the tail to exactly frames/fps stops it
    accumulating.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "trim_frames": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Decoded audio for the same clip. Trimmed by the "
                               "matching duration so sound stays locked to "
                               "picture. Leave unwired for silent clips."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Frame rate used to convert the trim into an "
                               "audio duration. Must match what you feed "
                               "Create Video."}),
                "match_tail": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Truncate the audio so its duration equals "
                               "frames/fps exactly. H3 rounds its audio grid up, "
                               "so each clip carries about 8ms of extra sound "
                               "that accumulates at every join in a chain."}),
                "video_crossfade_frames": ("INT", {
                    "default": 39, "min": 1, "max": 9999,
                    "tooltip": "Video overlap retained for final KJ crossfade. "
                               "Audio still trims the full context."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "INT")
    RETURN_NAMES = ("images", "audio", "crossfade_images", "crossfade_frames")
    FUNCTION = "trim"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Remove the leading pinned frames from a decoded H3 clip, "
                   "trimming picture and sound by the same duration.")

    def trim(self, images, trim_frames, audio=None, fps=24.0, match_tail=True,
             video_crossfade_frames=39):
        n = max(0, int(trim_frames))
        total = int(images.shape[0])
        if n >= total:
            raise ValueError(
                "h3_motion_context: asked to trim %d frames from a %d frame clip"
                % (n, total))
        out_images = images[n:] if n else images
        # Keep only the LAST part of the repeated context for the visual blend.
        # Example: context=90, crossfade=39 -> drop 51 repeated frames, retain
        # the final 39 matching frames, then append the genuinely new frames.
        requested_crossfade = max(1, int(video_crossfade_frames)) if n else 0
        crossfade_start, effective_crossfade = crossfade_plan(
            n, requested_crossfade)
        crossfade_images = images[crossfade_start:] if crossfade_start else images
        out_audio = audio
        if audio is not None:
            waveform = audio["waveform"]
            sr = int(audio["sample_rate"])
            seconds = n / float(fps)
            cut = int(round(seconds * sr))
            length = int(waveform.shape[-1])
            if cut >= length:
                raise ValueError(
                    "h3_motion_context: trimming %.3fs from %.3fs of audio would "
                    "leave nothing. Check that fps matches the clip."
                    % (seconds, length / sr))
            waveform = waveform[..., cut:]

            if match_tail:
                frames_left = total - n
                want = int(round(frames_left / float(fps) * sr))
                have = int(waveform.shape[-1])
                if have > want:
                    over = have - want
                    waveform = waveform[..., :want]
                    _LOG.info("h3_motion_context: tail trimmed %d samples "
                              "(%.2fms) so audio matches %d frames exactly",
                              over, over / sr * 1000.0, frames_left)
                elif have < want:
                    _LOG.warning("h3_motion_context: audio is %.2fms shorter than "
                                 "%d frames; leaving the tail alone",
                                 (want - have) / sr * 1000.0, frames_left)

            out_audio = {"waveform": waveform, "sample_rate": sr}
            _LOG.info("h3_motion_context: %d frames / %.4fs picture, %.4fs sound, "
                      "drift %.2fms",
                      total - n, (total - n) / float(fps),
                      int(waveform.shape[-1]) / sr,
                      abs((total - n) / float(fps) - int(waveform.shape[-1]) / sr) * 1000.0)
        elif n:
            _LOG.info("h3_motion_context: trimmed %d leading frames, %d remain. "
                      "No audio wired; if this clip has sound, mux it through "
                      "this node or it will run %.3fs ahead of the picture.",
                      n, total - n, n / float(fps))

        return (out_images, out_audio, crossfade_images, effective_crossfade)


def _resolve_latent_path(path, clip_index=0):
    """Turn the loader's path input into a concrete file.

    Accepts an absolute path, a path relative to ComfyUI's output folder,
    or a directory (in either form). For a directory:

      clip_index == 0   the NEWEST .safetensors inside is used. Simple,
                        but NOT retry-safe: re-rolling a clip loads the
                        rejected attempt's own save (see the node docs).
                        Its run counter also numbers ATTEMPTS, not clips.
      clip_index  > 0   exactly that clip's slot is loaded: clip 1 is
                        *_00001.safetensors. Auto-mode files carry a
                        trailing underscore (*_00001_.safetensors) and
                        are never matched, because their numbers count
                        runs and could hold a reject.
    """
    p = (path or "").strip().strip('"').strip("'")
    if not p:
        p = "h3_context"
    candidates = [p, os.path.join(folder_paths.get_output_directory(), p)]
    for c in candidates:
        if os.path.isfile(c):
            return c
        if os.path.isdir(c):
            idx = int(clip_index)
            if idx > 0:
                # indexed slots use the natural name: clip 2 lives in
                # *_00002.safetensors. Auto-mode files carry a trailing
                # underscore (*_00002_.safetensors) and are deliberately
                # NOT matched: their numbers count runs, not clips, so a
                # reject could be sitting in any of them.
                endings = ("_%05d.safetensors" % idx,
                           "_clip%03d.safetensors" % idx)  # older versions
                files = [os.path.join(c, f) for f in os.listdir(c)
                         if f.endswith(endings)]
                if not files:
                    near = [f for f in os.listdir(c)
                            if f.endswith("_%05d_.safetensors" % idx)]
                    hint = ""
                    if near:
                        hint = (" Found %s, which is an auto-numbered save "
                                "(trailing underscore = numbered by RUN, so "
                                "it may be a reject). If it really is clip "
                                "%d, rename it to drop the trailing "
                                "underscore: %s" %
                                (near[0], idx,
                                 near[0].replace("_%05d_" % idx,
                                                 "_%05d" % idx)))
                    raise FileNotFoundError(
                        "h3_motion_context: no saved latent for clip %d "
                        "(no *_%05d.safetensors in %s).%s"
                        % (idx, idx, c, hint))
            else:
                files = [os.path.join(c, f) for f in os.listdir(c)
                         if f.endswith(".safetensors")]
                if not files:
                    raise FileNotFoundError(
                        "h3_motion_context: no saved latents in %s. Run a "
                        "clip with the Save Latent node first." % c)
            return max(files, key=os.path.getmtime)
    raise FileNotFoundError(
        "h3_motion_context: %r is neither a file nor a folder (also tried "
        "relative to the ComfyUI output directory)." % p)


class MiniMaxH3MotionContextSaveLatent:
    """Save an H3 AV latent to disk so the NEXT run can load it.

    Wiring the sampler's output straight into context_latent is a cycle:
    the sampler would be consuming its own result. The latent that motion
    context needs is the PREVIOUS clip's, which lives in the previous run
    -- so it has to cross runs through disk, the same way the frames and
    audio already do. Stock Save/Load Latent can't serialise H3's nested
    video/audio pair; this saves the two streams side by side.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "The sampler's output latent (the same one "
                               "you wire into the decode nodes)."}),
                "filename_prefix": ("STRING", {
                    "default": "h3_context/clip",
                    "tooltip": "Saved under the ComfyUI output folder. The "
                               "default keeps all chain latents in one "
                               "folder so the Load node can always pick "
                               "the newest."}),
                "clip_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "Which clip of the chain THIS is. Saves to "
                               "that clip's fixed slot, so a re-roll "
                               "overwrites its own reject instead of "
                               "stacking new files. Generating clip 2: "
                               "set 2 here and 1 on the Load node. 0 = "
                               "old behaviour, a new numbered file every "
                               "run (numbers count runs, not clips)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("latent_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Save the sampler's AV latent so the next run's Motion "
                   "Context node can pin audio from it via the matching "
                   "Load node.")

    def save(self, latent, filename_prefix, clip_index=0):
        if _st_save is None:
            raise RuntimeError("h3_motion_context: safetensors is not "
                               "available; cannot save latents.")
        parts = _streams_from_latent(latent)
        if len(parts) < 2:
            raise ValueError(
                "h3_motion_context: latent has no audio stream; wire the "
                "sampler output of an H3 AV graph.")
        video = parts[0].cpu().contiguous()
        audio = parts[1].cpu().contiguous()
        folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        if int(clip_index) > 0:
            # fixed slot with the natural name: clip 2 -> *_00002. A
            # re-roll of this clip overwrites its own save, so rejects
            # never accumulate or get loaded later. Auto mode (below)
            # keeps a trailing underscore, which is what excludes its
            # run-numbered files from indexed loading.
            path = os.path.join(folder, "%s_%05d.safetensors"
                                % (filename, int(clip_index)))
        else:
            path = os.path.join(folder, "%s_%05d_.safetensors"
                                % (filename, counter))
        _st_save({"video": video, "audio": audio}, path,
                 metadata={"format": "h3_motion_context_av_v1"})
        _LOG.info("h3_motion_context: saved AV latent to %s (video %s, "
                  "audio %s)", path, tuple(video.shape), tuple(audio.shape))
        return (path,)


class MiniMaxH3MotionContextLoadLatent:
    """Load a saved H3 AV latent for the context_latent input.

    clip_index means exactly what it says: set it to the clip you want to
    CONTINUE FROM, and that clip's slot is loaded. Generating clip 2 from
    clip 1: Load node 1, Save node 2. Re-rolling clip 2 changes nothing --
    it reloads slot 1 and overwrites slot 2's reject. Accept, then bump
    both numbers.

    At 0 it loads the newest file in the folder instead. Simple, but NOT
    retry-safe: a re-roll's newest file is the rejected attempt's own
    save, so the retry gets conditioned on the audio you just rejected.

    The output is ONLY for the Motion Context node's context_latent input.
    It is not a decodable latent -- do not wire it into VAE decode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_path": ("STRING", {
                    "default": "h3_context",
                    "tooltip": "A saved latent file, or a folder (relative "
                               "paths resolve against the ComfyUI output "
                               "directory). Pointing at a specific FILE "
                               "always loads that file, ignoring "
                               "clip_index."}),
                "clip_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "The clip to CONTINUE FROM: that clip's "
                               "slot is loaded. Generating clip 2 from "
                               "clip 1: set 1 here and 2 on the Save "
                               "node. 0 = newest file in the folder "
                               "(NOT retry-safe: a re-roll loads its own "
                               "rejected audio)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Load a latent saved by H3 Motion Context Save Latent, "
                   "for the context_latent input only.")

    @classmethod
    def IS_CHANGED(cls, latent_path, clip_index=0):
        # the path string stays constant while the file behind it changes
        # (newest save, or an overwritten slot), so cache on the resolved
        # file identity instead -- otherwise ComfyUI would happily serve
        # a stale latent forever
        try:
            p = _resolve_latent_path(latent_path, clip_index)
            return "%s:%d" % (p, os.stat(p).st_mtime_ns)
        except Exception:
            return float("NaN")  # unresolvable: never cache

    def load(self, latent_path, clip_index=0):
        if _st_load is None:
            raise RuntimeError("h3_motion_context: safetensors is not "
                               "available; cannot load latents.")
        path = _resolve_latent_path(latent_path, clip_index)
        data = _st_load(path)
        if "video" not in data or "audio" not in data:
            raise ValueError(
                "h3_motion_context: %s is not an h3_motion_context latent "
                "(missing video/audio streams). Was it saved by the stock "
                "Save Latent node instead?" % path)
        _LOG.info("h3_motion_context: loaded AV latent from %s", path)
        # a plain list, not a NestedTensor: only this repo's context_latent
        # input accepts it, which is the point -- it cannot be mistaken
        # for a decodable latent without failing loudly downstream
        return ({"samples": [data["video"], data["audio"]]},)


class _DynamicKeyframeInputs(dict):
    """Dynamic backend input map for keyframe_image_N sockets."""

    def __contains__(self, key):
        return isinstance(key, str) and key.startswith("keyframe_image_")

    def __getitem__(self, key):
        if isinstance(key, str) and key.startswith("keyframe_image_"):
            return ("IMAGE",)
        raise KeyError(key)

    def get(self, key, default=None):
        if isinstance(key, str) and key.startswith("keyframe_image_"):
            return ("IMAGE",)
        return default

class MiniMaxH3CustomKeyframes:
    """Attach still-image H3 keyframes at arbitrary timeline positions."""

    MAX_KEYFRAMES = 32

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": (
                    "CONDITIONING",
                    {
                        "tooltip": (
                            "H3 conditioning. The node replaces its complete "
                            "minimax_keyframes list with the keyframes below."
                        )
                    },
                ),
                "vae": (
                    "VAE",
                    {
                        "tooltip": (
                            "MiniMax H3 video VAE used to encode each still."
                        )
                    },
                ),
                "latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Target MiniMax H3 AV latent; defines resolution "
                            "and exact frame count."
                        )
                    },
                ),
                "keyframe_state": (
                    "STRING",
                    {
                        "default": (
                            '{"count":3,"positions":[1,22,79]}'
                        ),
                        "multiline": False,
                        "tooltip": (
                            "Internal UI state. Normally managed by the "
                            "keyframe position controls."
                        ),
                    },
                ),
                "indexing": (
                    ["1-based", "0-based"],
                    {"default": "1-based"},
                ),
                "crop": (
                    ["disabled", "center"],
                    {"default": "disabled"},
                ),
            },
            "optional": _DynamicKeyframeInputs(),
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Pin still-image MiniMax H3 keyframes at arbitrary output-frame "
        "positions. Starts with 3 keyframes; use + Add keyframe to add more."
    )

    def apply(
        self,
        conditioning,
        vae,
        latent,
        keyframe_state,
        indexing="1-based",
        crop="disabled",
        **kwargs,
    ):
        _ensure_h3_native_guide_support()

        try:
            state = json.loads(keyframe_state or "{}")
        except Exception as exc:
            raise ValueError(
                "h3_motion_context: invalid H3 Custom Keyframes UI state"
            ) from exc

        positions = state.get("positions", [])
        count = int(state.get("count", len(positions)))

        if count < 1 or count > self.MAX_KEYFRAMES:
            raise ValueError(
                "h3_motion_context: Custom Keyframes count must be 1..%d"
                % self.MAX_KEYFRAMES
            )
        if len(positions) < count:
            raise ValueError(
                "h3_motion_context: %d keyframe slots but only %d saved "
                "positions" % (count, len(positions))
            )

        video = _video_from_latent(latent)
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(int(video.shape[2]))

        anchors = []
        for slot in range(1, count + 1):
            raw_position = int(positions[slot - 1])
            pixel_index = (
                raw_position - 1
                if indexing == "1-based"
                else raw_position
            )

            if pixel_index < 0 or pixel_index >= frame_count:
                low, high = (
                    (1, frame_count)
                    if indexing == "1-based"
                    else (0, frame_count - 1)
                )
                raise ValueError(
                    "h3_motion_context: keyframe %d position %d is "
                    "outside %d..%d"
                    % (slot, raw_position, low, high)
                )

            image = kwargs.get("keyframe_image_%d" % slot)
            if image is None:
                raise ValueError(
                    "h3_motion_context: keyframe %d has no image connected"
                    % slot
                )
            if getattr(image, "ndim", 0) != 4:
                raise ValueError(
                    "h3_motion_context: keyframe %d expected IMAGE "
                    "[B,H,W,C]" % slot
                )
            if int(image.shape[0]) != 1:
                raise ValueError(
                    "h3_motion_context: keyframe %d must receive exactly "
                    "one image, not a batch of %d"
                    % (slot, int(image.shape[0]))
                )

            anchors.append((pixel_index, slot, image))

        anchors.sort(key=lambda item: item[0])

        for i in range(1, len(anchors)):
            if anchors[i - 1][0] == anchors[i][0]:
                displayed = (
                    anchors[i][0] + 1
                    if indexing == "1-based"
                    else anchors[i][0]
                )
                raise ValueError(
                    "h3_motion_context: duplicate keyframe position %d"
                    % displayed
                )

        keyframes = []
        for pixel_index, slot, image in anchors:
            resized = _resize(image, width, height, crop)
            encoded = vae.encode(resized)

            if (
                getattr(encoded, "ndim", 0) != 5
                or int(encoded.shape[2]) != 1
            ):
                raise ValueError(
                    "h3_motion_context: keyframe %d encoded to %s; "
                    "expected one H3 still latent [B,C,1,H,W]"
                    % (
                        slot,
                        tuple(getattr(encoded, "shape", ())),
                    )
                )

            keyframes.append({
                "resolved_frame_index": int(pixel_index),
                "latent": encoded,
            })

        out = node_helpers.conditioning_set_values(
            conditioning, {"minimax_keyframes": keyframes})

        shown = [
            p + 1 if indexing == "1-based" else p
            for p, _, _ in anchors
        ]
        _LOG.info(
            "h3_motion_context: Custom Keyframes pinned %d anchors at %s "
            "in a %d-frame %dx%d target",
            len(keyframes),
            shown,
            frame_count,
            width,
            height,
        )

        return (out,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3MotionContext": MiniMaxH3MotionContext,
    "MiniMaxH3MotionContextTrim": MiniMaxH3MotionContextTrim,
    "MiniMaxH3MotionContextSaveLatent": MiniMaxH3MotionContextSaveLatent,
    "MiniMaxH3MotionContextLoadLatent": MiniMaxH3MotionContextLoadLatent,
    "MiniMaxH3CustomKeyframes": MiniMaxH3CustomKeyframes,
    "MiniMaxH3ExistingVideoMaskedContext": MiniMaxH3ExistingVideoMaskedContext,
    "MiniMaxH3MaskedAVBridge": MiniMaxH3MaskedAVBridge,
    "MiniMaxH3SongMaskedAVContext": MiniMaxH3SongMaskedAVContext,
    "MiniMaxH3AssembleExtension": MiniMaxH3AssembleExtension,
    "MiniMaxH3CropTo32": MiniMaxH3CropTo32,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3MotionContext": "H3 Motion Context",
    "MiniMaxH3MotionContextTrim": "H3 Motion Context Trim",
    "MiniMaxH3MotionContextSaveLatent": "H3 Motion Context Save Latent",
    "MiniMaxH3MotionContextLoadLatent": "H3 Motion Context Load Latent",
    "MiniMaxH3CustomKeyframes": "H3 Custom Keyframes",
    "MiniMaxH3ExistingVideoMaskedContext": "H3 Existing Video Masked Context",
    "MiniMaxH3MaskedAVBridge": "H3 Masked AV Bridge",
    "MiniMaxH3SongMaskedAVContext": "H3 Song Audio + Masked Video Context",
    "MiniMaxH3AssembleExtension": "H3 Assemble Existing Video Extension",
    "MiniMaxH3CropTo32": "H3 Crop Source To /32",
}
