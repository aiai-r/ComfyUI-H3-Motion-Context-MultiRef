"""Static checks for the Update 3 masked example workflows."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "example_workflows"


def _load(name):
    data = json.loads((WF / name).read_text(encoding="utf-8"))
    ids = {n["id"] for n in data["nodes"]}
    link_ids = {link[0] for link in data["links"]}
    assert len(ids) == len(data["nodes"])
    for link in data["links"]:
        assert link[1] in ids
        assert link[3] in ids
    for node in data["nodes"]:
        for inp in node.get("inputs", []):
            lid = inp.get("link")
            if lid is not None:
                assert lid in link_ids
        for out in node.get("outputs", []):
            for lid in out.get("links") or []:
                assert lid in link_ids
    return data


def _types(data):
    return [n["type"] for n in data["nodes"]]


def _node(data, type_name):
    return next(n for n in data["nodes"] if n["type"] == type_name)


def test_masked_one_video_example():
    data = _load("NEW - Latent Masking - AV Extension - Minimal Single Clip.json")
    types = _types(data)
    assert "MiniMaxH3ExistingVideoMaskedContext" in types
    assert "MiniMaxH3AddGuide" not in types
    assert "TrimAudioDuration" not in types
    assert "CreateVideo" not in types
    assert "SaveVideo" not in types
    assert types.count("VHS_LoadVideo") == 1
    assert types.count("VHS_VideoCombine") == 2
    assert types.count("MiniMaxH3MotionContextTrim") == 2
    assert types.count("MiniMaxH3AssembleExtension") == 1

    target = _node(data, "MiniMaxH3ImageToVideo")
    assert target["widgets_values"][3] == 192
    masked = _node(data, "MiniMaxH3ExistingVideoMaskedContext")
    assert masked["widgets_values"][:2] == [24.0, 39]
    stitch = _node(data, "ImageBatchExtendWithOverlap")
    assert stitch["widgets_values"] == [39, "source", "linear_blend"]


def test_masked_one_video_exact_audio_and_vhs_outputs():
    data = _load("NEW - Latent Masking - AV Extension - Minimal Single Clip.json")
    nodes = {n["id"]: n for n in data["nodes"]}
    links = {link[0]: link for link in data["links"]}

    masked = _node(data, "MiniMaxH3ExistingVideoMaskedContext")
    trims = [n for n in data["nodes"] if n["type"] == "MiniMaxH3MotionContextTrim"]
    main_trim = next(n for n in trims if next(i for i in n["inputs"] if i["name"] == "trim_frames").get("link") is not None)
    raw_trim = next(n for n in trims if next(i for i in n["inputs"] if i["name"] == "trim_frames").get("link") is None)

    # The main continuation trim must follow the actual preserved prefix count
    # emitted by the masked-context node; no hard-coded 1.625-second trim.
    trim_link = next(i for i in main_trim["inputs"] if i["name"] == "trim_frames")["link"]
    assert links[trim_link][1] == masked["id"]
    assert main_trim["widgets_values"] == [0, 24, True, 39]

    # trim_frames=0 + match_tail=True keeps all 192 pictures but clamps raw H3
    # decoded audio to exactly frames/fps for the standalone debug export.
    assert raw_trim["widgets_values"] == [0, 24, True, 39]

    assemble = _node(data, "MiniMaxH3AssembleExtension")
    assert assemble["widgets_values"] == [24, 24, "disabled"]
    cont_audio_link = next(i for i in assemble["inputs"] if i["name"] == "continuation_audio")["link"]
    assert links[cont_audio_link][1] == main_trim["id"]

    vhs = [n for n in data["nodes"] if n["type"] == "VHS_VideoCombine"]
    prefixes = {n["widgets_values"]["filename_prefix"] for n in vhs}
    assert prefixes == {
        "video/H3_masked_AV_extension_raw_H3_192f",
        "video/H3_masked_AV_extension_stitched",
    }
    assert all(n["widgets_values"]["frame_rate"] == 24 for n in vhs)
    assert all(n["widgets_values"]["trim_to_audio"] is False for n in vhs)

    raw = next(n for n in vhs if n["widgets_values"]["filename_prefix"].endswith("raw_H3_192f"))
    raw_img_link = next(i for i in raw["inputs"] if i["name"] == "images")["link"]
    raw_audio_link = next(i for i in raw["inputs"] if i["name"] == "audio")["link"]
    assert links[raw_img_link][1] == raw_trim["id"]
    assert links[raw_audio_link][1] == raw_trim["id"]


def test_masked_two_video_bridge_example():
    data = _load("NEW - Latent Masking - AV Bridge - Two Videos.json")
    types = _types(data)
    assert "MiniMaxH3MaskedAVBridge" in types
    assert "MiniMaxH3AddGuide" not in types
    target = _node(data, "MiniMaxH3ImageToVideo")
    assert target["widgets_values"][3] == 192
    bridge = _node(data, "MiniMaxH3MaskedAVBridge")
    assert bridge["widgets_values"][:3] == [24.0, 24.0, 39]
    stitches = [n for n in data["nodes"] if n["type"] == "ImageBatchExtendWithOverlap"]
    assert len(stitches) == 2
    assert all(n["widgets_values"] == [39, "source", "linear_blend"] for n in stitches)
    trim = _node(data, "TrimAudioDuration")
    assert trim["widgets_values"] == [1.625, 4.75]
