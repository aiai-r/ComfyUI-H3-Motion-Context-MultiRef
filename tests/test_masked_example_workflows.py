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



def test_masked_two_video_bridge_example():
    data = _load("UTILITY - AV Bridge.json")
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
