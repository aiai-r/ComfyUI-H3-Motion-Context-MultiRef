"""Regression for the reproducible asset-backed defaults in the checkpointed master-song workflow."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "example_workflows" / "NEW - Latent Masking - Music Video - Lip-Sync + Reference images.json"


def load(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def by_id(data):
    return {n["id"]: n for n in data["nodes"]}


def center(node):
    x, y = node["pos"]
    w, h = node.get("size", [0, 0])
    return x + w / 2, y + h / 2


def in_group(node, group):
    cx, cy = center(node)
    gx, gy, gw, gh = group["bounding"]
    return gx <= cx <= gx + gw and gy <= cy <= gy + gh


def test_checkpoint_music_workflow_defaults_to_included_six_clip_demo():
    master = load(MASTER)
    m = by_id(master)

    assert m[910]["widgets_values"][0] == "be6f4e89-4c3e-43e0-93f5-cc723ccd9b14.png"
    assert m[911]["widgets_values"][0] == "c90ee577-98eb-4f6c-9b0c-562a6b448d69.png"
    assert m[940]["widgets_values"][0] == "I'll Know You by the Scar.wav"

    assert m[101]["widgets_values"] == [15]
    assert m[970]["widgets_values"] == [8, "fixed"]
    assert m[1733]["widgets_values"] == [39, "fixed"]
    assert m[1734]["widgets_values"] == [39, "fixed"]
    assert m[1758]["widgets_values"] == [6, "fixed"]
    assert m[100]["widgets_values"] == ["16:9 (Widescreen)", 1, 32]
    assert m[973]["widgets_values"] == ["res_multistep"]

    expected = {
        1: (514005817509111, "05c1d170f31020134925d38ddd3424392337b4bc5c8fde77f04a6ce8aa8be6e1"),
        2: (903826866713850, "bf7e1c17bde534faa6cccea39dfa739fdcb48e90b9c1e15b998e01dace1c2e63"),
        3: (208140829245950, "8f431aad311b734f817ce9d660080946331b2c03a922d1fb89fb663bd86be81e"),
        4: (378941378675234, "e2e3f526d7a3dedb4081ebab75beab79c8572741849e2e24454cca99a2969c76"),
        5: (162450393808085, "2ce7e66dd37b3c9e67b6532e7db42cc7790d76779c0a05cd70afe8a3c05fef82"),
        6: (661271193620413, "72fc91bcd67eba00539c1e934f2b8ee5f42228b4e9f4edb702d9aa908670bb6d"),
    }
    for clip, (seed, prompt_sha256) in expected.items():
        ref_id = 110 if clip == 1 else clip * 100
        noise_id = 120 if clip == 1 else clip * 100 + 10
        prompt = m[ref_id]["widgets_values"][0]
        assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == prompt_sha256
        assert m[noise_id]["widgets_values"][0] == seed
        assert m[noise_id]["widgets_values"][1] == "fixed"


def test_checkpoint_music_workflow_uses_generic_prompts_after_demo_and_bypasses_7_to_20():
    master = load(MASTER)
    m = by_id(master)
    groups = {g["id"]: g for g in master["groups"]}

    templates = []
    def ref_id(clip):
        if clip == 1:
            return 110
        if 2 <= clip <= 7:
            return clip * 100
        return (clip + 2) * 100

    for clip in range(7, 21):
        prompt = m[ref_id(clip)]["widgets_values"][0]
        templates.append(prompt)
        assert "Replace this generic section" in prompt
        assert "protected master-song audio" in prompt
    assert len(set(templates)) == 1

    active = {3, 4, 5, 6, 7}  # Optional Clips 2-6.
    inactive = {11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26}  # Clips 7-20.

    for gid in active:
        members = [n for n in master["nodes"] if in_group(n, groups[gid])]
        assert members
        assert all(n.get("mode", 0) == 0 for n in members), (gid, [(n["id"], n.get("mode")) for n in members])

    for gid in inactive:
        members = [n for n in master["nodes"] if in_group(n, groups[gid])]
        assert members
        assert all(n.get("mode", 0) == 4 for n in members), (gid, [(n["id"], n.get("mode")) for n in members])
