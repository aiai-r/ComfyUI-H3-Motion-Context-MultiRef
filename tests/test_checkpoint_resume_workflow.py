import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "example_workflows" / "NEW - Latent Masking - Music Video - Lip-Sync + Reference images.json"


def _load():
    return json.loads(WF.read_text())


def test_checkpoint_resume_workflow_structure():
    data = _load()
    nodes = data["nodes"]
    assert sum(n["type"] == "SamplerCustomAdvanced" for n in nodes) == 20
    assert sum(n["type"] == "MiniMaxH3CheckpointSavePath" for n in nodes) == 20
    assert sum(n["type"] == "MiniMaxH3CheckpointTailFrames" for n in nodes) == 19
    assert sum(n["type"] == "MiniMaxH3CheckpointTrigger" for n in nodes) == 1
    final = next(n for n in nodes if n["id"] == 800)
    assert final["type"] == "MiniMaxH3AssembleCheckpoints"


def test_checkpoint_resume_workflow_seeds_are_fixed():
    data = _load()
    seeds = [n for n in data["nodes"] if n["type"] == "RandomNoise"]
    assert len(seeds) == 20
    assert all(n["widgets_values"][1] == "fixed" for n in seeds)


def test_checkpoint_workflow_has_only_clip_local_preview_image_branches():
    data = _load()
    nodes = data["nodes"]
    types = {n["type"] for n in nodes}
    assert "ImageBatchExtendWithOverlap" not in types
    assert "MiniMaxH3MotionContextTrim" not in types
    assert "TrimAudioDuration" not in types

    video_decodes = [n for n in nodes if n["type"] == "VAEDecode" and n.get("title", "").startswith("PREVIEW — CLIP ")]
    audio_decodes = [n for n in nodes if n["type"] == "VAEDecodeAudio" and n.get("title", "").startswith("PREVIEW — CLIP ")]
    combines = [n for n in nodes if n["type"] == "VHS_VideoCombine" and n.get("title", "").startswith("PREVIEW — CLIP ")]
    assert len(video_decodes) == len(audio_decodes) == len(combines) == 20
    assert all(n["widgets_values"]["save_output"] is False for n in combines)

    # Preview IMAGE/AUDIO outputs terminate at their own VHS node; they never feed
    # continuation, checkpoint assembly, or another clip.
    by_id = {n["id"]: n for n in nodes}
    links = {link[0]: link for link in data["links"]}
    for n in video_decodes + audio_decodes:
        for out in n["outputs"]:
            for link_id in out.get("links") or []:
                target = by_id[links[link_id][3]]
                assert target["type"] == "VHS_VideoCombine"


def test_optional_switch_and_trigger_reach_clip_20():
    data = _load()
    nodes = data["nodes"]
    switch = next(n for n in nodes if n.get("title", "").startswith("OPTIONAL CLIPS"))
    assert "2–20" in switch["title"]
    trigger = next(n for n in nodes if n["type"] == "MiniMaxH3CheckpointTrigger")
    names = [i["name"] for i in trigger["inputs"]]
    assert "checkpoint_20" in names
    assert names[-1] == "clip_count"


def test_music_final_assembler_has_direct_file_preview_support():
    src = (ROOT / "h3_checkpoint_resume.py").read_text()
    start = src.index("class MiniMaxH3AssembleCheckpoints:")
    end = src.index("class MiniMaxH3AssembleExtensionCheckpoints:", start)
    segment = src[start:end]
    assert "_final_video_node_output(out_path, (out_path, frames_written), fps)" in segment
    assert "_comfy_ui.PreviewVideo" in src
    assert "_comfy_ui.SavedResult" in src
    # The old custom HTML renderer was unreliable; final previews now use
    # ComfyUI's native PreviewVideo UI payload. The JS file is an inert stub so
    # runtime archives overwrite old installed renderer copies.
    js = (ROOT / "js" / "h3_final_video_preview.js").read_text()
    assert "native ui.PreviewVideo" in js
    assert "document.createElement" not in js


def _node_center(node):
    x, y = node["pos"]
    w, h = node.get("size", [0, 0])
    return (x + w * 0.5, y + h * 0.5)


def _group_contains_node_center(group, node):
    cx, cy = _node_center(node)
    gx, gy, gw, gh = group["bounding"]
    return gx <= cx < gx + gw and gy <= cy < gy + gh


def test_music_group_bypassers_do_not_capture_final_trigger_or_clip1_checkpoint():
    data = _load()
    nodes = data["nodes"]
    groups = data["groups"]
    trigger = next(n for n in nodes if n["type"] == "MiniMaxH3CheckpointTrigger")
    checkpoint1 = next(n for n in nodes if n.get("title") == "CHECKPOINT — Clip 1")

    trigger_groups = [g["title"] for g in groups if _group_contains_node_center(g, trigger)]
    assert trigger_groups == ["FINAL OUTPUT"]

    preview1 = next(g for g in groups if g["title"] == "CLIP PREVIEW — 01")
    assert not _group_contains_node_center(preview1, checkpoint1)

    preview_members = [n for n in nodes if _group_contains_node_center(preview1, n)]
    assert sorted(n["type"] for n in preview_members) == sorted([
        "MiniMaxH3CheckpointLoadPath",
        "VAEDecode",
        "VAEDecodeAudio",
        "VHS_VideoCombine",
    ])


def test_music_chain_is_disk_backed_between_samplers_and_previews():
    data = _load()
    nodes = {n["id"]: n for n in data["nodes"]}
    links = {l[0]: l for l in data["links"]}

    saves = sorted(
        (n for n in nodes.values() if n["type"] == "MiniMaxH3CheckpointSavePath"),
        key=lambda n: n["widgets_values"][1],
    )
    assert len(saves) == 20
    assert all(len(n["outputs"]) == 1 and n["outputs"][0]["type"] == "STRING" for n in saves)

    # Clips 2..20 depend on the previous checkpoint PATH, never the previous live latent.
    tails = [n for n in nodes.values() if n["type"] == "MiniMaxH3CheckpointTailFrames"]
    assert len(tails) == 19
    for n in tails:
        signal = next(i for i in n["inputs"] if i["name"] == "checkpoint_signal")
        link = links[signal["link"]]
        source = nodes[link[1]]
        assert source["type"] == "MiniMaxH3CheckpointSavePath"
        assert link[5] == "STRING"

    # VHS decode branches reload from disk through a path loader, so an evicted
    # sampler latent cannot cause an earlier sampler to be requested for preview.
    loaders = [n for n in nodes.values() if n["type"] == "MiniMaxH3CheckpointLoadPath"]
    assert len(loaders) == 20
    for n in loaders:
        inp = n["inputs"][0]
        link = links[inp["link"]]
        assert nodes[link[1]]["type"] == "MiniMaxH3CheckpointSavePath"
        assert link[5] == "STRING"
