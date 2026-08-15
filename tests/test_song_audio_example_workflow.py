"""Static regression for the single current asset-backed music-video workflow."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "example_workflows" / "NEW - Latent Masking - Music Video - Lip-Sync + Reference images.json"
ASSETS = ROOT / "example_workflows" / "assets"


def _load():
    with WORKFLOW.open("r", encoding="utf-8") as f:
        return json.load(f)


def _nodes(data, node_type):
    return [n for n in data["nodes"] if n.get("type") == node_type]


def _input(node, name):
    return next(i for i in node.get("inputs", []) if i.get("name") == name)


def _origin_by_link(data):
    return {link[0]: link[1] for link in data.get("links", [])}


def test_current_music_video_keeps_image_refs_and_master_song_out_of_ref_audio():
    data = _load()

    unets = _nodes(data, "UNETLoader")
    assert len(unets) == 1
    assert "minimax_h3_fl2va" in unets[0]["widgets_values"][0]

    refs = [n for n in _nodes(data, "MiniMaxH3ReferenceToVideo") if n.get("mode", 0) == 0]
    assert len(refs) == 6
    for node in refs:
        assert _input(node, "ref_images.ref_image_0")["link"] is not None
        assert _input(node, "ref_images.ref_image_1")["link"] is not None
        for inp in node.get("inputs", []):
            if inp.get("name", "").startswith("ref_audios.ref_audio_"):
                assert inp.get("link") is None

    song_nodes = [n for n in _nodes(data, "MiniMaxH3SongMaskedAVContext") if n.get("mode", 0) == 0]
    assert len(song_nodes) == 6
    for node in song_nodes:
        assert _input(node, "master_audio")["link"] is not None

    first = next(n for n in song_nodes if "Clip 1" in n.get("title", ""))
    assert _input(first, "source_frames")["link"] is None
    for node in song_nodes:
        if node is first:
            continue
        assert _input(node, "source_frames")["link"] is not None
        assert _input(node, "vae")["link"] is not None


def test_current_music_video_uses_original_master_audio_for_streamed_final_and_bundles_assets():
    data = _load()
    origins = _origin_by_link(data)

    load_audio = _nodes(data, "LoadAudio")
    assert len(load_audio) == 1
    master_id = load_audio[0]["id"]
    assert load_audio[0]["widgets_values"][0] == "I'll Know You by the Scar.wav"

    finals = _nodes(data, "MiniMaxH3AssembleCheckpoints")
    assert len(finals) == 1
    audio_link = _input(finals[0], "master_audio")["link"]
    assert origins[audio_link] == master_id

    expected = [
        "be6f4e89-4c3e-43e0-93f5-cc723ccd9b14.png",
        "c90ee577-98eb-4f6c-9b0c-562a6b448d69.png",
        "I'll Know You by the Scar.wav",
        "lyrics.txt",
    ]
    for name in expected:
        path = ASSETS / name
        assert path.is_file() and path.stat().st_size > 0


def test_only_one_music_video_workflow_is_shipped():
    files = sorted((ROOT / "example_workflows").glob("*Music Video*.json"))
    assert [p.name for p in files] == [WORKFLOW.name]
