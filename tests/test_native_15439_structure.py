from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODES = (ROOT / "nodes.py").read_text(encoding="utf-8")
COMPAT = (ROOT / "h3_compat.py").read_text(encoding="utf-8")
MASK = (ROOT / "h3_mask_compat.py").read_text(encoding="utf-8")


def test_classic_path_has_no_old_core_patch_imports():
    assert "patch_layout" not in NODES
    assert "patch_payload" not in NODES
    assert "MC_KEY" not in NODES
    assert "MC_AUDIO_KEY" not in NODES
    assert "apply_layout_patch" not in COMPAT
    assert "apply_payload_patch" not in COMPAT


def test_motion_context_emits_native_guide_shape():
    assert '"resolved_frame_index": 0, "latent": enc' in NODES
    assert 'holder["audio_latent"] = audio_latent' in NODES
    assert 'conditioning[0][1].get("minimax_keyframes", [])' in NODES
    assert '"minimax_frame_count"' not in NODES


def test_custom_keyframes_use_native_positions():
    assert '"resolved_frame_index": int(pixel_index)' in NODES


def test_15375_snapshot_rebased_for_15439_layout():
    assert '"cond_audio": max(t_a, aud_aug)' in MASK
    assert '"cond_audio": 2' in MASK
    assert 'frame_count=payload.get("frame_count")' not in MASK
