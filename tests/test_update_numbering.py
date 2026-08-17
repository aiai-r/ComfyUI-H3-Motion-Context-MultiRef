from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_modifications_contains_updates_1_through_6():
    text = (ROOT / "MODIFICATIONS.md").read_text(encoding="utf-8")

    for number in range(1, 7):
        assert f"Update {number}" in text


def test_update_history_is_consolidated():
    assert (ROOT / "MODIFICATIONS.md").exists()
    assert (ROOT / "TECHNICAL_ARCHITECTURE.md").exists()

    for name in (
        "H3_MASKED_AV_BRIDGE.md",
        "NATIVE_CORE_15439.md",
        "UPDATE_3_2026-08-14.md",
        "UPDATE_4_2026-08-14.md",
        "UPDATE_5_2026-08-15.md",
    ):
        assert not (ROOT / name).exists()


def test_update_4_and_5_history_is_preserved_in_modifications():
    text = (ROOT / "MODIFICATIONS.md").read_text(encoding="utf-8")

    assert "Update 4" in text
    assert "2026-08-14" in text
    assert "Update 5" in text
    assert "2026-08-15" in text


def test_update_6_direct_latent_history_is_preserved():
    text = (ROOT / "MODIFICATIONS.md").read_text(encoding="utf-8")
    assert "Update 6" in text
    assert "2026-08-17" in text
    assert "checkpoint-free direct-latent" in text
    assert "direct VHS streaming" in text or "direct single-pass" in text
    assert "timebase" in text
