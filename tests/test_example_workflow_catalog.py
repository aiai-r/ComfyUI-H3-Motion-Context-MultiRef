from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "example_workflows"


def test_example_workflow_catalog_is_tight_and_current():
    names = sorted(p.name for p in WF.glob("*.json"))
    assert names == sorted([
        "NEW - AV Extension.json",
        "NEW - Music Video.json",
        "OLD - Hybrid Extension.json",
        "OLD - Motion Context - Advanced.json",
        "OLD - Motion Context - Simple.json",
        "UTILITY - AV Bridge.json",
        "UTILITY - Custom Keyframes.json",
    ])
    assert not any("Live Latent" in name for name in names)
    assert not any("Latent Masking" in name for name in names)


def test_only_two_workflows_are_highlighted_as_new():
    names = sorted(p.name for p in WF.glob("NEW - *.json"))
    assert names == ["NEW - AV Extension.json", "NEW - Music Video.json"]
