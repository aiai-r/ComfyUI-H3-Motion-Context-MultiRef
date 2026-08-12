from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_custom_keyframe_frontend_is_exported():
    init_text = (ROOT / "__init__.py").read_text()
    assert 'WEB_DIRECTORY = "./js"' in init_text
    assert '"WEB_DIRECTORY"' in init_text
    assert (ROOT / "js" / "h3_custom_keyframes.js").is_file()
