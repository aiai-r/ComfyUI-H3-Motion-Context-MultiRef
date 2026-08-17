"""Run the repository CPU/static regressions without a ComfyUI installation.

Several test modules intentionally install different lightweight ``comfy`` mocks.
Loading and executing one test module at a time avoids pytest collection conflicts
between those mocks while still exercising every ``test_*`` function in the suite.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(path: Path):
    name = f"_h3_repo_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_all() -> int:
    passed = 0
    for path in sorted(TESTS.glob("test_*.py")):
        if path.name == "test_repo_suite.py":
            continue
        module = load(path)
        funcs = [
            getattr(module, name)
            for name in sorted(dir(module))
            if name.startswith("test_") and callable(getattr(module, name))
        ]
        for fn in funcs:
            fn()
            passed += 1
            print(f"PASS {path.name}::{fn.__name__}")
    print(f"PASS: {passed} repo CPU/static checks")
    return passed


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
