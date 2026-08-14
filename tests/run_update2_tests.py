"""Run repo CPU/static regressions without importing the ComfyUI plugin package.

ComfyUI custom-node roots contain ``__init__.py`` that expects to be imported by
ComfyUI. Plain pytest package discovery tries to import that file before the test
mocks exist. This runner loads the test modules directly by file path instead.
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
    name = f"_h3_update2_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    passed = 0
    for path in sorted(TESTS.glob("test_*.py")):
        module = load(path)
        funcs = [
            getattr(module, name)
            for name in sorted(dir(module))
            if name.startswith("test_") and callable(getattr(module, name))
        ]
        if funcs:
            for fn in funcs:
                fn()
                passed += 1
                print(f"PASS {path.name}::{fn.__name__}")
        elif callable(getattr(module, "main", None)):
            module.main()
            passed += 1
            print(f"PASS {path.name}::main")
    print(f"PASS: {passed} repo CPU/static checks")


if __name__ == "__main__":
    main()
