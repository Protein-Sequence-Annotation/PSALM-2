#!/usr/bin/env python3
"""CLI entry for psalm.data.build_ipr_expanded_test without importing the full psalm package."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_NAME = "_psalm_data_build_ipr_expanded_test"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "psalm" / "data" / "build_ipr_expanded_test.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
