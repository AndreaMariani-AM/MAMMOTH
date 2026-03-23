#!/usr/bin/env python3
"""
Test that all MIL models can be loaded from their base_mammoth.yaml config.

Run from repo root (mammoth_codebase) with an env that has torch and MIL-Lab deps
(e.g. conda env with nystrom_attention, smooth_topk for CLAM, etc.):

    cd /data/mammoth_codebase && python MIL-Lab/scripts/test_base_mammoth_load.py

Each model is instantiated via create_model('<model>.base_mammoth.uni.none', num_classes=2)
without loading pretrained weights. Failures indicate a mismatch between the
base_mammoth.yaml config and the model's config dataclass.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

# Run from repo root. Paths: repo (modules), repo/modules (components.py), MIL-Lab (src)
_SCRIPT = Path(__file__).resolve()
_MIL_LAB_ROOT = _SCRIPT.parent.parent
_MIL_LAB_SRC = _MIL_LAB_ROOT / "src"
_REPO_ROOT = _MIL_LAB_ROOT.parent
_REPO_MODULES = _REPO_ROOT / "modules"
# Prefer repo modules/components.py over MIL-Lab/src/components package (insert last so it is first)
for p in (_MIL_LAB_SRC, _MIL_LAB_ROOT, _REPO_ROOT, _REPO_MODULES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src._global_mappings import MODEL_ENTRYPOINTS
from src.builder import create_model


def test_all_base_mammoth(num_classes: int = 2) -> Dict[str, bool]:
    """
    Try to load each MIL model from its base_mammoth.yaml config.

    Returns:
        Dict mapping model name to True if load succeeded, False otherwise.
    """
    results = {}
    for model_name in sorted(MODEL_ENTRYPOINTS.keys()):
        full_name = f"{model_name}.base_mammoth.uni.none"
        try:
            model = create_model(full_name, num_classes=num_classes)
            results[model_name] = True
            print(f"  OK  {model_name}")
        except Exception as e:
            results[model_name] = False
            print(f"  FAIL {model_name}: {e}")
    return results


def main() -> int:
    print("Testing MIL model load from base_mammoth.yaml (no pretrained weights)\n")
    results = test_all_base_mammoth()
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{passed}/{total} models loaded successfully.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
