from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

# Keep imports in the untouched paper reference working after the thesis move.
__path__.append(str(REPO_ROOT / "thesis"))

__all__ = ["PACKAGE_ROOT", "REPO_ROOT"]
