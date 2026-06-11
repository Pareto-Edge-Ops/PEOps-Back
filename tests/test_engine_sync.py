"""Guard against drift between the vendored engine and its source of truth.

The peops engine is edited ONLY in the PEOps-PoC research repo and synced here
via scripts/sync_engine.sh. On machines that have the sibling checkout, any
byte difference fails this test; elsewhere (CI without the research repo) it
skips.
"""

import filecmp
import os
from pathlib import Path

import pytest

_BACK_ENGINE = Path(__file__).resolve().parents[1] / "peops"
_POC_ENGINE = Path(
    os.environ.get("PEOPS_POC_DIR", Path.home() / "Desktop" / "PEOps-PoC")
) / "peops"


def _tree_diff(a: Path, b: Path) -> list[str]:
    diffs: list[str] = []

    def walk(cmp: filecmp.dircmp) -> None:
        for name in cmp.left_only:
            if name == "__pycache__":
                continue
            diffs.append(f"only in {cmp.left}: {name}")
        for name in cmp.right_only:
            if name == "__pycache__":
                continue
            diffs.append(f"only in {cmp.right}: {name}")
        diffs.extend(f"differs: {cmp.left}/{n}" for n in cmp.diff_files)
        for sub in cmp.subdirs.values():
            if Path(sub.left).name == "__pycache__":
                continue
            walk(sub)

    walk(filecmp.dircmp(a, b, ignore=["__pycache__"]))
    return diffs


@pytest.mark.skipif(not _POC_ENGINE.is_dir(), reason="PEOps-PoC sibling not present")
def test_vendored_engine_matches_source_of_truth():
    diffs = _tree_diff(_POC_ENGINE, _BACK_ENGINE)
    assert not diffs, (
        "vendored engine drifted from PEOps-PoC — run scripts/sync_engine.sh:\n"
        + "\n".join(diffs)
    )
