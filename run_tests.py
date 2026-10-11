#!/usr/bin/env python
"""Run the Huntian test suite from a clean checkout.

    python run_tests.py

No installation is required: the core package is standard library only and the
suite uses :mod:`unittest`.

NumPy IS required to run the full suite. Four bundled plugins
(``physics.quantum.sp1d``, ``nonlinear.newton``, ``linear.blocktridiag`` and
``transport.dd``) import it at module level, so without NumPy plugin discovery
fails and those tests surface as errors rather than skips.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _entry in (str(ROOT / "python"), str(ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print()
    print(
        f"ran {result.testsRun} test(s): "
        f"{len(result.failures)} failure(s), {len(result.errors)} error(s), "
        f"{len(result.skipped)} skipped"
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
