"""ASE / Quantum ESPRESSO driver for the ``qe_*`` backends.

Hard requirement (design document §9): this module is imported **lazily** -
only when ``backend`` is ``qe_local`` or ``qe_slurm`` - so that the default
``table`` backend keeps the plugin free of third-party dependencies and the
JSON Deck keeps running in any environment.

Status: the driver contract and availability checks are in place; the actual
compute pipeline (relax -> SCF -> stress/Born -> postprocess) lands in DFT
roadmap phase 2 together with the D0-D3 rerun benchmarks.
"""

from __future__ import annotations

from typing import Any


class DftDriverError(Exception):
    """A Quantum ESPRESSO run could not be set up or completed."""


def backend_unavailable(backend: str) -> str | None:
    """Return why ``backend`` cannot run here, or ``None`` if it can.

    Only *availability* is probed (ase importable, pw.x on PATH); the driver
    itself is not implemented yet, which is reported separately.
    """
    try:
        import ase  # noqa: F401
    except ImportError:
        return (
            f"backend={backend!r} needs the 'ase' package (pip install ase); "
            f"backend='table' has no third-party dependency"
        )
    if backend == "qe_local" and not _pw_x_on_path():
        return (
            "backend='qe_local' needs the Quantum ESPRESSO 'pw.x' executable on PATH; "
            "backend='table' has no external dependency"
        )
    return None


def _pw_x_on_path() -> bool:
    import shutil

    return shutil.which("pw.x") is not None


def run_qe(*args: Any, **kwargs: Any) -> Any:
    """Execute a Quantum ESPRESSO workflow. Not implemented yet.

    Phase 2 of the DFT roadmap: structure relaxation, SCF, stress tensor,
    Born effective charges, Berry-phase polarization, then
    :mod:`materials_dft.postprocess` turns raw outputs into the same
    artifact contract the ``table`` backend serves.
    """
    raise DftDriverError(
        "backend='qe_local'/'qe_slurm' driver is not implemented yet "
        "(DFT roadmap phase 2); use backend='table'"
    )
