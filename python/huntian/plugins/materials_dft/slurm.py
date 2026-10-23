"""Slurm job generation and queue polling for ``backend="qe_slurm"``.

Lazy-imported exactly like :mod:`materials_dft.qe_driver`: the ``table``
backend must keep working on machines with neither ase nor cluster access.

Status: stub. Phase 3 of the DFT roadmap (SQS alloy scans) will generate job
scripts, submit them, poll the queue, and write single-point results into
``dft_cache/<cache_key>.json`` in the same schema the lookup table uses.
"""

from __future__ import annotations

from typing import Any


class SlurmBackendError(Exception):
    """A cluster job could not be generated, submitted or collected."""


def submit_sqs_scan(*args: Any, **kwargs: Any) -> Any:
    """Generate and submit an SQS supercell scan. Not implemented yet."""
    raise SlurmBackendError(
        "backend='qe_slurm' is not implemented yet (DFT roadmap phase 3, "
        "SQS alloy scans); use backend='table'"
    )
