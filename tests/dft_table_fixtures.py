"""Shared fixtures for the ``materials.dft`` tests."""

from __future__ import annotations

import importlib
import sys

#: A minimal valid lookup table for unit tests.
MINIMAL_TABLE = {
    "version": "9.9.9-test",
    "provenance": {"primary_source": "synthetic test fixture"},
    "numerics": {
        "functional": "hse06",
        "pseudopotential_family": "SSSP/PBEsol/efficiency@1.3.0",
        "ecut_wfc_ry": 60.0,
        "kspacing_1_A": 0.20,
        "sqs_seed": 1,
        "spin_orbit": False,
    },
    "materials": {
        "GaN": {
            "source": "synthetic fixture",
            "verification": "partially-verified",
            "convergence": None,
            "values": {
                "a_lattice_A": {"value": 3.19, "kind": "literature"},
                "Eg_eV": {"value": 3.45, "functional": "hse06"},
                "dft_eg_uncorrected_eV": {"value": 1.72, "functional": "pbe"},
            },
        }
    },
}


def table_io_module():
    """Import the plugin's table_io however discovery loaded the package.

    Plugins load under synthetic names (``huntian_plugins.materials_dft_<i>``),
    so a plain ``import`` cannot reach them; find the live package in
    ``sys.modules`` and import its sibling from there.
    """
    for name in sorted(sys.modules):
        if name.startswith("huntian_plugins.materials_dft_"):
            package = sys.modules[name]
            if hasattr(package, "__path__"):
                return importlib.import_module(f"{name}.table_io")
    raise AssertionError(
        "materials_dft plugin package is not loaded; build the registry first"
    )
