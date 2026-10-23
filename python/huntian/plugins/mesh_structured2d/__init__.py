"""Structured 2D mesh generator for a layered AlGaN/GaN HEMT.

Builds a tensor-product grid: a graded vertical axis derived from the layer
stack, and a lateral axis anchored on the gate footprint. The result is a plain
dictionary so that downstream modules stay free of any mesh-library dependency.
"""

from __future__ import annotations

from typing import Any, Iterable

from huntian.errors import Issue
from huntian.registry import BaseModule, ModuleCategory, ModuleManifest, ParamSpec

MANIFEST = ModuleManifest(
    module_id="mesh.structured2d",
    module_version="0.1.0",
    category=ModuleCategory.MESH,
    description=(
        "Structured (tensor-product) 2D mesh for a layered HEMT: graded vertical axis "
        "from the epitaxial stack, lateral axis anchored on the gate footprint."
    ),
    capabilities=("mesh", "mesh.structured2d"),
    requires=("structure",),
    produces=("mesh",),
    verification="partially-verified",
    author="Huntian",
    tags=("mesh", "structured"),
    parameters=(
        ParamSpec.integer(
            "nx",
            41,
            minimum=3,
            maximum=20001,
            doc="number of lateral nodes across the simulated access region",
        ),
        ParamSpec.float(
            "target_spacing_nm",
            1.0,
            unit="nm",
            minimum=1.0e-4,
            doc="desired vertical node spacing inside a layer",
        ),
        ParamSpec.integer(
            "min_nodes_per_layer",
            5,
            minimum=2,
            maximum=10000,
            doc="floor on the vertical cell count of any single layer",
        ),
        ParamSpec.float(
            "lateral_margin_um",
            0.0,
            unit="um",
            minimum=0.0,
            doc="extra lateral extent added outside the source and drain contacts",
        ),
    ),
)


class Structured2DMesh(BaseModule):
    """Generate the mesh description and write it to ``ctx['mesh']``."""

    def validate(self, ctx: Any) -> list[Issue]:
        structure = ctx.require("structure")
        if not structure.get("layers"):
            return [Issue("structure.layers", "the layer stack is empty", "required")]
        return []

    def execute(self, ctx: Any) -> Any:
        structure = ctx.require("structure")
        layers = list(structure["layers"])
        geometry = structure.get("geometry") or {}
        gate = structure.get("gate") or {}

        z_nm, layer_entries = self._build_vertical(layers)
        interfaces = [entry["bottom_nm"] for entry in layer_entries[:-1]]

        gate_length_nm = float(gate.get("length_um", 0.25)) * 1000.0
        source_spacing_nm = float(geometry.get("source_gate_spacing_um", 0.7)) * 1000.0
        drain_spacing_nm = float(geometry.get("gate_drain_spacing_um", 2.0)) * 1000.0
        margin_nm = float(self.param("lateral_margin_um")) * 1000.0

        half_gate = gate_length_nm / 2.0
        x_source = -(half_gate + source_spacing_nm)
        x_drain = half_gate + drain_spacing_nm
        x_min = x_source - margin_nm
        x_max = x_drain + margin_nm

        nx = max(2, int(self.param("nx")))
        step = (x_max - x_min) / (nx - 1)
        x_nm = [x_min + index * step for index in range(nx)]

        mesh = {
            "kind": "structured2d",
            "nx": nx,
            "nz": len(z_nm),
            "n_nodes": nx * len(z_nm),
            "n_cells": nx * (len(z_nm) - 1) if len(z_nm) > 1 else 0,
            "x_nm": x_nm,
            "z_nm": z_nm,
            "layers": layer_entries,
            "interfaces_nm": interfaces,
            "total_depth_nm": layer_entries[-1]["bottom_nm"] if layer_entries else 0.0,
            "lateral_extent_nm": [x_min, x_max],
            "vertical_spacing_nm": {
                "min": min(
                    (b - a for a, b in zip(z_nm, z_nm[1:])), default=0.0
                ),
                "max": max(
                    (b - a for a, b in zip(z_nm, z_nm[1:])), default=0.0
                ),
            },
            "contacts_nm": {
                "source": x_source,
                "gate_center": 0.0,
                "gate_left": -half_gate,
                "gate_right": half_gate,
                "drain": x_drain,
            },
            "notes": [
                "Vertical axis is graded per layer: node count follows target_spacing_nm "
                "with a floor of min_nodes_per_layer.",
                "The lateral axis is uniform; gate-edge refinement is a planned extension "
                "for a later mesh module version.",
            ],
        }
        ctx.put(
            "mesh",
            mesh,
            producer=MANIFEST.module_id,
            module_version=MANIFEST.module_version,
        )
        return ctx

    # -- helpers ---------------------------------------------------------
    def _build_vertical(
        self, layers: Iterable[dict[str, Any]]
    ) -> tuple[list[float], list[dict[str, Any]]]:
        """Return the global vertical node list and per-layer index slices."""
        spacing = float(self.param("target_spacing_nm"))
        floor = int(self.param("min_nodes_per_layer"))

        z_nm: list[float] = []
        entries: list[dict[str, Any]] = []
        cursor = 0.0

        for layer in layers:
            thickness = float(layer["thickness_nm"])
            n_cells = max(floor, int(round(thickness / spacing)))
            top = cursor
            bottom = cursor + thickness

            if not z_nm:
                z_nm.append(top)
            start_index = len(z_nm) - 1
            for index in range(1, n_cells + 1):
                z_nm.append(top + thickness * index / n_cells)
            end_index = len(z_nm) - 1

            entries.append(
                {
                    "name": layer["name"],
                    "role": layer["role"],
                    "material": layer["material"],
                    "x_al": float(layer.get("x_al", 0.0)),
                    "top_nm": top,
                    "bottom_nm": bottom,
                    "thickness_nm": thickness,
                    "n_cells": n_cells,
                    "n_nodes": n_cells + 1,
                    "node_slice": [start_index, end_index],
                }
            )
            cursor = bottom

        return z_nm, entries


MODULE = Structured2DMesh
