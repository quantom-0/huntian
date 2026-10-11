"""Tests for the module registry: selection, resolution, hot swap, discovery."""

from __future__ import annotations

import unittest
from typing import Any

from huntian.errors import (
    DependencyError,
    DuplicateModuleError,
    ModuleConfigError,
    ModuleLifecycleError,
    ModuleUnavailableError,
    RegistryError,
    VersionConflictError,
)
from huntian.registry import (
    BaseModule,
    ModuleCategory,
    ModuleManifest,
    ModuleRegistry,
    ModuleState,
    ParamSpec,
    discover_plugin_classes,
)

from .helpers import build_registry


# ---------------------------------------------------------------------------
# In-memory test modules
# ---------------------------------------------------------------------------


class Alpha(BaseModule):
    MANIFEST = ModuleManifest(
        module_id="test.alpha",
        module_version="1.0.0",
        category=ModuleCategory.PHYSICS,
        description="test producer",
        capabilities=("alpha",),
        produces=("a",),
        parameters=(
            ParamSpec.float("gain", 1.0, minimum=0.0, maximum=10.0, calibratable=True),
            ParamSpec.string("label", "alpha"),
        ),
    )

    def execute(self, ctx: Any) -> Any:
        ctx.put("a", self.param("gain"), producer="test.alpha")
        return ctx


class AlphaV2(BaseModule):
    MANIFEST = ModuleManifest(
        module_id="test.alpha",
        module_version="2.0.0",
        category=ModuleCategory.PHYSICS,
        description="test producer, v2",
        capabilities=("alpha",),
        produces=("a",),
        parameters=(ParamSpec.float("gain", 2.0, minimum=0.0, maximum=10.0),),
    )

    def execute(self, ctx: Any) -> Any:
        ctx.put("a", self.param("gain") * 100, producer="test.alpha")
        return ctx


class Beta(BaseModule):
    MANIFEST = ModuleManifest(
        module_id="test.beta",
        module_version="1.0.0",
        category=ModuleCategory.EXTRACT,
        description="test consumer",
        capabilities=("beta",),
        requires=("a",),
        produces=("b",),
        dependencies=("test.alpha>=1.0",),
    )

    def execute(self, ctx: Any) -> Any:
        ctx.put("b", ctx.require("a") + 1, producer="test.beta")
        return ctx


class Cyclic1(BaseModule):
    MANIFEST = ModuleManifest(
        module_id="test.cyc1",
        module_version="1.0.0",
        dependencies=("test.cyc2>=1.0",),
    )

    def execute(self, ctx: Any) -> Any:
        return ctx


class Cyclic2(BaseModule):
    MANIFEST = ModuleManifest(
        module_id="test.cyc2",
        module_version="1.0.0",
        dependencies=("test.cyc1>=1.0",),
    )

    def execute(self, ctx: Any) -> Any:
        return ctx


class Boom(BaseModule):
    MANIFEST = ModuleManifest(
        module_id="test.boom",
        module_version="1.0.0",
        description="always fails",
    )

    def execute(self, ctx: Any) -> Any:
        raise RuntimeError("boom")


class IncompatibleApi(BaseModule):
    MANIFEST = ModuleManifest(
        module_id="test.future",
        module_version="1.0.0",
        api_version="99.0",
    )

    def execute(self, ctx: Any) -> Any:
        return ctx


def fresh_registry() -> ModuleRegistry:
    registry = ModuleRegistry()
    for module_cls in (Alpha, AlphaV2, Beta, Cyclic1, Cyclic2, Boom):
        registry.register(module_cls, origin="test")
    return registry


# ---------------------------------------------------------------------------
# Registration and selection
# ---------------------------------------------------------------------------


class RegistrationTests(unittest.TestCase):
    def test_register_and_select_highest(self) -> None:
        registry = fresh_registry()
        self.assertEqual(registry.select("test.alpha").manifest.module_version, "2.0.0")

    def test_select_with_version_spec(self) -> None:
        registry = fresh_registry()
        record = registry.select("test.alpha", "^1.0.0")
        self.assertEqual(record.manifest.module_version, "1.0.0")

    def test_select_reports_available_versions(self) -> None:
        registry = fresh_registry()
        with self.assertRaises(ModuleUnavailableError) as caught:
            registry.select("test.alpha", "^9.0.0")
        self.assertIn("1.0.0", str(caught.exception))
        self.assertIn("2.0.0", str(caught.exception))

    def test_select_unknown_module_lists_ids(self) -> None:
        registry = fresh_registry()
        with self.assertRaises(ModuleUnavailableError) as caught:
            registry.select("test.nope")
        self.assertIn("test.alpha", str(caught.exception))

    def test_duplicate_registration_is_rejected(self) -> None:
        registry = ModuleRegistry()
        registry.register(Alpha)

        class AlphaClone(BaseModule):
            MANIFEST = Alpha.MANIFEST

            def execute(self, ctx: Any) -> Any:
                return ctx

        with self.assertRaises(DuplicateModuleError):
            registry.register(AlphaClone)

    def test_reregistering_the_same_class_is_idempotent(self) -> None:
        registry = ModuleRegistry()
        first = registry.register(Alpha)
        second = registry.register(Alpha)
        self.assertIs(first, second)

    def test_replace_true_overrides(self) -> None:
        registry = ModuleRegistry()
        registry.register(Alpha)
        record = registry.register(AlphaV2, replace=True)
        self.assertEqual(record.manifest.module_version, "2.0.0")

    def test_incompatible_api_version_is_rejected(self) -> None:
        registry = ModuleRegistry(api_version="1.0")
        with self.assertRaises(RegistryError):
            registry.register(IncompatibleApi)

    def test_unregister(self) -> None:
        registry = fresh_registry()
        self.assertEqual(registry.unregister("test.alpha", "1.0.0"), 1)
        self.assertEqual(registry.versions("test.alpha"), [registry.select("test.alpha").version])
        self.assertEqual(registry.unregister("test.alpha"), 1)
        self.assertEqual(registry.unregister("test.alpha"), 0)

    def test_records_filters(self) -> None:
        registry = fresh_registry()
        physics = registry.records(category=ModuleCategory.PHYSICS)
        self.assertTrue(all(r.manifest.category is ModuleCategory.PHYSICS for r in physics))
        self.assertEqual(len(registry.records(capability="beta")), 1)
        self.assertEqual(len(registry.records(module_id="test.alpha")), 2)

    def test_select_by_capability(self) -> None:
        registry = fresh_registry()
        records = registry.select_by_capability("alpha")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].manifest.module_version, "2.0.0")

    def test_health_and_catalog(self) -> None:
        registry = fresh_registry()
        health = registry.health()
        self.assertEqual(health["module_ids"], 5)
        self.assertTrue(health["ok"])
        self.assertEqual(len(registry.catalog()), 6)
        self.assertIn("test.alpha", registry.render_catalog())


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


class ResolutionTests(unittest.TestCase):
    def test_transitive_dependencies_are_resolved_in_order(self) -> None:
        registry = fresh_registry()
        plan = registry.resolve(["test.beta"])
        self.assertEqual(plan.ids(), ["test.alpha", "test.beta"])
        self.assertIn(("test.alpha", "test.beta"), plan.edges)

    def test_missing_module_raises(self) -> None:
        registry = fresh_registry()
        with self.assertRaises(ModuleUnavailableError):
            registry.resolve(["test.absent"])

    def test_version_conflict_raises(self) -> None:
        registry = fresh_registry()
        with self.assertRaises(VersionConflictError):
            registry.resolve(["test.alpha^1.0.0", "test.alpha^2.0.0"])

    def test_dependency_cycle_raises(self) -> None:
        registry = fresh_registry()
        with self.assertRaises(DependencyError) as caught:
            registry.resolve(["test.cyc1"])
        self.assertIn("cycle", str(caught.exception))

    def test_unverified_modules_produce_warnings(self) -> None:
        registry = fresh_registry()
        plan = registry.resolve(["test.beta"])
        self.assertTrue(any("unverified" in warning for warning in plan.warnings))

    def test_plan_records_lookup(self) -> None:
        registry = fresh_registry()
        plan = registry.resolve(["test.beta"])
        self.assertEqual(plan.record("test.alpha").module_id, "test.alpha")
        with self.assertRaises(ModuleUnavailableError):
            plan.record("test.absent")


# ---------------------------------------------------------------------------
# Instantiation and parameters
# ---------------------------------------------------------------------------


class InstantiationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = fresh_registry()

    def test_instantiate_applies_defaults(self) -> None:
        instance = self.registry.instantiate("test.alpha", version="1.0.0")
        self.assertEqual(instance.param("gain"), 1.0)
        self.assertEqual(instance.param("label"), "alpha")
        self.assertEqual(instance.state, ModuleState.INITIALIZED)

    def test_instantiate_without_version_picks_the_highest(self) -> None:
        instance = self.registry.instantiate("test.alpha")
        self.assertEqual(instance.param("gain"), 2.0)

    def test_unknown_parameter_is_rejected(self) -> None:
        with self.assertRaises(ModuleConfigError) as caught:
            self.registry.instantiate("test.alpha", params={"gain": 1.0, "nope": 2})
        self.assertIn("nope", str(caught.exception))

    def test_out_of_range_parameter_is_rejected(self) -> None:
        with self.assertRaises(ModuleConfigError) as caught:
            self.registry.instantiate("test.alpha", params={"gain": 99.0})
        self.assertIn("less than or equal", str(caught.exception))

    def test_wrong_type_parameter_is_rejected(self) -> None:
        with self.assertRaises(ModuleConfigError):
            self.registry.instantiate("test.alpha", params={"gain": "fast"})

    def test_run_auto_initializes_and_propagates_failure(self) -> None:
        instance = Boom()
        ctx = _StubContext()
        with self.assertRaises(RuntimeError):
            instance.run(ctx)
        self.assertEqual(instance.state, ModuleState.FAILED)

    def test_terminated_module_cannot_be_reinitialized(self) -> None:
        instance = self.registry.instantiate("test.alpha")
        instance.terminate()
        self.assertEqual(instance.state, ModuleState.TERMINATED)
        with self.assertRaises(ModuleLifecycleError):
            instance.initialize({})

    def test_terminate_is_idempotent(self) -> None:
        instance = self.registry.instantiate("test.alpha")
        instance.terminate()
        instance.terminate()

    def test_build_follows_plan_order(self) -> None:
        plan = self.registry.resolve(["test.beta"])
        instances = self.registry.build(plan)
        self.assertEqual([type(i).MANIFEST.module_id for i in instances], plan.ids())
        for instance in instances:
            self.registry.release(instance)

    def test_active_instances_are_tracked(self) -> None:
        instance = self.registry.instantiate("test.alpha")
        self.assertEqual(self.registry.active_instances.get("test.alpha"), 1)
        self.registry.release(instance)
        self.assertNotIn("test.alpha", self.registry.active_instances)


class HotSwapTests(unittest.TestCase):
    def test_new_version_is_picked_up(self) -> None:
        registry = fresh_registry()
        report = registry.hot_swap(AlphaV2)
        self.assertEqual(report.module_id, "test.alpha")
        self.assertEqual(report.active_version, "2.0.0")
        self.assertIn("1.0.0", report.previous_versions)
        self.assertEqual(registry.select("test.alpha").manifest.module_version, "2.0.0")
        self.assertEqual(report.live_instances_unaffected, 0)

    def test_live_instances_keep_running_the_old_version(self) -> None:
        registry = fresh_registry()
        registry.unregister("test.alpha", "2.0.0")
        instance = registry.instantiate("test.alpha", version="1.0.0", params={"gain": 5.0})
        report = registry.hot_swap(AlphaV2)
        self.assertEqual(report.live_instances_unaffected, 1)
        # The already-built instance is untouched and still answers with v1 behaviour.
        self.assertEqual(instance.param("gain"), 5.0)
        self.assertEqual(instance.state, ModuleState.INITIALIZED)
        self.assertEqual(registry.select("test.alpha").manifest.module_version, "2.0.0")

    def test_old_version_remains_selectable(self) -> None:
        registry = fresh_registry()
        registry.hot_swap(AlphaV2)
        old = registry.select("test.alpha", "^1.0.0")
        self.assertEqual(old.manifest.module_version, "1.0.0")

    def test_swap_emits_an_event(self) -> None:
        registry = fresh_registry()
        seen = []
        registry.bus.subscribe("registry.swapped", lambda event: seen.append(event))
        registry.hot_swap(AlphaV2)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].payload["active_version"], "2.0.0")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class DiscoveryTests(unittest.TestCase):
    def test_bundled_plugins_are_discovered(self) -> None:
        registry = build_registry()
        ids = registry.module_ids()
        for expected in (
            "mesh.structured2d",
            "physics.polarization.ambacher",
            "physics.quantum.analytic_tw",
            "physics.quantum.sp1d",
            "nonlinear.newton",
            "linear.blocktridiag",
            "transport.dd",
            "extract.fom",
            "materials.dft",
        ):
            self.assertIn(expected, ids)
        self.assertEqual(len(registry), 9)

    def test_multi_file_plugins_support_relative_imports(self) -> None:
        # quantum_sp1d is a plugin *package*: its __init__.py does
        # ``from .solver import ...``. Loading it proves the registry registers
        # a synthetic namespace parent so plugin-relative imports resolve.
        registry = build_registry()
        record = registry.select("physics.quantum.sp1d")
        self.assertTrue(record.manifest.module_version.startswith("1."))
        self.assertIn("quantum", record.manifest.capabilities)
        self.assertTrue(record.manifest.parameters)
        instance = record.module_cls()
        instance.initialize({})
        self.assertEqual(instance.state, ModuleState.INITIALIZED)

    def test_namespace_parent_is_registered(self) -> None:
        import sys

        build_registry()
        parents = [name for name in sys.modules if name.startswith("huntian_plugins")]
        self.assertTrue(parents, "expected synthetic plugin packages in sys.modules")
        self.assertIn("huntian_plugins", sys.modules)

    def test_discovery_reports_broken_plugins_without_aborting(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "good").mkdir()
            (root / "good" / "__init__.py").write_text(
                "from huntian.registry import BaseModule, ModuleManifest\n"
                "MANIFEST = ModuleManifest(module_id='tmp.good', module_version='1.0.0')\n"
                "class M(BaseModule):\n"
                "    def execute(self, ctx):\n"
                "        return ctx\n"
                "MODULE = M\n",
                encoding="utf-8",
            )
            (root / "broken").mkdir()
            (root / "broken" / "__init__.py").write_text(
                "raise RuntimeError('this plugin is broken')\n", encoding="utf-8"
            )
            registry = ModuleRegistry()
            report = registry.load_from_paths([root])
            self.assertEqual([r.module_id for r in report.loaded], ["tmp.good"])
            self.assertEqual(len(report.errors), 1)
            self.assertIn("broken", report.errors[0][1])
            self.assertFalse(report.ok)

    def test_missing_path_is_reported(self) -> None:
        found, errors = discover_plugin_classes(["does/not/exist"])
        self.assertEqual(found, [])
        self.assertEqual(len(errors), 1)


class _StubContext:
    """Minimal stand-in used only to prove the lifecycle guard."""

    def put(self, *args: Any, **kwargs: Any) -> None:
        pass

    def require(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def get(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def has(self, *args: Any, **kwargs: Any) -> bool:
        return True


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
