"""Tests for the service registry and the Deck's service slots.

Library modules (a nonlinear solver, a linear solver, a compute backend) are
not pipeline stages: they transform no context and produce no artifacts. They
are configured from the Deck and resolved on demand by the stages that need
them. These tests cover that wiring end to end.
"""

from __future__ import annotations

import unittest

from huntian import Executor, ModuleRegistry, Pipeline, ServiceRegistry, SimContext, load_deck
from huntian.deck import SERVICE_SLOTS, SLOT_CAPABILITY, Deck
from huntian.errors import ContextError, ModuleUnavailableError, ServiceUnavailableError
from huntian.registry import ModuleCategory, ModuleState

from . import DECK_PATH, DECK_SP1D_PATH
from .helpers import build_registry, builtin_materials


class ServiceRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.services = ServiceRegistry(cls.registry)

    def test_configure_records_the_binding(self) -> None:
        services = ServiceRegistry(self.registry)
        services.configure("nonlinear.newton", version="^1.0.0", params={"max_iterations": 5})
        self.assertEqual(services.configured()["nonlinear.newton"], "^1.0.0")

    def test_configure_rejects_an_unregistered_module(self) -> None:
        services = ServiceRegistry(self.registry)
        with self.assertRaises(ModuleUnavailableError):
            services.configure("nonlinear.absent")

    def test_configure_can_be_soft(self) -> None:
        services = ServiceRegistry(self.registry)
        services.configure("nonlinear.absent", allow_missing=True)
        self.assertIn("nonlinear.absent", services.configured())

    def test_get_instantiates_a_configured_module(self) -> None:
        services = ServiceRegistry(self.registry)
        services.configure("nonlinear.newton", params={"max_iterations": 9})
        instance = services.get("nonlinear.newton")
        self.assertEqual(instance.state, ModuleState.INITIALIZED)
        self.assertEqual(instance.option_defaults()["max_iterations"], 9)

    def test_get_caches_per_configuration(self) -> None:
        services = ServiceRegistry(self.registry)
        first = services.get("nonlinear.newton", params={"max_iterations": 7})
        again = services.get("nonlinear.newton", params={"max_iterations": 7})
        other = services.get("nonlinear.newton", params={"max_iterations": 8})
        self.assertIs(first, again)
        self.assertIsNot(first, other)

    def test_get_by_capability(self) -> None:
        services = ServiceRegistry(self.registry)
        instance = services.get_by_capability("nonlinear")
        self.assertEqual(type(instance).MANIFEST.module_id, "nonlinear.newton")

    def test_unknown_capability_raises(self) -> None:
        services = ServiceRegistry(self.registry)
        with self.assertRaises(ServiceUnavailableError):
            services.get_by_capability("teleporter")

    def test_soft_lookup_returns_none(self) -> None:
        services = ServiceRegistry(self.registry)
        self.assertIsNone(services.get("nonlinear.absent", required=False))
        self.assertIsNone(services.get_by_capability("teleporter", required=False))

    def test_has_mirrors_the_registry(self) -> None:
        services = ServiceRegistry(self.registry)
        self.assertTrue(services.has("nonlinear.newton"))
        self.assertFalse(services.has("nonlinear.absent"))

    def test_release_all_terminates_every_instance(self) -> None:
        services = ServiceRegistry(self.registry)
        instance = services.get("nonlinear.newton")
        self.assertEqual(len(services.instantiated()), 1)
        self.assertEqual(services.release_all(), 1)
        self.assertEqual(services.instantiated(), [])
        self.assertEqual(instance.state, ModuleState.TERMINATED)


class ServiceSlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.analytic = load_deck(DECK_PATH)
        cls.sp = load_deck(DECK_SP1D_PATH)

    def test_service_slots_are_the_solver_slots(self) -> None:
        self.assertEqual(SERVICE_SLOTS, ("backend", "nonlinear", "linear"))
        for slot in SERVICE_SLOTS:
            self.assertIn(slot, SLOT_CAPABILITY)

    def test_deck_separates_stages_from_services(self) -> None:
        stage_slots = [slot for slot, _ in self.sp.stage_bindings()]
        service_slots = [slot for slot, _ in self.sp.service_bindings()]
        self.assertNotIn("nonlinear", stage_slots)
        self.assertEqual(service_slots, ["nonlinear"])

    def test_pipeline_carries_services_separately(self) -> None:
        pipeline = Pipeline.from_deck(self.sp)
        stage_slots = [stage.slot for stage in pipeline.stages]
        self.assertNotIn("nonlinear", stage_slots)
        self.assertEqual([s.slot for s in pipeline.services], ["nonlinear"])

    def test_service_requirements_participate_in_resolution(self) -> None:
        pipeline = Pipeline.from_deck(self.sp)
        self.assertIn("nonlinear.newton^1.0.0", pipeline.requirements())
        plan = self.registry.resolve(pipeline.requirements(include_optional=False))
        self.assertIn("nonlinear.newton", plan.ids())

    def test_describe_mentions_services(self) -> None:
        self.assertIn("services:", Pipeline.from_deck(self.sp).describe())

    def test_analytic_deck_has_no_services(self) -> None:
        pipeline = Pipeline.from_deck(self.analytic)
        self.assertEqual(pipeline.services, [])


class ExecutorWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = build_registry()
        cls.materials = builtin_materials()
        cls.sp_deck = load_deck(DECK_SP1D_PATH)
        cls.ctx = SimContext(name="services", bus=cls.registry.bus)
        cls.result = Executor(cls.registry).run(
            Pipeline.from_deck(cls.sp_deck),
            ctx=cls.ctx,
            deck=cls.sp_deck,
            materials=cls.materials,
        )

    def test_executor_creates_and_configures_the_service_registry(self) -> None:
        self.assertIsInstance(self.ctx.services, ServiceRegistry)
        self.assertIn("nonlinear.newton", self.ctx.services.configured())

    def test_the_service_was_actually_instantiated(self) -> None:
        self.assertIn("nonlinear.newton", " ".join(self.ctx.services.instantiated()))

    def test_the_run_succeeded_and_used_newton(self) -> None:
        self.assertEqual(self.result.status, "success")
        quantum = self.ctx.get("quantum")
        self.assertEqual(quantum["algorithm"], "newton")
        self.assertGreater(quantum["linear_solves"], 0)

    def test_configured_service_parameters_reach_the_solver(self) -> None:
        binding = self.sp_deck.modules.nonlinear
        self.assertEqual(binding.params["max_iterations"], 40)
        instance = self.ctx.services.get("nonlinear.newton")
        self.assertEqual(instance.option_defaults()["max_iterations"], 40)

    def test_services_can_be_released_by_the_caller(self) -> None:
        services = ServiceRegistry(self.registry)
        services.get("nonlinear.newton")
        self.assertGreater(services.release_all(), 0)

    def test_a_bad_service_binding_fails_before_the_run(self) -> None:
        import json

        raw = json.loads(DECK_SP1D_PATH.read_text(encoding="utf-8"))
        raw["modules"]["nonlinear"]["module"] = "nonlinear.absent"
        deck = Deck(**raw)
        errors, _warnings = deck.validate_against_registry(self.registry)
        self.assertTrue(any(e.code == "module-unavailable" for e in errors))


class ContextServiceTests(unittest.TestCase):
    def test_require_services_explains_a_bare_context(self) -> None:
        ctx = SimContext()
        self.assertIsNone(ctx.services)
        with self.assertRaises(ContextError):
            ctx.require_services()

    def test_require_services_returns_the_registry(self) -> None:
        ctx = SimContext()
        ctx.services = ServiceRegistry(ModuleRegistry())
        self.assertIs(ctx.require_services(), ctx.services)

    def test_newton_algorithm_without_a_registry_is_rejected(self) -> None:
        # validate() catches the missing service before execute() runs.
        registry = build_registry()
        binding = registry.select("physics.quantum.sp1d")
        instance = binding.module_cls()
        instance.initialize({"algorithm": "newton"})

        class _Ctx:
            services = None

            def require(self, key):  # noqa: ANN001
                return {
                    "barrier_thickness_nm": 20.0,
                    "sigma_eff_C_m2": 0.0154,
                } if key == "polarization" else {}

        issues = instance.validate(_Ctx())
        self.assertTrue(any(issue.code == "service" for issue in issues))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
