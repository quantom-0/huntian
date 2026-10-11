"""Command line entry point.

Examples
--------
List every registered plugin and its parameters::

    python -m huntian.cli list-plugins --plugin-dir plugins

Validate a Deck without running it (the LLM repair-loop primitive)::

    python -m huntian.cli validate decks/demo_hemt.json --plugin-dir plugins

Run a Deck and write reproducibility metadata::

    python -m huntian.cli run decks/demo_hemt.json --plugin-dir plugins --metadata out/run.json

Emit the Deck JSON Schema (what an LLM is constrained against)::

    python -m huntian.cli schema --what deck
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

from .context import SimContext
from .deck import Deck, deck_json_schema, load_deck
from .errors import HuntianError, format_issues
from .materials import MaterialDB
from .pipeline import Executor, Pipeline
from .registry import ModuleRegistry

__all__ = ["main", "build_registry", "DEFAULT_PLUGIN_DIRS"]

DEFAULT_PLUGIN_DIRS = ("plugins",)


def build_registry(
    plugin_dirs: Iterable[str | Path],
    *,
    verbose: bool = False,
) -> tuple[ModuleRegistry, list[str]]:
    """Create a registry and load every plugin found under ``plugin_dirs``."""
    registry = ModuleRegistry()
    report = registry.load_from_paths(list(plugin_dirs))
    messages = [f"loaded {record}" for record in report.loaded]
    messages += [f"failed to load {path}: {error}" for path, error in report.errors]
    if verbose:
        for message in messages:
            print(f"  {message}", file=sys.stderr)
    return registry, messages


def _resolve_plugin_dirs(args: argparse.Namespace) -> list[str]:
    dirs = list(args.plugin_dir or DEFAULT_PLUGIN_DIRS)
    if not dirs:
        dirs = list(DEFAULT_PLUGIN_DIRS)
    return dirs


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_list_plugins(args: argparse.Namespace) -> int:
    registry, _messages = build_registry(_resolve_plugin_dirs(args))
    if not len(registry):
        print("no plugins found", file=sys.stderr)
        return 1
    print(registry.render_catalog())
    print()
    print(json.dumps(registry.health(), indent=2, sort_keys=True))
    for path, error in registry.discovery_errors:
        print(f"discovery error: {path}: {error}", file=sys.stderr)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    registry, _messages = build_registry(_resolve_plugin_dirs(args))
    materials = MaterialDB.builtin()
    if args.materials:
        materials = materials.merge(MaterialDB.load(args.materials), override=True)

    try:
        deck = load_deck(args.deck)
    except HuntianError as exc:
        if args.json:
            print(json.dumps({"ok": False, "stage": "parse", "error": str(exc)}, indent=2))
        else:
            print(f"deck parse failed:\n{exc}", file=sys.stderr)
        return 2

    errors, warnings = deck.validate_all(registry, materials)
    ok = not errors
    if args.json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "deck": deck.name,
                    "errors": [str(e) for e in errors],
                    "warnings": warnings,
                },
                indent=2,
            )
        )
    else:
        print(deck.describe())
        print()
        if errors:
            print("errors:")
            print(format_issues(errors))
        if warnings:
            print("warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        print()
        print("VALID" if ok else "INVALID")
    return 0 if ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    registry, _messages = build_registry(_resolve_plugin_dirs(args))
    materials = MaterialDB.builtin()
    if args.materials:
        materials = materials.merge(MaterialDB.load(args.materials), override=True)

    try:
        deck = load_deck(args.deck)
    except HuntianError as exc:
        print(f"deck parse failed:\n{exc}", file=sys.stderr)
        return 2

    errors, warnings = deck.validate_all(registry, materials)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if errors:
        print("deck is invalid:", file=sys.stderr)
        print(format_issues(errors), file=sys.stderr)
        return 2

    pipeline = Pipeline.from_deck(deck)
    executor = Executor(registry)
    ctx = SimContext(name=deck.name, bus=registry.bus)
    result = executor.run(
        pipeline, ctx=ctx, deck=deck, materials=materials, strict=not args.keep_going
    )

    print(f"pipeline: {pipeline.name}")
    for stage in result.stages:
        marker = "ok  " if stage.status == "ok" else "FAIL"
        detail = f" - {stage.error}" if stage.error else ""
        print(
            f"  [{marker}] {stage.slot:<14} {stage.module_id}@{stage.module_version} "
            f"({stage.duration_s * 1000:.1f} ms){detail}"
        )
    print()
    print(f"status      : {result.status}")
    print(f"fingerprint : {result.fingerprint}")
    print(f"artifacts   : {', '.join(result.artifacts)}")

    for key in args.show:
        print()
        print(f"--- {key} ---")
        print(json.dumps(ctx.get(key), indent=2, default=str))

    if args.metadata:
        target = Path(args.metadata)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(ctx.snapshot(), indent=2, default=str), encoding="utf-8")
        print(f"metadata    : {target}")

    return 0 if result.status != "failed" else 1


def cmd_schema(args: argparse.Namespace) -> int:
    if args.what == "deck":
        print(json.dumps(deck_json_schema(), indent=2))
        return 0
    registry, _messages = build_registry(_resolve_plugin_dirs(args))
    print(json.dumps(registry.catalog(), indent=2))
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    registry, _messages = build_registry(_resolve_plugin_dirs(args))
    print(json.dumps(registry.health(), indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    # Shared options live on a parent parser so they can be written *after* the
    # subcommand, which is the conventional CLI shape:
    #     huntian run decks/demo_hemt.json --plugin-dir plugins
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--plugin-dir",
        action="append",
        default=None,
        metavar="PATH",
        help=f"directory of plugins (repeatable, default: {', '.join(DEFAULT_PLUGIN_DIRS)})",
    )
    common.add_argument("--verbose", "-v", action="store_true", help="log discovery detail")

    parser = argparse.ArgumentParser(
        prog="huntian",
        description="Huntian - lightweight modular TCAD core for GaN RF HEMTs",
        parents=[common],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sub = subparsers.add_parser(
        "list-plugins", help="show the module catalogue", parents=[common]
    )
    sub.set_defaults(func=cmd_list_plugins)

    sub = subparsers.add_parser("health", help="show registry health as JSON", parents=[common])
    sub.set_defaults(func=cmd_health)

    sub = subparsers.add_parser(
        "validate", help="validate a Deck against the registry", parents=[common]
    )
    sub.add_argument("deck", help="path to a .json or .yaml Deck")
    sub.add_argument("--materials", help="extra material database to merge in")
    sub.add_argument("--json", action="store_true", help="emit machine-readable results")
    sub.set_defaults(func=cmd_validate)

    sub = subparsers.add_parser("run", help="run a Deck", parents=[common])
    sub.add_argument("deck", help="path to a .json or .yaml Deck")
    sub.add_argument("--materials", help="extra material database to merge in")
    sub.add_argument("--metadata", help="write run metadata JSON here")
    sub.add_argument(
        "--keep-going",
        action="store_true",
        help="do not abort on the first non-optional stage failure",
    )
    sub.add_argument(
        "--show", action="append", default=[], metavar="KEY", help="print an artifact"
    )
    sub.set_defaults(func=cmd_run)

    sub = subparsers.add_parser(
        "schema", help="emit JSON Schema or the plugin catalogue", parents=[common]
    )
    sub.add_argument("--what", choices=("deck", "plugins"), default="deck")
    sub.set_defaults(func=cmd_schema)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.func(args))
    except HuntianError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
