---
name: Bug report
about: Something is wrong — a test fails, a solver misbehaves, or the CLI errors out
title: ''
labels: bug
assignees: ''
---

<!--
Before filing: please run `python run_tests.py` and report whether the
baseline is green on your machine. A red baseline changes the diagnosis a lot.
-->

## Environment

| | |
|---|---|
| OS | |
| Python version (`python --version`) | |
| NumPy version (`python -c "import numpy; print(numpy.__version__)"`) | |
| PyYAML installed? | yes / no |
| Commit or version | |

## What you ran

<!--
Paste the exact command. Note that the CLI needs PYTHONPATH set from a
source checkout:

    $env:PYTHONPATH = "python"
    python -m huntian.cli run decks/<deck>.json --plugin-dir plugins

-->

```
```

## What happened

<!-- Full traceback or the exact wrong output. Do not summarise it. -->

```
```

## What you expected

<!--
If this is a numerical issue, state the expected behaviour as an invariant,
a closed-form value, or a convergence order rather than a magic number.
-->

## Baseline check

<!-- Output of the last line of `python run_tests.py`, e.g. "ran 426 test(s): ..." -->

```
```

## Anything else

<!-- Deck file, mesh settings, parameters, or a minimal reproduction. -->
