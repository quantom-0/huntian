<!--
Thanks for the PR! 🌸 The checklist below is not bureaucracy — every item
encodes something that has actually gone wrong in this project before.
See CONTRIBUTING.md for the reasoning behind each convention.
-->

## What and why

<!-- What changed, and what problem it solves. Link the issue if there is one. -->

## Scope

- [ ] Plugins only (`plugins/`, `data/`, `decks/`)
- [ ] Core (`python/huntian/`) — if checked, explain why a plugin alone cannot do it:
- [ ] Documentation only (`*.md`, `docs/`)

## Verification

- [ ] `python run_tests.py` → **426 tests, 0 failures, 0 errors**
      <!-- Paste the last line of the output. If the count changed, say why. -->
- [ ] `python demo.py` → exit code 0
- [ ] Changed files are valid UTF-8, **no BOM, no `U+FFFD`** (CONTRIBUTING.md §5)
- [ ] New numerical code ships with a verification anchor — closed-form
      solution, invariant, or convergence order (CONTRIBUTING.md §4)

```
<!-- last line of run_tests.py -->
```

## Numerical behaviour

- [ ] This PR does **not** change any numerical result
- [ ] This PR **does** change numerical results. Then fill in:
  - affected module and new version:
  - reproducibility fingerprint before → after:
  - why the change is correct:

<!--
Pinning note: the fingerprint deliberately does NOT serialise artifact
values, so a numerical change with an unchanged version number is
invisible to it. That is why the version must be bumped explicitly.
-->

## Honesty checklist

- [ ] I did **not** remove or soften anything in `README.md` §7 (诚实的边界)
- [ ] New limitations introduced by this PR are written into `README.md` §7
- [ ] `工作日志.md` updated per CONTRIBUTING.md §6 (`README.md` §3/§10 if scope changed)
- [ ] Design docs updated if the implementation deviates from them —
      deviations get **recorded**, not silently aligned
