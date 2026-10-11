---
name: Feature request
about: Propose a new plugin, a physics model, a solver backend, or a capability
title: ''
labels: enhancement
assignees: ''
---

## What problem does this solve

<!-- Describe the gap. "I want X" is weaker than "Y is currently impossible / wrong because Z". -->

## Proposed shape

<!--
If this is a plugin, say which category it falls into and what it would
`requires` / `produces`. If it needs a new Deck slot or a core change,
say so explicitly and explain why a plugin alone cannot do it.

See README.md §5 for the plugin contract and §3.6 for service slots.
-->

| | |
|---|---|
| Category (mesh / physics / solver / transport / extract / other) | |
| `requires` artifacts | |
| `produces` artifacts | |
| New third-party dependency? | yes (which) / no |
| Needs a core change? | yes (what) / no |

## Which debt does it address

<!--
`工作日志.md` §6 and README.md §10 track the known backlog. Does this
close one of those items, or is it new scope?
-->

## How would we know it works

<!--
Per CONTRIBUTING.md §4, numerical work should ship with a verification
anchor: a closed-form solution, an invariant, or a convergence order.
What would that be here?
-->
