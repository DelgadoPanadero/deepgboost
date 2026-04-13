# DeepGBoost — Claude Code Instructions

## Agent delegation

Use specialized agents proactively when the task matches their domain. Do not do the work yourself when an agent is better suited.

### When to spawn each agent

| Task | Agent |
|---|---|
| Analyze benchmark results, investigate underperformance vs XGBoost/GBM, propose algorithm improvements | `mathematician` |
| Implement code changes (algorithm, features, refactoring, tests) | `python-programmer` |
| API usability, docstrings, notebooks, README, CI/CD, releases | `dx` |

### Multi-agent workflows

For tasks that require both analysis and implementation, chain agents in order:

1. **mathematician** → analyzes and proposes (does NOT implement)
2. **python-programmer** → receives the mathematician's proposal and implements it
3. **dx** → updates docs, changelog, or CI if the public API or release is affected


### When NOT to delegate

- Simple one-file edits with no algorithmic risk → handle inline
- Questions about project structure or git history → handle inline
- Tasks that span all three agents (e.g. full feature end-to-end) → spawn all three sequentially
