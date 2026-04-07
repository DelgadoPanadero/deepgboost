---
name: kaggle-search
description: Search Kaggle for competition results, winning solutions, top notebooks, and leaderboard standings. Use this skill whenever the user wants to find how gradient boosting or other models performed in Kaggle competitions, look for winning solutions with tabular data, benchmark DeepGBoost against competition results, or find real-world datasets where ensembles were used. Trigger on phrases like "buscar en kaggle", "competiciones de kaggle", "kaggle results", "winning solution", "solución ganadora", "top kaggle notebooks", "leaderboard tabular", "qué modelos ganan en kaggle", or when the mathematician agent needs competitive benchmarks from real-world ML competitions.
version: 1.0.0
---

# Kaggle Search

Find competition results and winning solutions for tabular ML. See `references/kaggle-queries.md` for ready-made queries.

## Search methods (try in order)

**1. Kaggle CLI** (if `~/.kaggle/kaggle.json` exists):
```bash
kaggle competitions list --search "<query>" --sort-by prize
kaggle competitions leaderboard <slug> --show
kaggle kernels list --search "<query>" --sort-by voteCount --page-size 10
```

**2. WebSearch** (always available):
```
site:kaggle.com/competitions discussion "1st place solution" tabular
site:kaggle.com "<competition>" winning solution gradient boosting
```

**3. Public API** (WebFetch, no auth):
```
https://www.kaggle.com/api/v1/competitions/list?search=<query>&sortBy=prize&page=1
https://www.kaggle.com/api/v1/competitions/<slug>/leaderboard/view
```

## Output format

For each result:
```
**[Title]** (Year) — URL: https://www.kaggle.com/...
Task: tabular classification/regression | Dataset: N rows × M features
Winning model: ... | Key techniques: ...
Score: [metric] X.XXX (1st) vs X.XXX (baseline)
DeepGBoost relevance: [1-2 sentences]
```

## Notes
- Discussions tagged "1st place solution" explain *why* a method won — most informative source.
- When a neural network beats boosting, flag it: potential evidence for DeepGBoost's hybrid value.
- The Meta Kaggle public dataset tracks model popularity across all competitions historically.
