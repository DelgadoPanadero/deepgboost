---
name: research
description: Search academic papers (arxiv, Papers With Code, Semantic Scholar) and Kaggle competition results in parallel. Use whenever the user wants to find related work, literature review, search for papers, look up research on gradient boosting, ensemble methods, deep learning, or any ML topic — AND/OR when they want to find how models performed in Kaggle competitions, look for winning solutions with tabular data, or benchmark DeepGBoost against real-world competition results. Trigger on phrases like "buscar papers", "buscar artículos", "search papers", "find papers", "papers sobre", "qué dice la literatura sobre", "related work on", "arxiv search", "investigación sobre", "buscar en kaggle", "competiciones de kaggle", "kaggle results", "winning solution", "solución ganadora", "top kaggle notebooks", "leaderboard tabular", "qué modelos ganan en kaggle", or when the mathematician agent needs state-of-the-art references or competitive benchmarks.
version: 1.0.0
---

# Research

Search academic papers and Kaggle competition results. Run both in parallel unless the query clearly targets only one source. See `references/deepgboost-queries.md` and `references/kaggle-queries.md` for ready-made queries.

---

## Part 1 — Academic Papers

Search arxiv, Papers With Code, and Semantic Scholar in parallel.

### API endpoints

**arxiv** (Atom XML, no auth):
`http://export.arxiv.org/api/query?search_query=<QUERY>&max_results=10&sortBy=relevance`
Query syntax: `all:`, `ti:`, `abs:`, `au:` prefixes; combine with `AND`/`OR`/`ANDNOT`; spaces as `+`.

**Papers With Code** (JSON, no auth — includes `github_link`):
`https://paperswithcode.com/api/v1/papers/?q=<QUERY>&page=1`

**Semantic Scholar** (JSON, no auth — includes `citationCount`):
`https://api.semanticscholar.org/graph/v1/paper/search?query=<QUERY>&fields=title,abstract,authors,year,citationCount,url&limit=10`

If WebFetch fails on arxiv XML, fall back to `WebSearch` with `site:arxiv.org <query>`.

### Output format

For each paper:
```
**[Title]** (Year) — Authors: First Author et al.
URL: ... | Code: ... (if available) | Citations: N (if available)
Summary: [2-3 sentences: key contribution + relevance to DeepGBoost]
```

Group by category (Direct comparisons / Related methods / Theoretical foundations). End with **Key Takeaways**: what's been tried, SOTA, and gaps DeepGBoost could fill.

### Notes
- Prefer papers from 2020+ unless searching for foundational work.
- Highlight repos from Papers With Code — useful for the python-programmer agent as implementation reference.

---

## Part 2 — Kaggle Competitions

Find competition results and winning solutions for tabular ML.

### Search methods (try in order)

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

### Output format

For each result:
```
**[Title]** (Year) — URL: https://www.kaggle.com/...
Task: tabular classification/regression | Dataset: N rows × M features
Winning model: ... | Key techniques: ...
Score: [metric] X.XXX (1st) vs X.XXX (baseline)
DeepGBoost relevance: [1-2 sentences]
```

### Notes
- Discussions tagged "1st place solution" explain *why* a method won — most informative source.
- When a neural network beats boosting, flag it: potential evidence for DeepGBoost's hybrid value.
- The Meta Kaggle public dataset tracks model popularity across all competitions historically.
