---
name: paper-search
description: Search for academic papers on arxiv, Papers With Code, and Semantic Scholar. Use this skill whenever the user wants to find related work, literature review, search for papers, look up research on gradient boosting, ensemble methods, deep learning, or any ML topic. Trigger on phrases like "buscar papers", "buscar artículos", "search papers", "find papers", "papers sobre", "qué dice la literatura sobre", "related work on", "arxiv search", "investigación sobre", or when the mathematician agent needs to find state-of-the-art references to improve the algorithm.
version: 1.0.0
---

# Paper Search

Search arxiv, Papers With Code, and Semantic Scholar in parallel. See `references/deepgboost-queries.md` for ready-made queries.

## API endpoints

**arxiv** (Atom XML, no auth):
`http://export.arxiv.org/api/query?search_query=<QUERY>&max_results=10&sortBy=relevance`
Query syntax: `all:`, `ti:`, `abs:`, `au:` prefixes; combine with `AND`/`OR`/`ANDNOT`; spaces as `+`.

**Papers With Code** (JSON, no auth — includes `github_link`):
`https://paperswithcode.com/api/v1/papers/?q=<QUERY>&page=1`

**Semantic Scholar** (JSON, no auth — includes `citationCount`):
`https://api.semanticscholar.org/graph/v1/paper/search?query=<QUERY>&fields=title,abstract,authors,year,citationCount,url&limit=10`

If WebFetch fails on arxiv XML, fall back to `WebSearch` with `site:arxiv.org <query>`.

## Output format

For each paper:
```
**[Title]** (Year) — Authors: First Author et al.
URL: ... | Code: ... (if available) | Citations: N (if available)
Summary: [2-3 sentences: key contribution + relevance to DeepGBoost]
```

Group by category (Direct comparisons / Related methods / Theoretical foundations). End with **Key Takeaways**: what's been tried, SOTA, and gaps DeepGBoost could fill.

## Notes
- Prefer papers from 2020+ unless searching for foundational work.
- Highlight repos from Papers With Code — useful for the python-programmer agent as implementation reference.
