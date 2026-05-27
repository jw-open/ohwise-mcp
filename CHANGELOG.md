# Changelog

All notable changes to `ohwise-mcp` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-05-27

Initial release.

### Added

**Code graph tools**
- `build_code_graph(repo_path, graph_type)` — extract a knowledge graph from a code repository using codebase2graph; returns node/edge summary, high-fan-in/out nodes, and a `graph_id` for subsequent calls
- `rank_code_nodes(query, graph_id, k)` — rank graph nodes by relevance to a natural language query; returns top-k nodes with content snippets for LLM context
- `search_code_graph(keyword, graph_id, kind_filter, limit)` — search nodes by keyword; optionally filter by kind (function, class, file, table, route, …)

**Document graph tools**
- `build_doc_graph(paths, graph_type)` — build a knowledge graph from files or directories; supports PDF, DOCX, Markdown, HTML, CSV, JSON, PPTX via docs2graph
- `rank_doc_nodes(query, graph_id, k)` — rank document nodes by relevance using Personalized PageRank

**OhWise Studio pipeline tools**
- `start_pipeline(user_input, agent_ids, coordinator_id)` — trigger an OhWise coordinator pipeline; returns `thread_id`
- `get_pipeline_result(thread_id, poll_seconds)` — poll a pipeline for results; returns content and artifacts on completion

**Packaging**
- Apache-2.0 license
- Python 3.10+ requirement
- Optional extras: `[code]`, `[docs]`, `[all]`
- `ohwise-mcp` CLI entry point
- Graph tools work offline; Studio tools require `OHWISE_URL` + `OHWISE_TOKEN` env vars

---

[0.1.0]: https://github.com/jw-open/ohwise-mcp/releases/tag/v0.1.0
