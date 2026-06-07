# Changelog

All notable changes to `ohwise-mcp` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-06-07

### Added

**Code graph — context engineering**
- `get_task_context(query, repo_path, graph_type, k)` — build graph + rank nodes in one call; returns a Markdown-formatted context block ready to prepend to a prompt
- `get_node_detail(node_id, graph_id, repo_path, graph_type)` — full node content (untruncated) + all incoming/outgoing edges with neighbor summaries
- `find_impact(node_id, graph_id, repo_path, graph_type, depth)` — reverse reachability: find all callers and importers of a node, grouped by hop distance
- `trace_call_path(source_node, target_node, graph_id, repo_path, graph_type)` — BFS shortest directed path between two nodes; surfaces data/control flow
- `diff_graph(repo_path, since_commit, graph_type)` — git diff + graph overlay: nodes whose source files changed since a given commit

**Schema graph**
- `schema_context(query, ddl, k)` — Personalized PageRank over a schema graph; returns the minimal set of tables + relationships for SQL generation. Accepts raw DDL (no DB needed) or a live DB via `DATABASE_URL` env var

**Document store tools** (require `MONGO_URI` env var)
- `query_collection(collection, filter_json, database, limit)` — query a document collection with MongoDB filter syntax
- `list_collections(database)` — list collections in a database; pass `__databases__` to list all databases
- `infer_collection_schema(collection, database, sample_size)` — infer field types, examples, and coverage % by sampling documents

**Cache tools** (require `REDIS_URL` env var)
- `cache_get(key)` — get a value by key; auto-parses JSON values
- `cache_keys(pattern, count)` — non-blocking SCAN for keys matching a glob pattern
- `cache_publish(channel, message)` — publish to a pub/sub channel; returns receiver count

**Agent pipeline tools** (require `AGENT_BASE_URL` + `AGENT_TOKEN` env vars)
- `list_agents()` — list all available agents from the configured backend
- `run_agent(agent_id, user_input)` — renamed from `start_pipeline`; invoke an agent asynchronously
- `get_agent_result(thread_id, poll_seconds)` — renamed from `get_pipeline_result`; poll for result

### Changed
- FastMCP server name changed from `ohwise` to `graph-context`; all tool descriptions made provider-agnostic
- `start_pipeline` / `get_pipeline_result` renamed to `run_agent` / `get_agent_result`
- Agent backend env vars: `AGENT_BASE_URL` / `AGENT_TOKEN` (legacy `OHWISE_URL` / `OHWISE_TOKEN` still accepted as fallback)
- Extracted `_resolve_code_graph`, `_find_node`, `_rank_nodes` helpers to eliminate duplication
- New optional extras: `[mongo]`, `[redis]`, `[sql]`; `[all]` now includes all six optional deps

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
