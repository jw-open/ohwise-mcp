# ohwise-mcp

[![PyPI version](https://img.shields.io/pypi/v/ohwise-mcp.svg)](https://pypi.org/project/ohwise-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/ohwise-mcp.svg)](https://pypi.org/project/ohwise-mcp/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**MCP server connecting AI coding agents to graph-native code context and multi-agent pipelines.**

`ohwise-mcp` implements the [Model Context Protocol](https://modelcontextprotocol.io/) so that any MCP-compatible AI agent (Claude Code, Claude Desktop, and others) can:

- Build and query **knowledge graphs** from code repositories via [codebase2graph](https://github.com/jw-open/codebase2graph)
- Build and query **knowledge graphs** from documents via [docs2graph](https://github.com/jw-open/docs2graph)
- Rank the most relevant nodes for any query using **Personalized PageRank**
- Trigger and poll **OhWise Studio pipelines** for multi-agent task execution

**Pure Python. No LLM dependency for graph tools. Bring your own model.**

---

## Quick start

```bash
pip install ohwise-mcp[all]
```

### Add to Claude Code

```bash
claude mcp add ohwise -- ohwise-mcp
```

Or manually in your Claude Code config (`~/.claude.json` or `.mcp.json`):

```json
{
  "mcpServers": {
    "ohwise": {
      "command": "ohwise-mcp",
      "env": {
        "OHWISE_URL": "https://your-ohwise-instance.com",
        "OHWISE_TOKEN": "your-token-here"
      }
    }
  }
}
```

> `OHWISE_URL` and `OHWISE_TOKEN` are only required for Studio pipeline tools. Graph tools work offline without them.

---

## Tools

### Code graph tools

| Tool | Description |
|------|-------------|
| `build_code_graph(repo_path, graph_type)` | Extract a knowledge graph from a code repository |
| `rank_code_nodes(query, graph_id, k)` | Rank nodes by relevance to a query — get focused code context |
| `search_code_graph(keyword, graph_id, kind_filter)` | Find nodes by keyword or kind (function, class, file, …) |

**Graph types**: `all`, `call`, `entity`, `schema`, `workflow`, `infra`, `security`, `web`, `android`, `decision`, `folder`

### Document graph tools

| Tool | Description |
|------|-------------|
| `build_doc_graph(paths, graph_type)` | Extract a knowledge graph from documents (PDF, DOCX, MD, HTML, …) |
| `rank_doc_nodes(query, graph_id, k)` | Rank document nodes by relevance — get focused document context |

### OhWise Studio tools *(requires OHWISE_URL + OHWISE_TOKEN)*

| Tool | Description |
|------|-------------|
| `start_pipeline(user_input, agent_ids)` | Trigger an OhWise coordinator pipeline |
| `get_pipeline_result(thread_id, poll_seconds)` | Poll a pipeline for results |

---

## Example usage in Claude Code

Once configured, Claude Code can use graph context automatically:

```
> What does the authentication flow look like in this repo?
```

Claude Code calls `build_code_graph` → `rank_code_nodes("authentication flow")` → receives ranked nodes with call relationships and content snippets → answers with precise, relationship-aware context.

```
> Delegate this refactoring task to the OhWise pipeline and get back the plan
```

Claude Code calls `start_pipeline` → OhWise native agents run in parallel → Claude Code receives the synthesized result.

---

## Installation variants

```bash
# Graph tools only (no OhWise backend needed)
pip install ohwise-mcp[code]    # code graphs
pip install ohwise-mcp[docs]    # document graphs
pip install ohwise-mcp[all]     # both

# Core only (Studio pipeline tools work without graph extras)
pip install ohwise-mcp
```

---

## Python API

The tools are also importable directly:

```python
from ohwise_mcp.server import mcp

# Run as MCP server
mcp.run()
```

---

## Design principles

- **No LLM dependency** — graph construction is pure static analysis
- **Offline-first** — code and document graph tools work without any server
- **Composable** — works standalone or alongside OhWise Studio
- **Model-agnostic** — any MCP-compatible agent can use these tools

---

## Related projects

| Package | What it does |
|---------|-------------|
| [codebase2graph](https://github.com/jw-open/codebase2graph) | Code repository → knowledge graph |
| [docs2graph](https://github.com/jw-open/docs2graph) | Documents → knowledge graph |
| [graph2sql](https://github.com/jw-open/graph2sql) | Schema graph → SQL context |

---

## Contributing

```bash
git clone https://github.com/jw-open/ohwise-mcp
cd ohwise-mcp
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

Apache-2.0 — see [LICENSE](LICENSE)
