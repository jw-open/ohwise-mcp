"""
Graph-native context tools exposed as MCP tools.

Standalone tools (no backend required):
  Code graph:
    build_code_graph        — extract knowledge graph from a code repository
    rank_code_nodes         — rank nodes by relevance to a query
    search_code_graph       — keyword search across graph nodes
    get_task_context        — build + rank in one call, returns formatted context block
    get_node_detail         — full node content + all neighborhood edges
    find_impact             — nodes that depend on a given node (reverse reachability)
    trace_call_path         — shortest directed path between two nodes
    diff_graph              — nodes affected by recent git changes

  Document graph:
    build_doc_graph         — extract knowledge graph from documents
    rank_doc_nodes          — rank document nodes by relevance

  Schema graph:
    schema_context          — relevant schema subgraph for a natural language query

Document store tools (require MONGO_URI env var):
    query_collection        — query a document collection
    list_collections        — list collections in a database
    infer_collection_schema — infer field types from sampled documents

Cache tools (require REDIS_URL env var):
    cache_get               — get a value by key
    cache_keys              — list keys matching a pattern
    cache_publish           — publish a message to a channel

Agent pipeline tools (require AGENT_BASE_URL + AGENT_TOKEN env vars):
    list_agents             — list available agents
    run_agent               — trigger an agent run asynchronously
    get_agent_result        — poll an agent run for results
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "graph-context",
    instructions=(
        "Graph-native context and analysis tools for AI coding agents. "
        "Use build_code_graph + rank_code_nodes (or get_task_context) to retrieve "
        "relationship-aware code context before editing. "
        "Use get_node_detail / find_impact / trace_call_path for precise navigation. "
        "Use diff_graph to see what changed since a commit. "
        "Use build_doc_graph + rank_doc_nodes for document retrieval. "
        "Use schema_context to get relevant schema tables for SQL generation. "
        "Use query_collection / list_collections / infer_collection_schema for document store access. "
        "Use cache_get / cache_keys for cache inspection. "
        "Use run_agent to delegate complex tasks to a remote agent pipeline."
    ),
)

# ---------------------------------------------------------------------------
# In-memory graph cache — reused across rank/search calls within a session
# ---------------------------------------------------------------------------
_graph_cache: dict[str, Any] = {}


def _graph_cache_key(path: str, graph_type: str) -> str:
    return f"{Path(path).resolve()}::{graph_type}"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_code_graph(
    graph_id: Optional[str],
    repo_path: Optional[str],
    graph_type: str,
) -> tuple[Any, Optional[str]]:
    """Return (graph, error). error is None on success."""
    try:
        from code2graph import build_graph  # type: ignore
    except ImportError:
        return None, "code2graph not installed. Run: pip install code2graph"

    if graph_id and graph_id in _graph_cache:
        return _graph_cache[graph_id], None

    if repo_path:
        path = str(Path(repo_path).resolve())
        key = _graph_cache_key(path, graph_type)
        if key in _graph_cache:
            return _graph_cache[key], None
        try:
            graph = build_graph(path, graph_type=graph_type)
            _graph_cache[key] = graph
            return graph, None
        except Exception as exc:
            return None, str(exc)

    return None, "Provide graph_id (from build_code_graph) or repo_path."


def _find_node(graph: Any, query_str: str) -> Optional[Any]:
    """Find a node by exact ID or label substring (first match)."""
    node = graph.nodes.get(query_str)
    if node:
        return node
    kw = query_str.lower()
    for n in graph.nodes.values():
        if kw in n.label.lower() or kw in n.id.lower():
            return n
    return None


def _agent_base_url() -> str:
    return (os.environ.get("AGENT_BASE_URL") or os.environ.get("OHWISE_URL", "")).rstrip("/")


def _agent_token() -> str:
    return os.environ.get("AGENT_TOKEN") or os.environ.get("OHWISE_TOKEN", "")


def _agent_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_agent_token()}", "Content-Type": "application/json"}


def _mongo_client() -> tuple[Any, Optional[str]]:
    """Return (client, error). error is None on success."""
    try:
        from pymongo import MongoClient  # type: ignore
    except ImportError:
        return None, "pymongo not installed. Run: pip install pymongo"
    uri = os.environ.get("MONGO_URI", "")
    if not uri:
        return None, "MONGO_URI environment variable not set."
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        return client, None
    except Exception as exc:
        return None, str(exc)


def _redis_client() -> tuple[Any, Optional[str]]:
    """Return (redis_client, error). error is None on success."""
    try:
        import redis as redis_lib  # type: ignore
    except ImportError:
        return None, "redis not installed. Run: pip install redis"
    url = os.environ.get("REDIS_URL", "")
    if not url:
        return None, "REDIS_URL environment variable not set."
    try:
        r = redis_lib.from_url(url, socket_connect_timeout=5)
        return r, None
    except Exception as exc:
        return None, str(exc)


def _rank_nodes(graph: Any, query: str, k: int) -> list[tuple[float, Any]]:
    """Score and rank graph nodes by token overlap + structural centrality."""
    query_tokens = set(query.lower().split())
    in_count: dict[str, int] = {}
    out_count: dict[str, int] = {}
    for e in graph.edges.values():
        out_count[e.from_id] = out_count.get(e.from_id, 0) + 1
        in_count[e.to_id] = in_count.get(e.to_id, 0) + 1

    scored: list[tuple[float, Any]] = []
    for node in graph.nodes.values():
        text = (
            f"{node.label} {node.id} {node.content or ''} "
            f"{' '.join(str(v) for v in node.attributes.values())}"
        ).lower()
        overlap = sum(1 for t in query_tokens if t in text)
        centrality = in_count.get(node.id, 0) + out_count.get(node.id, 0)
        score = overlap * 3 + centrality * 0.1
        if overlap > 0 or centrality > 2:
            scored.append((score, node))

    scored.sort(key=lambda x: -x[0])
    return scored[:k]


# ---------------------------------------------------------------------------
# Code graph — build / rank / search
# ---------------------------------------------------------------------------

@mcp.tool()
def build_code_graph(
    repo_path: str,
    graph_type: str = "all",
) -> str:
    """
    Build a knowledge graph from a code repository.

    Returns a JSON summary: node/edge counts, nodes by kind, high-fan-in nodes
    (core utilities) and high-fan-out nodes (orchestrators), and a graph_id
    to reuse in subsequent calls without rebuilding.

    Parameters
    ----------
    repo_path : str
        Absolute or relative path to the repository root.
    graph_type : str
        One of: all, call, entity, schema, workflow, infra, security, web,
        android, decision, folder. Default: all.
    """
    try:
        from code2graph import build_graph  # type: ignore
    except ImportError:
        return json.dumps({"error": "code2graph not installed. Run: pip install code2graph"})

    path = str(Path(repo_path).resolve())
    key = _graph_cache_key(path, graph_type)

    try:
        graph = build_graph(path, graph_type=graph_type)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    _graph_cache[key] = graph

    all_nodes = list(graph.nodes.values())
    all_edges = list(graph.edges.values())

    nodes_by_kind: dict[str, int] = {}
    for n in all_nodes:
        kind = n.attributes.get("kind", "unknown")
        nodes_by_kind[kind] = nodes_by_kind.get(kind, 0) + 1

    edge_labels: dict[str, int] = {}
    for e in all_edges:
        edge_labels[e.label] = edge_labels.get(e.label, 0) + 1

    in_count: dict[str, int] = {}
    out_count: dict[str, int] = {}
    for e in all_edges:
        out_count[e.from_id] = out_count.get(e.from_id, 0) + 1
        in_count[e.to_id] = in_count.get(e.to_id, 0) + 1

    id_to_label = {n.id: n.label for n in all_nodes}
    top_fan_in = sorted(in_count.items(), key=lambda x: -x[1])[:5]
    top_fan_out = sorted(out_count.items(), key=lambda x: -x[1])[:5]

    return json.dumps({
        "graph_id": key,
        "repo_path": path,
        "graph_type": graph_type,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "nodes_by_kind": nodes_by_kind,
        "edges_by_label": edge_labels,
        "high_fan_in": [
            {"id": nid, "label": id_to_label.get(nid, nid), "incoming": cnt}
            for nid, cnt in top_fan_in
        ],
        "high_fan_out": [
            {"id": nid, "label": id_to_label.get(nid, nid), "outgoing": cnt}
            for nid, cnt in top_fan_out
        ],
        "tip": f"Call get_task_context(query='...', repo_path='{path}') for a formatted context block.",
    }, indent=2)


@mcp.tool()
def rank_code_nodes(
    query: str,
    graph_id: Optional[str] = None,
    repo_path: Optional[str] = None,
    graph_type: str = "all",
    k: int = 10,
) -> str:
    """
    Rank code graph nodes by relevance to a query.

    Returns the top-k most relevant nodes with content snippets —
    use as focused LLM context before editing code.

    Parameters
    ----------
    query : str
        Natural language query (e.g. "authentication flow", "database connection").
    graph_id : str, optional
        ID returned by build_code_graph. If omitted, repo_path must be provided.
    repo_path : str, optional
        Build graph from this path if graph_id is not provided.
    graph_type : str
        Graph type when building from repo_path. Default: all.
    k : int
        Number of top nodes to return. Default: 10.
    """
    graph, err = _resolve_code_graph(graph_id, repo_path, graph_type)
    if err:
        return json.dumps({"error": err})

    top = _rank_nodes(graph, query, k)
    results = [
        {
            "id": node.id,
            "label": node.label,
            "kind": node.attributes.get("kind", ""),
            "file": node.attributes.get("file", ""),
            "line": node.attributes.get("line", ""),
            "score": round(score, 2),
            "content": (node.content or "")[:400],
        }
        for score, node in top
    ]

    return json.dumps({
        "query": query,
        "top_nodes": results,
        "total_matched": len(top),
    }, indent=2)


@mcp.tool()
def search_code_graph(
    keyword: str,
    graph_id: Optional[str] = None,
    repo_path: Optional[str] = None,
    graph_type: str = "all",
    kind_filter: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    Search code graph nodes by keyword, optionally filtered by kind.

    Useful for finding all functions, classes, or files matching a name pattern.

    Parameters
    ----------
    keyword : str
        Text to search in node labels, IDs, and content.
    graph_id : str, optional
        ID returned by build_code_graph.
    repo_path : str, optional
        Build graph from this path if graph_id is not provided.
    graph_type : str
        Graph type. Default: all.
    kind_filter : str, optional
        Filter to nodes of this kind (function, class, file, table, route, etc.).
    limit : int
        Max results. Default: 20.
    """
    graph, err = _resolve_code_graph(graph_id, repo_path, graph_type)
    if err:
        return json.dumps({"error": err})

    kw = keyword.lower()
    results = []
    for node in graph.nodes.values():
        if kind_filter and node.attributes.get("kind", "") != kind_filter:
            continue
        searchable = f"{node.label} {node.id} {node.content or ''}".lower()
        if kw in searchable:
            results.append({
                "id": node.id,
                "label": node.label,
                "kind": node.attributes.get("kind", ""),
                "file": node.attributes.get("file", ""),
                "line": node.attributes.get("line", ""),
                "content_preview": (node.content or "")[:200],
            })
        if len(results) >= limit:
            break

    return json.dumps({"keyword": keyword, "kind_filter": kind_filter, "results": results}, indent=2)


# ---------------------------------------------------------------------------
# Code graph — new context engineering tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_task_context(
    query: str,
    repo_path: str,
    graph_type: str = "all",
    k: int = 15,
) -> str:
    """
    Build a code graph and return a formatted context block for a task in one call.

    Combines build_code_graph + rank_code_nodes and returns a Markdown-formatted
    context block ready to prepend to a prompt. Caches the graph for reuse.

    Parameters
    ----------
    query : str
        Natural language description of the task (e.g. "auth token validation").
    repo_path : str
        Path to the repository root.
    graph_type : str
        Graph type. Default: all.
    k : int
        Number of top nodes to include. Default: 15.
    """
    graph, err = _resolve_code_graph(None, repo_path, graph_type)
    if err:
        return json.dumps({"error": err})

    top = _rank_nodes(graph, query, k)
    path = str(Path(repo_path).resolve())
    key = _graph_cache_key(path, graph_type)

    lines = [
        f"# Context for: {query}",
        f"# Repo: {path}  graph_type: {graph_type}  graph_id: {key}",
        "",
    ]
    for rank_i, (score, node) in enumerate(top, 1):
        kind = node.attributes.get("kind", "node")
        file_ = node.attributes.get("file", "")
        line_ = node.attributes.get("line", "")
        loc = f"  [{file_}:{line_}]" if file_ else ""
        content = (node.content or "")[:600]
        lines.append(f"## {rank_i}. {node.label}  ({kind}){loc}  score={score:.1f}")
        if content:
            lines.append(f"```\n{content}\n```")
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
def get_node_detail(
    node_id: str,
    graph_id: Optional[str] = None,
    repo_path: Optional[str] = None,
    graph_type: str = "all",
) -> str:
    """
    Return the full content and neighborhood of a specific graph node.

    Includes complete (untruncated) content, all incoming and outgoing edges,
    and a summary of each neighbor node.

    Parameters
    ----------
    node_id : str
        Exact node ID, or a label substring (first match used).
    graph_id : str, optional
        Graph ID from build_code_graph.
    repo_path : str, optional
        Build graph from this path if graph_id is not provided.
    graph_type : str
        Graph type. Default: all.
    """
    graph, err = _resolve_code_graph(graph_id, repo_path, graph_type)
    if err:
        return json.dumps({"error": err})

    node = _find_node(graph, node_id)
    if node is None:
        return json.dumps({"error": f"Node '{node_id}' not found."})

    incoming = []
    outgoing = []
    for e in graph.edges.values():
        if e.to_id == node.id:
            src = graph.nodes.get(e.from_id)
            incoming.append({
                "from_id": e.from_id,
                "from_label": src.label if src else e.from_id,
                "from_kind": src.attributes.get("kind", "") if src else "",
                "edge_label": e.label,
            })
        elif e.from_id == node.id:
            tgt = graph.nodes.get(e.to_id)
            outgoing.append({
                "to_id": e.to_id,
                "to_label": tgt.label if tgt else e.to_id,
                "to_kind": tgt.attributes.get("kind", "") if tgt else "",
                "edge_label": e.label,
            })

    return json.dumps({
        "id": node.id,
        "label": node.label,
        "kind": node.attributes.get("kind", ""),
        "file": node.attributes.get("file", ""),
        "line": node.attributes.get("line", ""),
        "attributes": node.attributes,
        "content": node.content or "",
        "degree": {"in": len(incoming), "out": len(outgoing)},
        "incoming_edges": incoming,
        "outgoing_edges": outgoing,
    }, indent=2)


@mcp.tool()
def find_impact(
    node_id: str,
    graph_id: Optional[str] = None,
    repo_path: Optional[str] = None,
    graph_type: str = "all",
    depth: int = 2,
) -> str:
    """
    Find all nodes that depend on a given node (reverse reachability).

    Traverses the graph backwards from node_id to discover callers, importers,
    and anything else affected by changes to that node. Results are grouped by
    hop distance.

    Parameters
    ----------
    node_id : str
        Node ID or label substring to start from.
    graph_id : str, optional
        Graph ID from build_code_graph.
    repo_path : str, optional
        Build graph from this path if graph_id is not provided.
    graph_type : str
        Graph type. Default: all.
    depth : int
        Number of reverse hops to traverse. Default: 2.
    """
    graph, err = _resolve_code_graph(graph_id, repo_path, graph_type)
    if err:
        return json.dumps({"error": err})

    start = _find_node(graph, node_id)
    if start is None:
        return json.dumps({"error": f"Node '{node_id}' not found."})

    # Build reverse adjacency: to_id -> [from_id, ...]
    reverse_adj: dict[str, list[str]] = {}
    for e in graph.edges.values():
        reverse_adj.setdefault(e.to_id, []).append(e.from_id)

    visited = {start.id}
    frontier = {start.id}
    layers: list[list[dict]] = []

    for _ in range(depth):
        next_frontier: set[str] = set()
        for nid in frontier:
            for caller_id in reverse_adj.get(nid, []):
                if caller_id not in visited:
                    visited.add(caller_id)
                    next_frontier.add(caller_id)
        if not next_frontier:
            break
        layer = []
        for nid in next_frontier:
            n = graph.nodes.get(nid)
            if n:
                layer.append({
                    "id": n.id,
                    "label": n.label,
                    "kind": n.attributes.get("kind", ""),
                    "file": n.attributes.get("file", ""),
                    "line": n.attributes.get("line", ""),
                })
        layers.append(layer)
        frontier = next_frontier

    return json.dumps({
        "node": {"id": start.id, "label": start.label, "kind": start.attributes.get("kind", "")},
        "total_affected": sum(len(layer) for layer in layers),
        "affected_by_depth": layers,
        "tip": "These nodes call or import the target — review before editing.",
    }, indent=2)


@mcp.tool()
def trace_call_path(
    source_node: str,
    target_node: str,
    graph_id: Optional[str] = None,
    repo_path: Optional[str] = None,
    graph_type: str = "all",
) -> str:
    """
    Find the shortest directed call path between two nodes.

    Uses BFS on the directed graph. Useful for understanding how data or control
    flows between two parts of the codebase before refactoring.

    Parameters
    ----------
    source_node : str
        Starting node ID or label substring.
    target_node : str
        Destination node ID or label substring.
    graph_id : str, optional
        Graph ID from build_code_graph.
    repo_path : str, optional
        Build graph from this path if graph_id is not provided.
    graph_type : str
        Graph type. Default: all.
    """
    graph, err = _resolve_code_graph(graph_id, repo_path, graph_type)
    if err:
        return json.dumps({"error": err})

    src = _find_node(graph, source_node)
    if src is None:
        return json.dumps({"error": f"Source node '{source_node}' not found."})
    tgt = _find_node(graph, target_node)
    if tgt is None:
        return json.dumps({"error": f"Target node '{target_node}' not found."})

    if src.id == tgt.id:
        n = graph.nodes.get(src.id)
        return json.dumps({"path": [{"id": n.id, "label": n.label}], "path_length": 0})

    # Forward adjacency: from_id -> [(to_id, edge_label), ...]
    fwd_adj: dict[str, list[tuple[str, str]]] = {}
    for e in graph.edges.values():
        fwd_adj.setdefault(e.from_id, []).append((e.to_id, e.label))

    # BFS — track path as list of (node_id, edge_label_to_next)
    queue: deque[tuple[str, list[tuple[str, str]]]] = deque()
    queue.append((src.id, []))
    visited = {src.id}

    while queue:
        current_id, path_so_far = queue.popleft()
        for next_id, edge_label in fwd_adj.get(current_id, []):
            new_path = path_so_far + [(next_id, edge_label)]
            if next_id == tgt.id:
                # Reconstruct full path
                full_path = []
                prev_id = src.id
                for step_to, step_edge in new_path:
                    prev_node = graph.nodes.get(prev_id)
                    full_path.append({
                        "id": prev_id,
                        "label": prev_node.label if prev_node else prev_id,
                        "kind": prev_node.attributes.get("kind", "") if prev_node else "",
                        "edge_to_next": step_edge,
                    })
                    prev_id = step_to
                tgt_node = graph.nodes.get(tgt.id)
                full_path.append({
                    "id": tgt.id,
                    "label": tgt_node.label if tgt_node else tgt.id,
                    "kind": tgt_node.attributes.get("kind", "") if tgt_node else "",
                })
                return json.dumps({
                    "source": source_node,
                    "target": target_node,
                    "path_length": len(full_path) - 1,
                    "path": full_path,
                }, indent=2)
            if next_id not in visited:
                visited.add(next_id)
                queue.append((next_id, new_path))

    return json.dumps({
        "source": source_node,
        "target": target_node,
        "path": None,
        "note": "No directed path found between these nodes.",
    }, indent=2)


@mcp.tool()
def diff_graph(
    repo_path: str,
    since_commit: str,
    graph_type: str = "all",
) -> str:
    """
    Show graph nodes whose source files changed since a given git commit.

    Runs git diff to find changed files, builds (or reuses) the code graph,
    and returns all nodes whose file attribute was modified.

    Parameters
    ----------
    repo_path : str
        Path to the repository root (must be a git repo).
    since_commit : str
        Git ref to diff from (commit hash, branch name, HEAD~N, tag, etc.).
    graph_type : str
        Graph type. Default: all.
    """
    path = str(Path(repo_path).resolve())

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", since_commit, "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return json.dumps({"error": f"git diff failed: {result.stderr.strip()}"})
        changed_files_abs = {
            str((Path(path) / f.strip()).resolve())
            for f in result.stdout.strip().splitlines()
            if f.strip()
        }
    except Exception as exc:
        return json.dumps({"error": f"git error: {exc}"})

    if not changed_files_abs:
        return json.dumps({
            "since_commit": since_commit,
            "changed_files": [],
            "affected_nodes": [],
            "note": "No files changed since that commit.",
        })

    graph, err = _resolve_code_graph(None, repo_path, graph_type)
    if err:
        return json.dumps({"error": err})

    affected = []
    for node in graph.nodes.values():
        node_file = node.attributes.get("file", "")
        if not node_file:
            continue
        nf = Path(node_file)
        node_abs = str(nf.resolve()) if nf.is_absolute() else str((Path(path) / node_file).resolve())
        if node_abs in changed_files_abs:
            affected.append({
                "id": node.id,
                "label": node.label,
                "kind": node.attributes.get("kind", ""),
                "file": node_file,
                "line": node.attributes.get("line", ""),
            })

    changed_rel = sorted(
        str(Path(f).relative_to(path)) if f.startswith(path) else f
        for f in changed_files_abs
    )

    return json.dumps({
        "since_commit": since_commit,
        "changed_files": changed_rel,
        "affected_node_count": len(affected),
        "affected_nodes": affected,
    }, indent=2)


# ---------------------------------------------------------------------------
# Document graph tools
# ---------------------------------------------------------------------------

@mcp.tool()
def build_doc_graph(
    paths: str,
    graph_type: str = "knowledge",
) -> str:
    """
    Build a knowledge graph from documents.

    Accepts a single file path, directory path, or comma-separated list of paths.
    Supports PDF, DOCX, Markdown, HTML, CSV, JSON, PPTX, and plain text.

    Parameters
    ----------
    paths : str
        File path, directory, or comma-separated list of paths.
    graph_type : str
        One of: knowledge, schema, decision. Default: knowledge.
    """
    try:
        from docs2graph import (  # type: ignore
            extract_decision_graph,
            extract_knowledge_graph,
            extract_schema_graph,
            load_document,
        )
    except ImportError:
        return json.dumps({"error": "docs2graph not installed. Run: pip install docs2graph"})

    path_list = [p.strip() for p in paths.split(",")]
    texts: list[str] = []
    loaded_paths: list[str] = []
    errors: list[str] = []

    for p in path_list:
        resolved = Path(p).resolve()
        if resolved.is_dir():
            for f in resolved.rglob("*"):
                if f.is_file() and f.suffix.lower() in {".md", ".txt", ".pdf", ".docx", ".html", ".csv", ".json"}:
                    try:
                        texts.append(load_document(str(f)))
                        loaded_paths.append(str(f))
                    except Exception as e:
                        errors.append(f"{f}: {e}")
        elif resolved.is_file():
            try:
                texts.append(load_document(str(resolved)))
                loaded_paths.append(str(resolved))
            except Exception as e:
                errors.append(f"{resolved}: {e}")
        else:
            errors.append(f"Not found: {p}")

    if not texts:
        return json.dumps({"error": "No documents loaded.", "errors": errors})

    combined = "\n\n".join(texts)

    try:
        if graph_type == "schema":
            graph = extract_schema_graph(combined)
        elif graph_type == "decision":
            graph = extract_decision_graph(combined)
        else:
            graph = extract_knowledge_graph(combined)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    key = f"doc::{','.join(loaded_paths)}::{graph_type}"
    _graph_cache[key] = graph

    doc_nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    doc_edges = graph.get("edges", []) if isinstance(graph, dict) else []

    nodes_by_kind: dict[str, int] = {}
    for n in doc_nodes:
        kind = (n.get("attributes") or {}).get("kind", "unknown")
        nodes_by_kind[kind] = nodes_by_kind.get(kind, 0) + 1

    return json.dumps({
        "graph_id": key,
        "graph_type": graph_type,
        "files_loaded": len(loaded_paths),
        "node_count": len(doc_nodes),
        "edge_count": len(doc_edges),
        "nodes_by_kind": nodes_by_kind,
        "errors": errors,
        "tip": f"Call rank_doc_nodes(graph_id='{key}', query='...') to retrieve relevant context.",
    }, indent=2)


@mcp.tool()
def rank_doc_nodes(
    query: str,
    graph_id: str,
    k: int = 8,
) -> str:
    """
    Rank document graph nodes by relevance to a query using Personalized PageRank.

    Returns the top-k most relevant nodes with content — use as LLM context
    for document question answering.

    Parameters
    ----------
    query : str
        Natural language question or topic.
    graph_id : str
        ID returned by build_doc_graph.
    k : int
        Number of top nodes to return. Default: 8.
    """
    try:
        from docs2graph import personalized_page_rank  # type: ignore
    except ImportError:
        return json.dumps({"error": "docs2graph not installed. Run: pip install docs2graph"})

    graph = _graph_cache.get(graph_id)
    if graph is None:
        return json.dumps({"error": f"Graph '{graph_id}' not found. Call build_doc_graph first."})

    try:
        ranked = personalized_page_rank(query, graph, k=k)
        results = [
            {
                "id": node["id"],
                "label": node["label"],
                "kind": (node.get("attributes") or {}).get("kind", ""),
                "content": (node.get("content") or "")[:600],
            }
            for node in ranked.get("nodes", [])
        ]
        return json.dumps({"query": query, "top_nodes": results}, indent=2)
    except Exception:
        # Fallback: token overlap ranking
        query_tokens = set(query.lower().split())
        doc_nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
        scored = []
        for node in doc_nodes:
            text = f"{node['label']} {node.get('content') or ''}".lower()
            score = sum(1 for t in query_tokens if t in text)
            if score > 0:
                scored.append((score, node))
        scored.sort(key=lambda x: -x[0])
        results = [
            {
                "id": n["id"],
                "label": n["label"],
                "kind": (n.get("attributes") or {}).get("kind", ""),
                "content": (n.get("content") or "")[:600],
            }
            for _, n in scored[:k]
        ]
        return json.dumps({"query": query, "top_nodes": results, "note": "fallback ranking used"}, indent=2)


# ---------------------------------------------------------------------------
# Schema graph tool
# ---------------------------------------------------------------------------

@mcp.tool()
def schema_context(
    query: str,
    ddl: Optional[str] = None,
    k: int = 3,
) -> str:
    """
    Return the most relevant schema tables and relationships for a natural language query.

    Uses Personalized PageRank to surface the minimal set of tables needed to
    answer the query — ideal context for SQL generation.

    Provide schema via one of:
    - ddl parameter: raw SQL CREATE TABLE statements (no live DB required)
    - DATABASE_URL env var: SQLAlchemy connection string for live DB introspection

    Parameters
    ----------
    query : str
        Natural language question (e.g. "monthly revenue by customer").
    ddl : str, optional
        SQL DDL string (CREATE TABLE ...). If omitted, DATABASE_URL env var is used.
    k : int
        Number of top seed tables. Default: 3.
    """
    try:
        from graph2sql import SchemaGraph  # type: ignore
    except ImportError:
        return json.dumps({"error": "graph2sql not installed. Run: pip install graph2sql"})

    try:
        if ddl:
            graph = SchemaGraph.from_ddl(ddl)
        else:
            db_url = os.environ.get("DATABASE_URL", "")
            if not db_url:
                return json.dumps({
                    "error": "Provide the ddl parameter or set the DATABASE_URL environment variable."
                })
            try:
                from sqlalchemy import create_engine  # type: ignore
            except ImportError:
                return json.dumps({"error": "sqlalchemy not installed. Run: pip install sqlalchemy"})
            engine = create_engine(db_url)
            graph = SchemaGraph.from_sqlalchemy(engine)
    except Exception as exc:
        return json.dumps({"error": f"Schema build failed: {exc}"})

    try:
        result = graph.rank(query, k=k)
    except Exception as exc:
        return json.dumps({"error": f"Ranking failed: {exc}"})

    nodes = result.get("nodes", [])
    edges = result.get("edges", [])

    context_lines = [f"# Schema context for: {query}", ""]
    for node in nodes:
        score = node.get("score")
        score_str = f"  [relevance={score:.4f}]" if score is not None else ""
        context_lines.append(f"## {node['label']}{score_str}")
        if node.get("content"):
            context_lines.append(f"```sql\n{node['content']}\n```")
        context_lines.append("")

    if edges:
        context_lines.append("## Relationships")
        for edge in edges:
            context_lines.append(
                f"- {edge.get('from', '')} --[{edge.get('label', '')}]--> {edge.get('to', '')}"
            )
        context_lines.append("")

    return json.dumps({
        "query": query,
        "context": "\n".join(context_lines),
        "tables": [n["label"] for n in nodes],
        "relationship_count": len(edges),
    }, indent=2)


# ---------------------------------------------------------------------------
# Document store tools
# ---------------------------------------------------------------------------

@mcp.tool()
def query_collection(
    collection: str,
    filter_json: str = "{}",
    database: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    Query a document collection and return matching documents.

    Requires MONGO_URI environment variable (standard MongoDB connection string).

    Parameters
    ----------
    collection : str
        Collection name.
    filter_json : str
        JSON filter object (MongoDB query syntax). Default: {} (all documents).
    database : str, optional
        Database name. If omitted, uses the database specified in MONGO_URI.
    limit : int
        Maximum documents to return. Default: 20.
    """
    client, err = _mongo_client()
    if err:
        return json.dumps({"error": err})

    try:
        filter_doc = json.loads(filter_json)
    except json.JSONDecodeError as exc:
        client.close()
        return json.dumps({"error": f"Invalid filter JSON: {exc}"})

    try:
        db = client[database] if database else client.get_default_database()
        docs = list(db[collection].find(filter_doc, limit=limit))
        for doc in docs:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
        return json.dumps({"collection": collection, "count": len(docs), "documents": docs}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    finally:
        client.close()


@mcp.tool()
def list_collections(
    database: Optional[str] = None,
) -> str:
    """
    List all collections in a database, or list all available databases.

    Requires MONGO_URI environment variable.

    Parameters
    ----------
    database : str, optional
        Database name to list collections in.
        Pass "__databases__" to list all databases instead.
        If omitted, uses the database specified in MONGO_URI.
    """
    client, err = _mongo_client()
    if err:
        return json.dumps({"error": err})

    try:
        if database == "__databases__":
            dbs = client.list_database_names()
            return json.dumps({"databases": sorted(dbs)}, indent=2)
        db = client[database] if database else client.get_default_database()
        collections = db.list_collection_names()
        return json.dumps({"database": db.name, "collections": sorted(collections)}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    finally:
        client.close()


@mcp.tool()
def infer_collection_schema(
    collection: str,
    database: Optional[str] = None,
    sample_size: int = 20,
) -> str:
    """
    Infer the field schema of a collection by sampling documents.

    Returns field names, inferred types, example values, and coverage percentage
    (how often each field is present across sampled documents).

    Requires MONGO_URI environment variable.

    Parameters
    ----------
    collection : str
        Collection name.
    database : str, optional
        Database name. If omitted, uses the database specified in MONGO_URI.
    sample_size : int
        Number of documents to sample. Default: 20.
    """
    client, err = _mongo_client()
    if err:
        return json.dumps({"error": err})

    try:
        db = client[database] if database else client.get_default_database()
        docs = list(db[collection].find({}, limit=sample_size))
    except Exception as exc:
        client.close()
        return json.dumps({"error": str(exc)})
    finally:
        client.close()

    if not docs:
        return json.dumps({"collection": collection, "fields": {}, "note": "No documents found."})

    schema: dict[str, dict] = {}
    for doc in docs:
        for k, v in doc.items():
            if k == "_id":
                continue
            type_name = type(v).__name__
            if k not in schema:
                schema[k] = {"type": type_name, "example": str(v)[:100], "present_in": 0}
            elif schema[k]["type"] != type_name:
                schema[k]["type"] = "mixed"
            schema[k]["present_in"] += 1

    for k in schema:
        schema[k]["coverage_pct"] = round(schema[k]["present_in"] / len(docs) * 100, 1)
        del schema[k]["present_in"]

    return json.dumps({
        "collection": collection,
        "sampled": len(docs),
        "field_count": len(schema),
        "fields": schema,
    }, indent=2)


# ---------------------------------------------------------------------------
# Cache tools
# ---------------------------------------------------------------------------

@mcp.tool()
def cache_get(key: str) -> str:
    """
    Get a value from the cache by key.

    Requires REDIS_URL environment variable.

    Parameters
    ----------
    key : str
        Cache key to retrieve.
    """
    r, err = _redis_client()
    if err:
        return json.dumps({"error": err})

    try:
        value = r.get(key)
        if value is None:
            return json.dumps({"key": key, "value": None, "exists": False})
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                value = value.hex()
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass
        return json.dumps({"key": key, "value": value, "exists": True}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def cache_keys(
    pattern: str = "*",
    count: int = 50,
) -> str:
    """
    List cache keys matching a glob pattern.

    Uses SCAN to avoid blocking the server. Requires REDIS_URL environment variable.

    Parameters
    ----------
    pattern : str
        Glob pattern to match keys (e.g. "session:*", "user:42:*"). Default: "*".
    count : int
        Maximum number of keys to return. Default: 50.
    """
    r, err = _redis_client()
    if err:
        return json.dumps({"error": err})

    try:
        keys = []
        for key in r.scan_iter(match=pattern, count=100):
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            keys.append(key)
            if len(keys) >= count:
                break
        return json.dumps({"pattern": pattern, "count": len(keys), "keys": sorted(keys)}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def cache_publish(
    channel: str,
    message: str,
) -> str:
    """
    Publish a message to a pub/sub channel.

    Returns the number of subscribers that received the message.
    Requires REDIS_URL environment variable.

    Parameters
    ----------
    channel : str
        Channel name to publish to.
    message : str
        Message payload (plain string or JSON string).
    """
    r, err = _redis_client()
    if err:
        return json.dumps({"error": err})

    try:
        receivers = r.publish(channel, message)
        return json.dumps({"channel": channel, "message": message, "receivers": receivers}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Agent pipeline tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_agents() -> str:
    """
    List all available agents from the configured agent backend.

    Requires AGENT_BASE_URL and AGENT_TOKEN environment variables.
    """
    import urllib.request

    base_url = _agent_base_url()
    if not base_url:
        return json.dumps({"error": "AGENT_BASE_URL environment variable not set."})
    if not _agent_token():
        return json.dumps({"error": "AGENT_TOKEN environment variable not set."})

    try:
        req = urllib.request.Request(
            f"{base_url}/api/agents",
            headers=_agent_headers(),
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps(body, indent=2)


@mcp.tool()
def run_agent(
    agent_id: str,
    user_input: str,
) -> str:
    """
    Invoke an agent with the given input and return a thread ID for polling.

    The agent executes asynchronously. Call get_agent_result(thread_id) to
    retrieve the output when ready.

    Requires AGENT_BASE_URL and AGENT_TOKEN environment variables.

    Parameters
    ----------
    agent_id : str
        Agent identifier to invoke.
    user_input : str
        Task or question to send to the agent.
    """
    import urllib.request

    base_url = _agent_base_url()
    if not base_url:
        return json.dumps({"error": "AGENT_BASE_URL environment variable not set."})
    if not _agent_token():
        return json.dumps({"error": "AGENT_TOKEN environment variable not set."})

    payload = json.dumps({"agent_id": agent_id, "user_input": user_input}).encode()

    try:
        req = urllib.request.Request(
            f"{base_url}/api/run-agent",
            data=payload,
            headers=_agent_headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    thread_id = body.get("thread_id", "")
    return json.dumps({
        "thread_id": thread_id,
        "group_id": body.get("group_id", ""),
        "status": "started",
        "tip": f"Call get_agent_result(thread_id='{thread_id}') to poll for the result.",
    }, indent=2)


@mcp.tool()
def get_agent_result(
    thread_id: str,
    poll_seconds: int = 60,
) -> str:
    """
    Poll an agent run for results.

    Waits up to poll_seconds for the run to complete, returning the final
    answer and any artifacts produced.

    Requires AGENT_BASE_URL and AGENT_TOKEN environment variables.

    Parameters
    ----------
    thread_id : str
        Thread ID returned by run_agent.
    poll_seconds : int
        Maximum seconds to wait for completion. Default: 60.
    """
    import urllib.request

    base_url = _agent_base_url()
    if not base_url:
        return json.dumps({"error": "AGENT_BASE_URL environment variable not set."})

    deadline = time.time() + poll_seconds
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"{base_url}/api/missions/{thread_id}",
                headers=_agent_headers(),
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())

            if not body.get("success"):
                time.sleep(3)
                continue

            mission = body.get("data", {})
            logs = mission.get("log_entries", [])
            artifacts = mission.get("artifacts", [])

            has_answer = any(
                e.get("message_type") in ("answer", "completed") or e.get("status") == "Completed"
                for e in logs
            )
            if has_answer or artifacts:
                answer_text = ""
                for entry in reversed(logs):
                    if entry.get("message_type") in ("answer", "completed"):
                        answer_text = entry.get("content", "")
                        break
                if not answer_text and logs:
                    answer_text = logs[-1].get("content", "")

                return json.dumps({
                    "thread_id": thread_id,
                    "status": "completed",
                    "answer": answer_text,
                    "artifacts": [
                        {
                            "title": a.get("title", ""),
                            "type": a.get("type", ""),
                            "content": a.get("content", ""),
                        }
                        for a in artifacts
                    ],
                    "log_count": len(logs),
                }, indent=2)
        except Exception:
            pass
        time.sleep(4)

    return json.dumps({
        "thread_id": thread_id,
        "status": "pending",
        "note": f"Agent still running after {poll_seconds}s. Call get_agent_result again to continue polling.",
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
