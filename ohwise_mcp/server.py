"""
ohwise-mcp server — exposes OhWise graph tools as MCP tools.

Standalone tools (no OhWise backend required):
  - build_code_graph   : extract knowledge graph from a code repository
  - rank_code_nodes    : rank graph nodes by relevance to a query
  - search_code_graph  : find nodes by keyword or kind
  - build_doc_graph    : extract knowledge graph from documents
  - rank_doc_nodes     : rank document graph nodes by relevance

OhWise Studio tools (require OHWISE_URL + OHWISE_TOKEN env vars):
  - start_pipeline     : trigger a Studio coordinator run
  - get_pipeline_result: poll a coordinator run for results
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "ohwise",
    instructions=(
        "OhWise graph tools for AI/ML engineering. "
        "Use build_code_graph + rank_code_nodes to get focused, relationship-aware code context "
        "before editing. Use build_doc_graph + rank_doc_nodes for document retrieval. "
        "Use start_pipeline to delegate complex multi-step tasks to OhWise native agents."
    ),
)

# ---------------------------------------------------------------------------
# In-memory graph cache so multiple rank/search calls reuse the same graph
# ---------------------------------------------------------------------------
_graph_cache: dict[str, Any] = {}


def _graph_cache_key(path: str, graph_type: str) -> str:
    return f"{Path(path).resolve()}::{graph_type}"


# ---------------------------------------------------------------------------
# Code graph tools
# ---------------------------------------------------------------------------

@mcp.tool()
def build_code_graph(
    repo_path: str,
    graph_type: str = "all",
) -> str:
    """
    Build a knowledge graph from a code repository using codebase2graph.

    Returns a JSON summary: node count, edge count, graph_id, top entry points,
    high-fan-in nodes (core utilities), and high-fan-out nodes (orchestrators).

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
        return json.dumps({"error": "codebase2graph not installed. Run: pip install codebase2graph"})

    path = str(Path(repo_path).resolve())
    key = _graph_cache_key(path, graph_type)

    try:
        graph = build_graph(path, graph_type=graph_type)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    # Cache for subsequent rank/search calls
    _graph_cache[key] = graph

    # Build summary — .nodes and .edges are dicts keyed by ID
    all_nodes = list(graph.nodes.values())
    all_edges = list(graph.edges.values())

    nodes_by_kind: dict[str, int] = {}
    for n in all_nodes:
        kind = n.attributes.get("kind", "unknown")
        nodes_by_kind[kind] = nodes_by_kind.get(kind, 0) + 1

    edge_labels: dict[str, int] = {}
    for e in all_edges:
        edge_labels[e.label] = edge_labels.get(e.label, 0) + 1

    # Fan-in / fan-out
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
        "high_fan_in": [{"id": nid, "label": id_to_label.get(nid, nid), "incoming": cnt} for nid, cnt in top_fan_in],
        "high_fan_out": [{"id": nid, "label": id_to_label.get(nid, nid), "outgoing": cnt} for nid, cnt in top_fan_out],
        "tip": f"Call rank_code_nodes(graph_id='{key}', query='...') to get ranked context for a query.",
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
    Rank code graph nodes by relevance to a query using Personalized PageRank.

    Returns the top-k most relevant nodes with their content snippets —
    ideal for building focused LLM context before editing code.

    Parameters
    ----------
    query : str
        Natural language query (e.g. "authentication flow", "database connection").
    graph_id : str, optional
        ID returned by build_code_graph. If omitted, repo_path must be provided.
    repo_path : str, optional
        If graph_id is not provided, build the graph from this path first.
    graph_type : str
        Graph type to use when building from repo_path. Default: all.
    k : int
        Number of top nodes to return. Default: 10.
    """
    try:
        from code2graph import build_graph  # type: ignore
    except ImportError:
        return json.dumps({"error": "codebase2graph not installed. Run: pip install codebase2graph"})

    # Resolve graph
    graph = None
    if graph_id and graph_id in _graph_cache:
        graph = _graph_cache[graph_id]
    elif repo_path:
        path = str(Path(repo_path).resolve())
        key = _graph_cache_key(path, graph_type)
        if key in _graph_cache:
            graph = _graph_cache[key]
        else:
            try:
                graph = build_graph(path, graph_type=graph_type)
                _graph_cache[key] = graph
            except Exception as exc:
                return json.dumps({"error": str(exc)})
    else:
        return json.dumps({"error": "Provide graph_id (from build_code_graph) or repo_path."})

    if graph is None:
        return json.dumps({"error": "Graph not found. Call build_code_graph first."})

    # Simple relevance ranking: token overlap + structural centrality
    query_tokens = set(query.lower().split())
    node_scores: list[tuple[float, Any]] = []

    in_count: dict[str, int] = {}
    out_count: dict[str, int] = {}
    for e in graph.edges.values():
        out_count[e.from_id] = out_count.get(e.from_id, 0) + 1
        in_count[e.to_id] = in_count.get(e.to_id, 0) + 1

    for node in graph.nodes.values():
        text = f"{node.label} {node.id} {node.content or ''} {' '.join(str(v) for v in node.attributes.values())}".lower()
        token_overlap = sum(1 for t in query_tokens if t in text)
        centrality = in_count.get(node.id, 0) + out_count.get(node.id, 0)
        score = token_overlap * 3 + centrality * 0.1
        if token_overlap > 0 or centrality > 2:
            node_scores.append((score, node))

    node_scores.sort(key=lambda x: -x[0])
    top_nodes = node_scores[:k]

    results = []
    for score, node in top_nodes:
        results.append({
            "id": node.id,
            "label": node.label,
            "kind": node.attributes.get("kind", ""),
            "file": node.attributes.get("file", ""),
            "line": node.attributes.get("line", ""),
            "score": round(score, 2),
            "content": (node.content or "")[:400],
        })

    return json.dumps({
        "query": query,
        "top_nodes": results,
        "total_matched": len(node_scores),
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
    Search code graph nodes by keyword or filter by kind.

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
    try:
        from code2graph import build_graph  # type: ignore
    except ImportError:
        return json.dumps({"error": "codebase2graph not installed. Run: pip install codebase2graph"})

    graph = None
    if graph_id and graph_id in _graph_cache:
        graph = _graph_cache[graph_id]
    elif repo_path:
        path = str(Path(repo_path).resolve())
        key = _graph_cache_key(path, graph_type)
        if key in _graph_cache:
            graph = _graph_cache[key]
        else:
            try:
                graph = build_graph(path, graph_type=graph_type)
                _graph_cache[key] = graph
            except Exception as exc:
                return json.dumps({"error": str(exc)})
    else:
        return json.dumps({"error": "Provide graph_id or repo_path."})

    if graph is None:
        return json.dumps({"error": "Graph not found. Call build_code_graph first."})

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
# Document graph tools
# ---------------------------------------------------------------------------

@mcp.tool()
def build_doc_graph(
    paths: str,
    graph_type: str = "knowledge",
) -> str:
    """
    Build a knowledge graph from documents using docs2graph.

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
        from docs2graph import extract_knowledge_graph, extract_schema_graph, extract_decision_graph, load_document  # type: ignore
    except ImportError:
        return json.dumps({"error": "docs2graph not installed. Run: pip install docs2graph"})

    path_list = [p.strip() for p in paths.split(",")]
    texts = []
    loaded_paths = []
    errors = []

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

    # docs2graph returns a dict: {"nodes": [...], "edges": [...]}
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

    Returns the top-k most relevant nodes with their content — use as
    LLM context for document question answering.

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
        return json.dumps({"error": f"Graph '{graph_id}' not found in cache. Call build_doc_graph first."})

    try:
        # personalized_page_rank(query, graph, k=k) — query is first arg
        ranked = personalized_page_rank(query, graph, k=k)
        # ranked is a dict {"nodes": [...], "edges": [...]}
        results = []
        for node in ranked.get("nodes", []):
            results.append({
                "id": node["id"],
                "label": node["label"],
                "kind": (node.get("attributes") or {}).get("kind", ""),
                "content": (node.get("content") or "")[:600],
            })
        return json.dumps({"query": query, "top_nodes": results}, indent=2)
    except Exception as exc:
        # Fallback: token overlap ranking over docs2graph dict nodes
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
# OhWise Studio pipeline tools
# ---------------------------------------------------------------------------

def _ohwise_headers() -> dict[str, str]:
    token = os.environ.get("OHWISE_TOKEN", "")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


@mcp.tool()
def start_pipeline(
    user_input: str,
    agent_ids: str = "",
    coordinator_id: str = "",
) -> str:
    """
    Trigger an OhWise Studio coordinator pipeline.

    Delegates a complex task to OhWise native agents. The coordinator selects
    and dispatches specialized agents, gathers results, and returns a synthesized answer.

    Requires environment variables: OHWISE_URL, OHWISE_TOKEN.

    Parameters
    ----------
    user_input : str
        The task or question for the pipeline.
    agent_ids : str
        Comma-separated agent IDs to make available. Leave empty to use defaults.
    coordinator_id : str
        Coordinator agent ID. Leave empty to use the default coordinator.
    """
    import urllib.request

    base_url = os.environ.get("OHWISE_URL", "").rstrip("/")
    if not base_url:
        return json.dumps({"error": "OHWISE_URL environment variable not set."})
    if not os.environ.get("OHWISE_TOKEN"):
        return json.dumps({"error": "OHWISE_TOKEN environment variable not set."})

    import uuid
    thread_id = str(uuid.uuid4())
    group_id = str(uuid.uuid4())

    payload = json.dumps({
        "user_input": user_input,
        "thread_id": thread_id,
        "group_id": group_id,
        "agent_ids": [a.strip() for a in agent_ids.split(",") if a.strip()],
        "coordinator_agent_id": coordinator_id or None,
    }).encode()

    try:
        req = urllib.request.Request(
            f"{base_url}/api/internal/coordinator-dispatch",
            data=payload,
            headers=_ohwise_headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    return json.dumps({
        "thread_id": thread_id,
        "group_id": group_id,
        "status": "started",
        "tip": f"Call get_pipeline_result(thread_id='{thread_id}') to poll for the result.",
    }, indent=2)


@mcp.tool()
def get_pipeline_result(
    thread_id: str,
    poll_seconds: int = 30,
) -> str:
    """
    Poll an OhWise Studio pipeline for results.

    Waits up to poll_seconds for the pipeline to complete, then returns
    the latest messages from the coordinator thread.

    Requires environment variables: OHWISE_URL, OHWISE_TOKEN.

    Parameters
    ----------
    thread_id : str
        Thread ID returned by start_pipeline.
    poll_seconds : int
        How long to wait for completion. Default: 30.
    """
    import urllib.request

    base_url = os.environ.get("OHWISE_URL", "").rstrip("/")
    if not base_url:
        return json.dumps({"error": "OHWISE_URL environment variable not set."})

    deadline = time.time() + poll_seconds
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"{base_url}/api/messages?thread_id={thread_id}&limit=10",
                headers=_ohwise_headers(),
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
            messages = body.get("data", [])
            # Look for completed coordinator message
            for msg in reversed(messages):
                if msg.get("sender_type") == 1 and msg.get("message_status") == "Completed":
                    return json.dumps({
                        "thread_id": thread_id,
                        "status": "completed",
                        "content": msg.get("content", ""),
                        "artifacts": msg.get("artifacts", []),
                    }, indent=2)
        except Exception:
            pass
        time.sleep(3)

    return json.dumps({
        "thread_id": thread_id,
        "status": "pending",
        "note": f"Pipeline still running after {poll_seconds}s. Call again to check.",
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
