"""Tests for ohwise-mcp server tools."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_repo(tmp_path: Path) -> Path:
    """Create a minimal Python repo for testing."""
    (tmp_path / "auth.py").write_text(
        "def login(user, pw):\n    return db_query(user)\n\ndef db_query(q):\n    return q\n"
    )
    (tmp_path / "main.py").write_text(
        "from auth import login\n\ndef run():\n    login('admin', 'pass')\n"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# build_code_graph
# ---------------------------------------------------------------------------

class TestBuildCodeGraph:
    def test_returns_json_with_node_count(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph

        repo = _make_temp_repo(tmp_path)
        result = json.loads(build_code_graph(str(repo), graph_type="call"))
        assert "node_count" in result
        assert result["node_count"] > 0
        assert "graph_id" in result

    def test_invalid_path_returns_error(self):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph

        result = json.loads(build_code_graph("/nonexistent/path/xyz"))
        assert "error" in result

    def test_graph_cached_after_build(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph, _graph_cache

        repo = _make_temp_repo(tmp_path)
        result = json.loads(build_code_graph(str(repo), graph_type="folder"))
        graph_id = result["graph_id"]
        assert graph_id in _graph_cache


# ---------------------------------------------------------------------------
# rank_code_nodes
# ---------------------------------------------------------------------------

class TestRankCodeNodes:
    def test_returns_ranked_results(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph, rank_code_nodes

        repo = _make_temp_repo(tmp_path)
        build_result = json.loads(build_code_graph(str(repo), graph_type="call"))
        graph_id = build_result["graph_id"]

        result = json.loads(rank_code_nodes("login authentication", graph_id=graph_id, k=5))
        assert "top_nodes" in result
        assert isinstance(result["top_nodes"], list)

    def test_missing_graph_returns_error(self):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import rank_code_nodes

        result = json.loads(rank_code_nodes("query", graph_id="nonexistent::id"))
        assert "error" in result

    def test_no_graph_id_or_path_returns_error(self):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import rank_code_nodes

        result = json.loads(rank_code_nodes("query"))
        assert "error" in result


# ---------------------------------------------------------------------------
# search_code_graph
# ---------------------------------------------------------------------------

class TestSearchCodeGraph:
    def test_finds_nodes_by_keyword(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph, search_code_graph

        repo = _make_temp_repo(tmp_path)
        build_result = json.loads(build_code_graph(str(repo), graph_type="call"))
        graph_id = build_result["graph_id"]

        result = json.loads(search_code_graph("login", graph_id=graph_id))
        assert "results" in result
        assert isinstance(result["results"], list)

    def test_empty_keyword_returns_results(self, tmp_path):
        pytest.importorskip("code2graph")
        from ohwise_mcp.server import build_code_graph, search_code_graph

        repo = _make_temp_repo(tmp_path)
        build_result = json.loads(build_code_graph(str(repo), graph_type="folder"))
        graph_id = build_result["graph_id"]

        result = json.loads(search_code_graph("auth", graph_id=graph_id))
        assert "results" in result


# ---------------------------------------------------------------------------
# build_doc_graph
# ---------------------------------------------------------------------------

class TestBuildDocGraph:
    def test_builds_from_markdown_file(self, tmp_path):
        pytest.importorskip("docs2graph")
        from ohwise_mcp.server import build_doc_graph

        md = tmp_path / "notes.md"
        md.write_text("# Architecture\n\nThe system uses a DAG-based pipeline for agent coordination.\n\n## Components\n\nCoordinator dispatches tasks to specialized agents.\n")

        result = json.loads(build_doc_graph(str(md)))
        assert "node_count" in result
        assert result["node_count"] >= 0
        assert "graph_id" in result

    def test_missing_path_returns_error(self):
        pytest.importorskip("docs2graph")
        from ohwise_mcp.server import build_doc_graph

        result = json.loads(build_doc_graph("/nonexistent/file.md"))
        assert "error" in result


# ---------------------------------------------------------------------------
# Studio pipeline tools (no backend — just check error messages)
# ---------------------------------------------------------------------------

class TestRunAgent:
    def test_missing_env_returns_error(self):
        from ohwise_mcp.server import run_agent

        env_backup = os.environ.pop("AGENT_BASE_URL", None)
        ohwise_backup = os.environ.pop("OHWISE_URL", None)
        try:
            result = json.loads(run_agent("test task"))
            assert "error" in result
            assert "AGENT_BASE_URL" in result["error"]
        finally:
            if env_backup:
                os.environ["AGENT_BASE_URL"] = env_backup
            if ohwise_backup:
                os.environ["OHWISE_URL"] = ohwise_backup


class TestGetAgentResult:
    def test_missing_env_returns_error(self):
        from ohwise_mcp.server import get_agent_result

        env_backup = os.environ.pop("AGENT_BASE_URL", None)
        ohwise_backup = os.environ.pop("OHWISE_URL", None)
        try:
            result = json.loads(get_agent_result("test-thread-id", poll_seconds=1))
            assert "error" in result or "status" in result
        finally:
            if env_backup:
                os.environ["AGENT_BASE_URL"] = env_backup
            if ohwise_backup:
                os.environ["OHWISE_URL"] = ohwise_backup
